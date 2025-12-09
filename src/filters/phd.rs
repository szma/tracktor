//! Gaussian Mixture Probability Hypothesis Density (GM-PHD) Filter
//!
//! Implementation of the GM-PHD filter for multi-target tracking.
//!
//! Reference: Vo, B.-N., & Ma, W.-K. (2006). "The Gaussian Mixture
//! Probability Hypothesis Density Filter"

use ::core::marker::PhantomData;
use nalgebra::RealField;
use num_traits::Float;

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use crate::models::{BirthModel, ClutterModel, ObservationModel, TransitionModel};
use crate::types::gaussian::{innovation_likelihood, GaussianMixture, GaussianState};
use crate::types::spaces::{ComputeInnovation, Measurement, StateVector};
use crate::types::transforms::{compute_innovation_covariance, compute_kalman_gain, joseph_update};

// ============================================================================
// Filter Phase Markers
// ============================================================================

/// Marker type indicating a predicted filter state.
#[derive(Debug, Clone, Copy)]
pub struct Predicted;

/// Marker type indicating an updated filter state.
#[derive(Debug, Clone, Copy)]
pub struct Updated;

// ============================================================================
// Update Result
// ============================================================================

/// Statistics from a filter update step, reporting any numerical issues.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, Default)]
pub struct UpdateStats {
    /// Number of components where Kalman gain computation failed (singular innovation covariance)
    pub singular_covariance_count: usize,
    /// Number of components where likelihood computation returned zero or failed
    pub zero_likelihood_count: usize,
}

#[cfg(feature = "alloc")]
impl UpdateStats {
    /// Returns true if any numerical issues were encountered.
    pub fn has_issues(&self) -> bool {
        self.singular_covariance_count > 0 || self.zero_likelihood_count > 0
    }
}

// ============================================================================
// PHD Filter State
// ============================================================================

/// The state of a GM-PHD filter at a particular phase.
///
/// The `Phase` parameter encodes whether this is a predicted or updated state,
/// ensuring correct operation ordering at compile time.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct PhdFilterState<T: RealField, const N: usize, Phase> {
    /// Gaussian mixture representing the PHD
    pub mixture: GaussianMixture<T, N>,
    /// Current time step
    pub time_step: u32,
    /// Phase marker
    _phase: PhantomData<Phase>,
}

#[cfg(feature = "alloc")]
impl<T: RealField + Copy, const N: usize> PhdFilterState<T, N, Updated> {
    /// Creates a new filter state in the updated phase.
    pub fn new() -> Self {
        Self {
            mixture: GaussianMixture::new(),
            time_step: 0,
            _phase: PhantomData,
        }
    }

    /// Creates a filter state from an initial mixture.
    pub fn from_mixture(mixture: GaussianMixture<T, N>) -> Self {
        Self {
            mixture,
            time_step: 0,
            _phase: PhantomData,
        }
    }

    /// Returns the expected number of targets.
    pub fn expected_target_count(&self) -> T {
        self.mixture.total_weight()
    }

    /// Predicts the PHD to the next time step.
    ///
    /// This method consumes the updated state and returns a predicted state.
    pub fn predict<Trans, Birth>(
        self,
        transition_model: &Trans,
        birth_model: &Birth,
        dt: T,
    ) -> PhdFilterState<T, N, Predicted>
    where
        Trans: TransitionModel<T, N>,
        Birth: BirthModel<T, N>,
    {
        let transition_matrix = transition_model.transition_matrix(dt);
        let process_noise = transition_model.process_noise(dt);

        let mut predicted = GaussianMixture::with_capacity(
            self.mixture.len() + birth_model.birth_components().len(),
        );

        // Predict surviving components
        for component in self.mixture.iter() {
            let p_s = transition_model.survival_probability(&component.mean);

            let predicted_component = GaussianState {
                weight: component.weight * p_s,
                mean: transition_matrix.apply_state(&component.mean),
                covariance: transition_matrix
                    .propagate_covariance(&component.covariance)
                    .add(&process_noise),
            };

            predicted.push(predicted_component);
        }

        // Add birth components
        for birth_component in birth_model.birth_components() {
            predicted.push(birth_component.clone());
        }

        PhdFilterState {
            mixture: predicted,
            time_step: self.time_step + 1,
            _phase: PhantomData,
        }
    }
}

#[cfg(feature = "alloc")]
impl<T: RealField + Copy, const N: usize> Default for PhdFilterState<T, N, Updated> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "alloc")]
impl<T: RealField + Float + Copy, const N: usize> PhdFilterState<T, N, Predicted> {
    /// Updates the PHD with a set of measurements.
    ///
    /// This method consumes the predicted state and returns an updated state along
    /// with statistics about any numerical issues encountered during the update.
    ///
    /// # Returns
    /// A tuple of (updated_state, stats) where stats contains counts of any
    /// numerical issues (singular covariances, zero likelihoods).
    pub fn update_with_stats<const M: usize, Obs, Clutter>(
        self,
        measurements: &[Measurement<T, M>],
        observation_model: &Obs,
        clutter_model: &Clutter,
    ) -> (PhdFilterState<T, N, Updated>, UpdateStats)
    where
        Obs: ObservationModel<T, N, M>,
        Clutter: ClutterModel<T, M>,
    {
        let obs_matrix = observation_model.observation_matrix();
        let meas_noise = observation_model.measurement_noise();
        let mut stats = UpdateStats::default();

        // Pre-compute per-component data
        let component_data: Vec<_> = self
            .mixture
            .iter()
            .map(|c| {
                let p_d = observation_model.detection_probability(&c.mean);
                let predicted_meas = obs_matrix.observe(&c.mean);
                let innovation_cov =
                    compute_innovation_covariance(&c.covariance, &obs_matrix, &meas_noise);
                let kalman_gain = compute_kalman_gain(&c.covariance, &obs_matrix, &innovation_cov);

                if kalman_gain.is_none() {
                    stats.singular_covariance_count += 1;
                }

                (p_d, predicted_meas, innovation_cov, kalman_gain)
            })
            .collect();

        let mut updated =
            GaussianMixture::with_capacity(self.mixture.len() * (measurements.len() + 1));

        // Missed detection components
        for (i, component) in self.mixture.iter().enumerate() {
            let p_d = component_data[i].0;
            let missed_weight = component.weight * (T::one() - p_d);

            if missed_weight > T::zero() {
                updated.push(GaussianState {
                    weight: missed_weight,
                    mean: component.mean,
                    covariance: component.covariance,
                });
            }
        }

        // Detection components (one for each component-measurement pair)
        for measurement in measurements {
            // Compute denominator for weight normalization
            let mut weight_sum = clutter_model.clutter_intensity(measurement);

            let detection_weights: Vec<T> = self
                .mixture
                .iter()
                .enumerate()
                .map(|(i, c)| {
                    let (p_d, ref predicted_meas, ref innovation_cov, _) = component_data[i];

                    let innovation = measurement.innovation(*predicted_meas);
                    let likelihood = innovation_likelihood(&innovation, innovation_cov.as_matrix());

                    // Track zero likelihoods (can indicate numerical issues)
                    if likelihood <= T::zero() {
                        stats.zero_likelihood_count += 1;
                    }

                    let detection_weight = p_d * c.weight * likelihood;
                    weight_sum += detection_weight;

                    detection_weight
                })
                .collect();

            // Add updated components
            for (i, component) in self.mixture.iter().enumerate() {
                let (_, ref predicted_meas, _, ref kalman_gain) = component_data[i];

                if let Some(ref gain) = kalman_gain {
                    let normalized_weight = detection_weights[i] / weight_sum;

                    if normalized_weight > T::zero() {
                        let innovation = measurement.innovation(*predicted_meas);
                        let correction = gain.correct(&innovation);
                        let updated_mean = StateVector::from_svector(
                            component.mean.as_svector() + correction.as_svector(),
                        );
                        let updated_cov =
                            joseph_update(&component.covariance, gain, &obs_matrix, &meas_noise);

                        updated.push(GaussianState {
                            weight: normalized_weight,
                            mean: updated_mean,
                            covariance: updated_cov,
                        });
                    }
                }
                // Note: If kalman_gain is None, the component is skipped for this measurement.
                // This is tracked in stats.singular_covariance_count above.
            }
        }

        (
            PhdFilterState {
                mixture: updated,
                time_step: self.time_step,
                _phase: PhantomData,
            },
            stats,
        )
    }

    /// Updates the PHD with a set of measurements.
    ///
    /// This method consumes the predicted state and returns an updated state.
    /// For access to numerical issue statistics, use `update_with_stats` instead.
    pub fn update<const M: usize, Obs, Clutter>(
        self,
        measurements: &[Measurement<T, M>],
        observation_model: &Obs,
        clutter_model: &Clutter,
    ) -> PhdFilterState<T, N, Updated>
    where
        Obs: ObservationModel<T, N, M>,
        Clutter: ClutterModel<T, M>,
    {
        self.update_with_stats(measurements, observation_model, clutter_model)
            .0
    }

    /// Returns the expected number of targets (before update).
    pub fn expected_target_count(&self) -> T {
        self.mixture.total_weight()
    }
}

// ============================================================================
// GM-PHD Filter
// ============================================================================

/// Complete GM-PHD filter with models.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct GmPhdFilter<T, Trans, Obs, Clutter, Birth, const N: usize, const M: usize>
where
    T: RealField,
    Trans: TransitionModel<T, N>,
    Obs: ObservationModel<T, N, M>,
    Clutter: ClutterModel<T, M>,
    Birth: BirthModel<T, N>,
{
    /// Transition model
    pub transition: Trans,
    /// Observation model
    pub observation: Obs,
    /// Clutter model
    pub clutter: Clutter,
    /// Birth model
    pub birth: Birth,
    _marker: PhantomData<T>,
}

#[cfg(feature = "alloc")]
impl<T, Trans, Obs, Clutter, Birth, const N: usize, const M: usize>
    GmPhdFilter<T, Trans, Obs, Clutter, Birth, N, M>
where
    T: RealField + Float + Copy,
    Trans: TransitionModel<T, N>,
    Obs: ObservationModel<T, N, M>,
    Clutter: ClutterModel<T, M>,
    Birth: BirthModel<T, N>,
{
    /// Creates a new GM-PHD filter with the specified models.
    pub fn new(transition: Trans, observation: Obs, clutter: Clutter, birth: Birth) -> Self {
        Self {
            transition,
            observation,
            clutter,
            birth,
            _marker: PhantomData,
        }
    }

    /// Creates an initial filter state.
    pub fn initial_state(&self) -> PhdFilterState<T, N, Updated> {
        PhdFilterState::new()
    }

    /// Creates an initial filter state from prior components.
    pub fn initial_state_from(
        &self,
        components: Vec<GaussianState<T, N>>,
    ) -> PhdFilterState<T, N, Updated> {
        PhdFilterState::from_mixture(GaussianMixture::from_components(components))
    }

    /// Runs one complete predict-update cycle with statistics.
    ///
    /// Returns both the updated state and statistics about any numerical issues.
    pub fn step_with_stats(
        &self,
        state: PhdFilterState<T, N, Updated>,
        measurements: &[Measurement<T, M>],
        dt: T,
    ) -> (PhdFilterState<T, N, Updated>, UpdateStats) {
        let predicted = state.predict(&self.transition, &self.birth, dt);
        predicted.update_with_stats(measurements, &self.observation, &self.clutter)
    }

    /// Runs one complete predict-update cycle.
    ///
    /// For access to numerical issue statistics, use `step_with_stats` instead.
    pub fn step(
        &self,
        state: PhdFilterState<T, N, Updated>,
        measurements: &[Measurement<T, M>],
        dt: T,
    ) -> PhdFilterState<T, N, Updated> {
        self.step_with_stats(state, measurements, dt).0
    }
}

// ============================================================================
// State Extraction (Legacy - prefer utils::extraction)
// ============================================================================

/// Extracts target states from a PHD filter state.
///
/// This is a simple extraction method. For more sophisticated extraction
/// strategies (local maxima, expected count, etc.), use `crate::utils::extract_targets`
/// with an `ExtractionConfig`.
#[cfg(feature = "alloc")]
#[deprecated(
    since = "0.2.0",
    note = "Use crate::utils::extract_targets with ExtractionConfig instead"
)]
pub fn extract_states<T: RealField + Float + Copy, const N: usize, Phase>(
    state: &PhdFilterState<T, N, Phase>,
    threshold: T,
) -> Vec<(StateVector<T, N>, T)> {
    state
        .mixture
        .iter()
        .filter(|c| c.weight >= threshold)
        .map(|c| (c.mean, c.weight))
        .collect()
}

/// Extracts states using MAP (Maximum A Posteriori) estimation.
///
/// Returns the N states with highest weights.
///
/// For more sophisticated extraction strategies, use `crate::utils::extract_targets`
/// with `ExtractionConfig::top_n()`.
#[cfg(feature = "alloc")]
#[deprecated(
    since = "0.2.0",
    note = "Use crate::utils::extract_targets with ExtractionConfig::top_n() instead"
)]
pub fn extract_states_map<T: RealField + Float + Copy, const N: usize, Phase>(
    state: &PhdFilterState<T, N, Phase>,
    n_targets: usize,
) -> Vec<(StateVector<T, N>, T)> {
    let mut components: Vec<_> = state.mixture.iter().map(|c| (c.mean, c.weight)).collect();

    // Sort by weight descending
    components.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));

    components.into_iter().take(n_targets).collect()
}

/// Estimates the number of targets by rounding total weight.
///
/// For a more complete API, use `crate::utils::estimate_cardinality`.
#[cfg(feature = "alloc")]
#[deprecated(
    since = "0.2.0",
    note = "Use crate::utils::estimate_cardinality instead"
)]
pub fn estimate_target_count<T: RealField + Float + Copy, const N: usize, Phase>(
    state: &PhdFilterState<T, N, Phase>,
) -> usize {
    let total = state.mixture.total_weight();
    num_traits::Float::round(total).to_usize().unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{ConstantVelocity2D, FixedBirthModel, PositionSensor2D, UniformClutter2D};
    use crate::types::spaces::StateCovariance;

    #[cfg(feature = "alloc")]
    #[test]
    fn test_phd_filter_creation() {
        let transition = ConstantVelocity2D::new(1.0_f64, 0.99);
        let observation = PositionSensor2D::new(1.0, 0.9);
        let clutter = UniformClutter2D::new(10.0, (0.0, 100.0), (0.0, 100.0));
        let birth = FixedBirthModel::<f64, 4>::new();

        let filter = GmPhdFilter::new(transition, observation, clutter, birth);
        let state = filter.initial_state();

        assert_eq!(state.mixture.len(), 0);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_phd_predict() {
        let transition = ConstantVelocity2D::new(1.0_f64, 0.99);
        let observation = PositionSensor2D::new(1.0, 0.9);
        let clutter = UniformClutter2D::new(10.0, (0.0, 100.0), (0.0, 100.0));

        let mut birth = FixedBirthModel::<f64, 4>::new();
        birth.add_birth_location(
            0.1,
            StateVector::from_array([50.0, 50.0, 0.0, 0.0]),
            StateCovariance::identity(),
        );

        let filter = GmPhdFilter::new(transition, observation, clutter, birth);

        // Start with one component
        let initial = filter.initial_state_from(vec![GaussianState::new(
            1.0,
            StateVector::from_array([10.0, 20.0, 1.0, 2.0]),
            StateCovariance::identity(),
        )]);

        let predicted = initial.predict(&filter.transition, &filter.birth, 1.0);

        // Should have original (survived) + birth components
        assert_eq!(predicted.mixture.len(), 2);

        // Check that position was predicted correctly
        let target = &predicted.mixture.components[0];
        assert!((target.mean.index(0) - 11.0).abs() < 1e-5); // x + vx*dt
        assert!((target.mean.index(1) - 22.0).abs() < 1e-5); // y + vy*dt
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_phd_update() {
        let transition = ConstantVelocity2D::new(0.1_f64, 0.99);
        let observation = PositionSensor2D::new(1.0, 0.9);
        let clutter = UniformClutter2D::new(1.0, (0.0, 100.0), (0.0, 100.0));
        let birth = FixedBirthModel::<f64, 4>::new();

        let filter = GmPhdFilter::new(transition, observation, clutter, birth);

        // Start with one component
        let initial = filter.initial_state_from(vec![GaussianState::new(
            1.0,
            StateVector::from_array([10.0, 20.0, 0.0, 0.0]),
            StateCovariance::from_matrix(nalgebra::matrix![
                1.0, 0.0, 0.0, 0.0;
                0.0, 1.0, 0.0, 0.0;
                0.0, 0.0, 1.0, 0.0;
                0.0, 0.0, 0.0, 1.0
            ]),
        )]);

        let predicted = initial.predict(&filter.transition, &filter.birth, 1.0);

        // Update with a measurement near the target
        let measurements = [Measurement::from_array([10.5, 20.5])];
        let updated = predicted.update(&measurements, &filter.observation, &filter.clutter);

        // Should have components for missed detection and detection
        assert!(updated.mixture.len() >= 1);

        // Total weight should still be approximately 1 (one target)
        let total_weight = updated.expected_target_count();
        assert!(total_weight > 0.5 && total_weight < 1.5);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_state_extraction() {
        let mut mixture = GaussianMixture::new();
        mixture.push(GaussianState::new(
            0.8,
            StateVector::from_array([10.0, 20.0, 1.0, 2.0]),
            StateCovariance::identity(),
        ));
        mixture.push(GaussianState::new(
            0.3,
            StateVector::from_array([50.0, 60.0, 0.0, 0.0]),
            StateCovariance::identity(),
        ));

        let state: PhdFilterState<f64, 4, Updated> = PhdFilterState::from_mixture(mixture);

        // Extract with threshold 0.5
        let targets = extract_states(&state, 0.5);
        assert_eq!(targets.len(), 1);
        assert!((targets[0].0.index(0) - 10.0).abs() < 1e-10);

        // Extract MAP with n=2
        let targets = extract_states_map(&state, 2);
        assert_eq!(targets.len(), 2);
    }
}
