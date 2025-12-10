//! Single-sensor LMB Filter
//!
//! Implementation of the Labeled Multi-Bernoulli filter for multi-target
//! tracking with track identity preservation.
//!
//! # References
//!
//! - Reuter, S., Vo, B.-T., Vo, B.-N., & Dietmayer, K. (2014). "The Labeled
//!   Multi-Bernoulli Filter." *IEEE Transactions on Signal Processing*, 62(12),
//!   3246-3260.
//!
//! - Vo, B.-T., & Vo, B.-N. (2013). "Labeled Random Finite Sets and Multi-Object
//!   Conjugate Priors." *IEEE Transactions on Signal Processing*, 61(13),
//!   3460-3475.

use core::marker::PhantomData;
use nalgebra::{ComplexField, RealField};
use num_traits::Float;

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use super::cardinality::map_cardinality_estimate;
use super::types::{LmbTrack, LmbTrackSet, PosteriorGrid};
#[allow(unused_imports)]
use super::updaters::TrackUpdater;
use crate::filters::phd::{Predicted, UpdateStats, Updated};
use crate::models::{ClutterModel, NonlinearObservationModel, ObservationModel, TransitionModel};
use crate::types::gaussian::GaussianState;
use crate::types::labels::{BernoulliTrack, Label, LabelGenerator};
use crate::types::spaces::{ComputeInnovation, Measurement, StateCovariance, StateVector};
use crate::types::transforms::{compute_innovation_covariance, compute_kalman_gain, joseph_update};

// ============================================================================
// LBP Configuration
// ============================================================================

/// Configuration for Loopy Belief Propagation (LBP) data association.
#[derive(Debug, Clone)]
pub struct LbpConfig<T: RealField> {
    /// Maximum number of LBP iterations
    pub max_iterations: usize,
    /// Convergence tolerance (maximum message change between iterations)
    pub tolerance: T,
}

impl<T: RealField + Float> LbpConfig<T> {
    /// Creates the default LBP configuration.
    ///
    /// Default values:
    /// - `max_iterations`: 50
    /// - `tolerance`: 1e-6
    pub fn default_config() -> Self {
        Self {
            max_iterations: 50,
            tolerance: T::from_f64(1e-6).unwrap(),
        }
    }

    /// Creates a custom LBP configuration.
    pub fn new(max_iterations: usize, tolerance: T) -> Self {
        Self {
            max_iterations,
            tolerance,
        }
    }
}

impl<T: RealField + Float> Default for LbpConfig<T> {
    fn default() -> Self {
        Self::default_config()
    }
}

// ============================================================================
// Labeled Birth Model Trait
// ============================================================================

/// Birth model that generates labeled tracks for LMB filters.
///
/// Extends the basic [`BirthModel`] with the ability to generate
/// labeled Bernoulli tracks with unique identifiers.
#[cfg(feature = "alloc")]
pub trait LabeledBirthModel<T: RealField, const N: usize> {
    /// Generate birth tracks with labels at the current time step.
    ///
    /// The label generator should be used to create unique labels
    /// for each new track.
    fn birth_tracks(&self, label_gen: &mut LabelGenerator) -> Vec<BernoulliTrack<T, N>>;

    /// Returns the expected number of birth targets per time step.
    fn expected_birth_count(&self) -> T;
}

/// A birth model that produces no birth tracks.
///
/// Useful for testing, benchmarking, or scenarios where track
/// initialization is handled externally.
///
/// # Example
///
/// ```
/// use tracktor::filters::lmb::{LabeledBirthModel, NoBirthModel};
/// use tracktor::types::labels::LabelGenerator;
///
/// let birth = NoBirthModel;
/// let mut label_gen = LabelGenerator::new();
///
/// // Type annotation needed to specify state dimension
/// let tracks: Vec<tracktor::types::labels::BernoulliTrack<f64, 4>> =
///     birth.birth_tracks(&mut label_gen);
/// assert!(tracks.is_empty());
///
/// // Use turbofish to specify types for expected_birth_count
/// assert_eq!(LabeledBirthModel::<f64, 4>::expected_birth_count(&birth), 0.0);
/// ```
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, Copy, Default)]
pub struct NoBirthModel;

#[cfg(feature = "alloc")]
impl<T: RealField, const N: usize> LabeledBirthModel<T, N> for NoBirthModel {
    fn birth_tracks(&self, _label_gen: &mut LabelGenerator) -> Vec<BernoulliTrack<T, N>> {
        Vec::new()
    }

    fn expected_birth_count(&self) -> T {
        T::zero()
    }
}

// ============================================================================
// LMB Filter State
// ============================================================================

/// The state of an LMB filter at a particular phase.
///
/// The `Phase` parameter encodes whether this is a predicted or updated state,
/// ensuring correct operation ordering at compile time.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct LmbFilterState<T: RealField, const N: usize, Phase> {
    /// LMB track set
    pub tracks: LmbTrackSet<T, N>,
    /// Label generator for new tracks
    pub label_gen: LabelGenerator,
    /// Current time step
    pub time_step: u32,
    /// Phase marker
    _phase: PhantomData<Phase>,
}

#[cfg(feature = "alloc")]
impl<T: RealField + Copy, const N: usize> LmbFilterState<T, N, Updated> {
    /// Creates a new filter state in the updated phase.
    pub fn new() -> Self {
        Self {
            tracks: LmbTrackSet::new(),
            label_gen: LabelGenerator::new(),
            time_step: 0,
            _phase: PhantomData,
        }
    }

    /// Creates a filter state from initial tracks.
    pub fn from_tracks(tracks: Vec<LmbTrack<T, N>>) -> Self {
        Self {
            tracks: LmbTrackSet { tracks },
            label_gen: LabelGenerator::new(),
            time_step: 0,
            _phase: PhantomData,
        }
    }

    /// Creates an Updated state from components (for internal use).
    pub(crate) fn from_components(
        tracks: LmbTrackSet<T, N>,
        label_gen: LabelGenerator,
        time_step: u32,
    ) -> Self {
        Self {
            tracks,
            label_gen,
            time_step,
            _phase: PhantomData,
        }
    }

    /// Returns the expected number of targets.
    pub fn expected_target_count(&self) -> T {
        self.tracks.expected_cardinality()
    }

    /// Returns the number of tracks.
    pub fn num_tracks(&self) -> usize {
        self.tracks.len()
    }

    /// Predicts the LMB density to the next time step.
    ///
    /// This method consumes the updated state and returns a predicted state.
    /// Applies survival probability and adds birth tracks.
    pub fn predict<Trans, Birth>(
        mut self,
        transition_model: &Trans,
        birth_model: &Birth,
        dt: T,
    ) -> LmbFilterState<T, N, Predicted>
    where
        Trans: TransitionModel<T, N>,
        Birth: LabeledBirthModel<T, N>,
    {
        let transition_matrix = transition_model.transition_matrix(dt);
        let process_noise = transition_model.process_noise(dt);

        // Advance label generator to new time step
        self.label_gen.advance_time();

        // Predict surviving tracks
        for track in self.tracks.iter_mut() {
            // Update existence with survival probability
            // Use first component's mean for state-dependent survival
            if let Some(first_component) = track.components.iter().next() {
                let p_s = transition_model.survival_probability(&first_component.mean);
                track.existence *= p_s;
            }

            // Predict each component through the motion model
            for component in track.components.iter_mut() {
                component.mean = transition_matrix.apply_state(&component.mean);
                component.covariance = transition_matrix
                    .propagate_covariance(&component.covariance)
                    .add(&process_noise);
            }
        }

        // Add birth tracks
        let birth_tracks = birth_model.birth_tracks(&mut self.label_gen);
        for bt in birth_tracks {
            self.tracks.push(LmbTrack::from_bernoulli(bt));
        }

        LmbFilterState {
            tracks: self.tracks,
            label_gen: self.label_gen,
            time_step: self.time_step + 1,
            _phase: PhantomData,
        }
    }
}

#[cfg(feature = "alloc")]
impl<T: RealField + Copy, const N: usize> Default for LmbFilterState<T, N, Updated> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "alloc")]
impl<T: RealField + Float + Copy, const N: usize> LmbFilterState<T, N, Predicted> {
    /// Creates a Predicted state from components (for internal use).
    pub(crate) fn from_components(
        tracks: LmbTrackSet<T, N>,
        label_gen: LabelGenerator,
        time_step: u32,
    ) -> Self {
        Self {
            tracks,
            label_gen,
            time_step,
            _phase: PhantomData,
        }
    }

    /// Updates the LMB density with measurements.
    ///
    /// This method consumes the predicted state and returns an updated state.
    /// Uses Loopy Belief Propagation for data association with default parameters.
    pub fn update<const M: usize, Obs, Clutter>(
        self,
        measurements: &[Measurement<T, M>],
        observation_model: &Obs,
        clutter_model: &Clutter,
    ) -> (LmbFilterState<T, N, Updated>, UpdateStats)
    where
        Obs: ObservationModel<T, N, M>,
        Clutter: ClutterModel<T, M>,
    {
        self.update_with_lbp(
            measurements,
            observation_model,
            clutter_model,
            50,
            T::from_f64(1e-6).unwrap(),
        )
    }

    /// Updates the LMB density with measurements using configurable LBP parameters.
    ///
    /// This method allows customizing the Loopy Belief Propagation parameters
    /// via an [`LbpConfig`] struct.
    pub fn update_with_config<const M: usize, Obs, Clutter>(
        self,
        measurements: &[Measurement<T, M>],
        observation_model: &Obs,
        clutter_model: &Clutter,
        lbp_config: &LbpConfig<T>,
    ) -> (LmbFilterState<T, N, Updated>, UpdateStats)
    where
        Obs: ObservationModel<T, N, M>,
        Clutter: ClutterModel<T, M>,
    {
        self.update_with_lbp(
            measurements,
            observation_model,
            clutter_model,
            lbp_config.max_iterations,
            lbp_config.tolerance,
        )
    }

    /// Updates with LBP association using custom parameters.
    pub fn update_with_lbp<const M: usize, Obs, Clutter>(
        mut self,
        measurements: &[Measurement<T, M>],
        observation_model: &Obs,
        clutter_model: &Clutter,
        max_iterations: usize,
        tolerance: T,
    ) -> (LmbFilterState<T, N, Updated>, UpdateStats)
    where
        Obs: ObservationModel<T, N, M>,
        Clutter: ClutterModel<T, M>,
    {
        let mut stats = UpdateStats::default();
        let n_tracks = self.tracks.len();
        let n_meas = measurements.len();

        if n_tracks == 0 {
            return (
                LmbFilterState {
                    tracks: self.tracks,
                    label_gen: self.label_gen,
                    time_step: self.time_step,
                    _phase: PhantomData,
                },
                stats,
            );
        }

        let obs_matrix = observation_model.observation_matrix();
        let meas_noise = observation_model.measurement_noise();

        // Precompute posteriors and likelihoods for all track-measurement pairs
        let mut posteriors = PosteriorGrid::new(n_tracks, n_meas);
        let mut likelihood_matrix: Vec<Vec<T>> = vec![vec![T::zero(); n_meas]; n_tracks];
        let mut detection_probs: Vec<T> = Vec::with_capacity(n_tracks);

        for (i, track) in self.tracks.iter().enumerate() {
            // Get detection probability (use first component's mean)
            let p_d = track
                .components
                .iter()
                .next()
                .map(|c| observation_model.detection_probability(&c.mean))
                .unwrap_or(T::zero());
            detection_probs.push(p_d);

            for (j, measurement) in measurements.iter().enumerate() {
                // Sum likelihood over all components
                let mut total_likelihood = T::zero();
                let mut best_posterior: Option<(StateVector<T, N>, StateCovariance<T, N>, T)> =
                    None;
                let mut best_weight = T::zero();

                for component in track.components.iter() {
                    let predicted_meas = obs_matrix.observe(&component.mean);
                    let innovation = measurement.innovation(predicted_meas);
                    let innovation_cov = compute_innovation_covariance(
                        &component.covariance,
                        &obs_matrix,
                        &meas_noise,
                    );

                    // Compute likelihood
                    let likelihood = crate::types::gaussian::innovation_likelihood(
                        &innovation,
                        innovation_cov.as_matrix(),
                    );

                    if likelihood <= T::zero() {
                        stats.zero_likelihood_count += 1;
                        continue;
                    }

                    let weighted_likelihood = component.weight * likelihood;
                    total_likelihood += weighted_likelihood;

                    // Track best posterior for this component
                    if weighted_likelihood > best_weight {
                        if let Some(kalman_gain) =
                            compute_kalman_gain(&component.covariance, &obs_matrix, &innovation_cov)
                        {
                            let correction = kalman_gain.correct(&innovation);
                            let updated_mean = StateVector::from_svector(
                                component.mean.as_svector() + correction.as_svector(),
                            );
                            let updated_cov = joseph_update(
                                &component.covariance,
                                &kalman_gain,
                                &obs_matrix,
                                &meas_noise,
                            );

                            best_posterior = Some((updated_mean, updated_cov, likelihood));
                            best_weight = weighted_likelihood;
                        } else {
                            stats.singular_covariance_count += 1;
                        }
                    }
                }

                likelihood_matrix[i][j] = total_likelihood;
                if let Some(posterior) = best_posterior {
                    posteriors.set(i, j, posterior);
                }
            }
        }

        // Compute association weights using LBP
        let (marginal_weights, existence_updates, lbp_converged, lbp_iterations) =
            loopy_belief_propagation(
                &likelihood_matrix,
                &detection_probs,
                &self.tracks.existence_probabilities(),
                measurements,
                clutter_model,
                max_iterations,
                tolerance,
            );

        // Record LBP convergence info
        stats.lbp_converged = Some(lbp_converged);
        stats.lbp_iterations = Some(lbp_iterations);

        // Update tracks with association results
        for (i, track) in self.tracks.iter_mut().enumerate() {
            // Update existence probability
            track.existence = existence_updates[i];

            if n_meas == 0 {
                // No measurements - missed detection only
                continue;
            }

            // Create new mixture based on association weights
            let mut new_components =
                crate::types::gaussian::GaussianMixture::with_capacity(n_meas + 1);

            // Miss detection component (weight = marginal_weights[i][0])
            let miss_weight = marginal_weights[i][0];
            if miss_weight > T::from(1e-10).unwrap() {
                for component in track.components.iter() {
                    new_components.push(GaussianState {
                        weight: component.weight * miss_weight,
                        mean: component.mean,
                        covariance: component.covariance,
                    });
                }
            }

            // Detection components (one for each measurement)
            for j in 0..n_meas {
                let det_weight = marginal_weights[i][j + 1]; // +1 because index 0 is miss
                if det_weight > T::from(1e-10).unwrap() {
                    if let Some((mean, cov, _)) = posteriors.get(i, j) {
                        new_components.push(GaussianState {
                            weight: det_weight,
                            mean: *mean,
                            covariance: *cov,
                        });
                    }
                }
            }

            // Normalize weights
            let total_weight = new_components.total_weight();
            if total_weight > T::zero() {
                for component in new_components.iter_mut() {
                    component.weight /= total_weight;
                }
            }

            track.components = new_components;
        }

        (
            LmbFilterState {
                tracks: self.tracks,
                label_gen: self.label_gen,
                time_step: self.time_step,
                _phase: PhantomData,
            },
            stats,
        )
    }

    /// Returns the expected number of targets (before update).
    pub fn expected_target_count(&self) -> T {
        self.tracks.expected_cardinality()
    }

    /// Updates the LMB density with measurements using EKF-style linearization.
    ///
    /// This method is for nonlinear observation models where the observation
    /// function h(x) and its Jacobian depend on the state. The linearization
    /// is performed at each track's predicted state.
    ///
    /// # Arguments
    /// - `measurements`: Measurements in the sensor's native coordinate system
    /// - `observation_model`: Nonlinear observation model with `observe()` and `jacobian_at()`
    /// - `clutter_model`: Clutter model in the same coordinate system as measurements
    ///
    /// # Example
    /// ```ignore
    /// let sensor = RangeBearingSensor::new(sigma_r, sigma_b, p_d);
    /// let clutter = UniformClutterRangeBearing::new(rate, range_bounds, bearing_bounds);
    ///
    /// let measurements = vec![Measurement::from_array([range, bearing])];
    /// let (updated, stats) = predicted.update_ekf(&measurements, &sensor, &clutter);
    /// ```
    pub fn update_ekf<const M: usize, Obs, Clutter>(
        self,
        measurements: &[Measurement<T, M>],
        observation_model: &Obs,
        clutter_model: &Clutter,
    ) -> (LmbFilterState<T, N, Updated>, UpdateStats)
    where
        Obs: NonlinearObservationModel<T, N, M>,
        Clutter: ClutterModel<T, M>,
    {
        self.update_ekf_with_lbp(
            measurements,
            observation_model,
            clutter_model,
            50,
            T::from_f64(1e-6).unwrap(),
        )
    }

    /// Updates with EKF linearization using configurable LBP parameters.
    ///
    /// This method allows customizing the Loopy Belief Propagation parameters
    /// via an [`LbpConfig`] struct.
    pub fn update_ekf_with_config<const M: usize, Obs, Clutter>(
        self,
        measurements: &[Measurement<T, M>],
        observation_model: &Obs,
        clutter_model: &Clutter,
        lbp_config: &LbpConfig<T>,
    ) -> (LmbFilterState<T, N, Updated>, UpdateStats)
    where
        Obs: NonlinearObservationModel<T, N, M>,
        Clutter: ClutterModel<T, M>,
    {
        self.update_ekf_with_lbp(
            measurements,
            observation_model,
            clutter_model,
            lbp_config.max_iterations,
            lbp_config.tolerance,
        )
    }

    /// Updates with EKF linearization and LBP association using custom parameters.
    pub fn update_ekf_with_lbp<const M: usize, Obs, Clutter>(
        mut self,
        measurements: &[Measurement<T, M>],
        observation_model: &Obs,
        clutter_model: &Clutter,
        max_iterations: usize,
        tolerance: T,
    ) -> (LmbFilterState<T, N, Updated>, UpdateStats)
    where
        Obs: NonlinearObservationModel<T, N, M>,
        Clutter: ClutterModel<T, M>,
    {
        let mut stats = UpdateStats::default();
        let n_tracks = self.tracks.len();
        let n_meas = measurements.len();

        if n_tracks == 0 {
            return (
                LmbFilterState {
                    tracks: self.tracks,
                    label_gen: self.label_gen,
                    time_step: self.time_step,
                    _phase: PhantomData,
                },
                stats,
            );
        }

        let meas_noise = observation_model.measurement_noise();

        // Precompute posteriors and likelihoods for all track-measurement pairs
        let mut posteriors = PosteriorGrid::new(n_tracks, n_meas);
        let mut likelihood_matrix: Vec<Vec<T>> = vec![vec![T::zero(); n_meas]; n_tracks];
        let mut detection_probs: Vec<T> = Vec::with_capacity(n_tracks);

        for (i, track) in self.tracks.iter().enumerate() {
            // Get detection probability (use first component's mean)
            let p_d = track
                .components
                .iter()
                .next()
                .map(|c| observation_model.detection_probability(&c.mean))
                .unwrap_or(T::zero());
            detection_probs.push(p_d);

            for (j, measurement) in measurements.iter().enumerate() {
                // Sum likelihood over all components
                let mut total_likelihood = T::zero();
                let mut best_posterior: Option<(StateVector<T, N>, StateCovariance<T, N>, T)> =
                    None;
                let mut best_weight = T::zero();

                for component in track.components.iter() {
                    // EKF: Linearize at component mean
                    let obs_matrix = match observation_model.jacobian_at(&component.mean) {
                        Some(h) => h,
                        None => {
                            // Jacobian undefined (e.g., target at sensor)
                            stats.singular_covariance_count += 1;
                            continue;
                        }
                    };

                    // EKF: Predicted measurement using nonlinear function
                    let predicted_meas = observation_model.observe(&component.mean);

                    // Innovation in measurement space: y = z - h(x)
                    let innovation = measurement.innovation(predicted_meas);

                    let innovation_cov = compute_innovation_covariance(
                        &component.covariance,
                        &obs_matrix,
                        &meas_noise,
                    );

                    // Compute likelihood
                    let likelihood = crate::types::gaussian::innovation_likelihood(
                        &innovation,
                        innovation_cov.as_matrix(),
                    );

                    if likelihood <= T::zero() {
                        stats.zero_likelihood_count += 1;
                        continue;
                    }

                    let weighted_likelihood = component.weight * likelihood;
                    total_likelihood += weighted_likelihood;

                    // Track best posterior for this component
                    if weighted_likelihood > best_weight {
                        if let Some(kalman_gain) =
                            compute_kalman_gain(&component.covariance, &obs_matrix, &innovation_cov)
                        {
                            let correction = kalman_gain.correct(&innovation);
                            let updated_mean = StateVector::from_svector(
                                component.mean.as_svector() + correction.as_svector(),
                            );
                            let updated_cov = joseph_update(
                                &component.covariance,
                                &kalman_gain,
                                &obs_matrix,
                                &meas_noise,
                            );

                            best_posterior = Some((updated_mean, updated_cov, likelihood));
                            best_weight = weighted_likelihood;
                        } else {
                            stats.singular_covariance_count += 1;
                        }
                    }
                }

                likelihood_matrix[i][j] = total_likelihood;
                if let Some(posterior) = best_posterior {
                    posteriors.set(i, j, posterior);
                }
            }
        }

        // Compute association weights using LBP
        let (marginal_weights, existence_updates, lbp_converged, lbp_iterations) =
            loopy_belief_propagation(
                &likelihood_matrix,
                &detection_probs,
                &self.tracks.existence_probabilities(),
                measurements,
                clutter_model,
                max_iterations,
                tolerance,
            );

        // Record LBP convergence info
        stats.lbp_converged = Some(lbp_converged);
        stats.lbp_iterations = Some(lbp_iterations);

        // Update tracks with association results
        for (i, track) in self.tracks.iter_mut().enumerate() {
            // Update existence probability
            track.existence = existence_updates[i];

            if n_meas == 0 {
                // No measurements - missed detection only
                continue;
            }

            // Create new mixture based on association weights
            let mut new_components =
                crate::types::gaussian::GaussianMixture::with_capacity(n_meas + 1);

            // Miss detection component (weight = marginal_weights[i][0])
            let miss_weight = marginal_weights[i][0];
            if miss_weight > T::from(1e-10).unwrap() {
                for component in track.components.iter() {
                    new_components.push(GaussianState {
                        weight: component.weight * miss_weight,
                        mean: component.mean,
                        covariance: component.covariance,
                    });
                }
            }

            // Detection components (one for each measurement)
            for j in 0..n_meas {
                let det_weight = marginal_weights[i][j + 1]; // +1 because index 0 is miss
                if det_weight > T::from(1e-10).unwrap() {
                    if let Some((mean, cov, _)) = posteriors.get(i, j) {
                        new_components.push(GaussianState {
                            weight: det_weight,
                            mean: *mean,
                            covariance: *cov,
                        });
                    }
                }
            }

            // Normalize weights
            let total_weight = new_components.total_weight();
            if total_weight > T::zero() {
                for component in new_components.iter_mut() {
                    component.weight /= total_weight;
                }
            }

            track.components = new_components;
        }

        (
            LmbFilterState {
                tracks: self.tracks,
                label_gen: self.label_gen,
                time_step: self.time_step,
                _phase: PhantomData,
            },
            stats,
        )
    }
}

// ============================================================================
// Loopy Belief Propagation
// ============================================================================

/// Performs Loopy Belief Propagation for data association.
///
/// Returns:
/// - Marginal association weights: [n_tracks][n_meas + 1] where index 0 is miss
/// - Updated existence probabilities
/// - Whether LBP converged (true if tolerance was met, false if max_iterations was hit)
/// - Number of iterations run
#[cfg(feature = "alloc")]
fn loopy_belief_propagation<
    T: RealField + Float + Copy,
    const M: usize,
    Clutter: ClutterModel<T, M>,
>(
    likelihood_matrix: &[Vec<T>],
    detection_probs: &[T],
    existence_probs: &[T],
    measurements: &[Measurement<T, M>],
    clutter_model: &Clutter,
    max_iterations: usize,
    tolerance: T,
) -> (Vec<Vec<T>>, Vec<T>, bool, usize) {
    let n_tracks = likelihood_matrix.len();
    let n_meas = if n_tracks > 0 {
        likelihood_matrix[0].len()
    } else {
        0
    };

    if n_tracks == 0 || n_meas == 0 {
        // No measurements - all tracks get miss detection
        let weights: Vec<Vec<T>> = existence_probs.iter().map(|_| vec![T::one()]).collect();
        let updated_existence: Vec<T> = existence_probs
            .iter()
            .zip(detection_probs.iter())
            .map(|(&r, &p_d)| {
                // Miss detection update: r' = r(1-p_d) / (1 - r*p_d)
                let numerator = r * (T::one() - p_d);
                let denominator = T::one() - r * p_d;
                if denominator > T::zero() {
                    numerator / denominator
                } else {
                    r
                }
            })
            .collect();
        // No iterations needed, considered converged
        return (weights, updated_existence, true, 0);
    }

    // Compute clutter intensities
    let clutter_intensities: Vec<T> = measurements
        .iter()
        .map(|z| clutter_model.clutter_intensity(z))
        .collect();

    // Compute phi vector: phi[i] = r * (1 - p_d)
    let phi: Vec<T> = existence_probs
        .iter()
        .zip(detection_probs.iter())
        .map(|(&r, &p_d)| r * (T::one() - p_d))
        .collect();

    // Compute eta vector: eta[i] = 1 - r * p_d
    let eta: Vec<T> = existence_probs
        .iter()
        .zip(detection_probs.iter())
        .map(|(&r, &p_d)| T::one() - r * p_d)
        .collect();

    // Compute psi matrix: psi[i][j] = r * p_d * L[i][j] / (kappa[j] * eta[i])
    let mut psi: Vec<Vec<T>> = vec![vec![T::zero(); n_meas]; n_tracks];
    for i in 0..n_tracks {
        let r = existence_probs[i];
        let p_d = detection_probs[i];
        for j in 0..n_meas {
            let kappa = clutter_intensities[j];
            if kappa > T::zero() && eta[i] > T::zero() {
                psi[i][j] = r * p_d * likelihood_matrix[i][j] / (kappa * eta[i]);
            }
        }
    }

    // Initialize messages: sigma[i][j] = 1
    let mut sigma_mt: Vec<Vec<T>> = vec![vec![T::one(); n_tracks]; n_meas];
    let mut sigma_tm: Vec<Vec<T>> = vec![vec![T::one(); n_meas]; n_tracks];

    // LBP iterations - track convergence and iteration count
    let mut lbp_converged = false;
    let mut lbp_iterations = 0usize;

    for iter in 0..max_iterations {
        lbp_iterations = iter + 1;
        let sigma_mt_old = sigma_mt.clone();

        // Compute B = psi .* sigma_mt (element-wise)
        // B[i][j] = psi[i][j] * sigma_mt[j][i]
        let mut b: Vec<Vec<T>> = vec![vec![T::zero(); n_meas]; n_tracks];
        for i in 0..n_tracks {
            for j in 0..n_meas {
                b[i][j] = psi[i][j] * sigma_mt[j][i];
            }
        }

        // Compute row sums of B
        let b_row_sums: Vec<T> = b
            .iter()
            .map(|row| row.iter().fold(T::zero(), |acc, &x| acc + x))
            .collect();

        // Update sigma_tm: sigma_tm[i][j] = psi[i][j] / (-B[i][j] + rowsum(B[i]) + 1)
        for i in 0..n_tracks {
            for j in 0..n_meas {
                let denom = -b[i][j] + b_row_sums[i] + T::one();
                sigma_tm[i][j] = if denom > T::zero() {
                    psi[i][j] / denom
                } else {
                    T::one()
                };
            }
        }

        // Compute column sums of sigma_tm
        let mut sigma_tm_col_sums: Vec<T> = vec![T::zero(); n_meas];
        for row in sigma_tm.iter().take(n_tracks) {
            for j in 0..n_meas {
                sigma_tm_col_sums[j] += row[j];
            }
        }

        // Update sigma_mt: sigma_mt[j][i] = 1 / (-sigma_tm[i][j] + colsum(sigma_tm[j]) + 1)
        for j in 0..n_meas {
            for i in 0..n_tracks {
                let denom = -sigma_tm[i][j] + sigma_tm_col_sums[j] + T::one();
                sigma_mt[j][i] = if denom > T::zero() {
                    T::one() / denom
                } else {
                    T::one()
                };
            }
        }

        // Check convergence
        let mut max_delta = T::zero();
        for j in 0..n_meas {
            for i in 0..n_tracks {
                let delta = ComplexField::abs(sigma_mt[j][i] - sigma_mt_old[j][i]);
                if delta > max_delta {
                    max_delta = delta;
                }
            }
        }
        if max_delta < tolerance {
            lbp_converged = true;
            break;
        }
    }

    // Compute final B matrix
    let mut b: Vec<Vec<T>> = vec![vec![T::zero(); n_meas]; n_tracks];
    for i in 0..n_tracks {
        for j in 0..n_meas {
            b[i][j] = psi[i][j] * sigma_mt[j][i];
        }
    }

    // Compute Gamma matrix: Gamma[i] = [phi[i], B[i] * eta[i]]
    // Then W[i] = Gamma[i] / sum(Gamma[i])
    let mut marginal_weights: Vec<Vec<T>> = Vec::with_capacity(n_tracks);
    let mut updated_existence: Vec<T> = Vec::with_capacity(n_tracks);

    for i in 0..n_tracks {
        let mut gamma = Vec::with_capacity(n_meas + 1);
        gamma.push(phi[i]); // Miss detection

        for b_val in b[i].iter().take(n_meas) {
            gamma.push(*b_val * eta[i]);
        }

        let gamma_sum: T = gamma.iter().fold(T::zero(), |acc, &x| acc + x);

        // Marginal weights
        let weights: Vec<T> = if gamma_sum > T::zero() {
            gamma.iter().map(|&g| g / gamma_sum).collect()
        } else {
            let mut w = vec![T::zero(); n_meas + 1];
            w[0] = T::one(); // All weight on miss
            w
        };
        marginal_weights.push(weights);

        // Updated existence: r' = gamma_sum / (eta[i] + gamma_sum - phi[i])
        let denom = eta[i] + gamma_sum - phi[i];
        let r_updated = if denom > T::zero() {
            gamma_sum / denom
        } else {
            existence_probs[i]
        };
        updated_existence.push(r_updated);
    }

    (
        marginal_weights,
        updated_existence,
        lbp_converged,
        lbp_iterations,
    )
}

// ============================================================================
// State Extraction
// ============================================================================

/// Extracted target state from an LMB filter.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct LmbEstimate<T: RealField, const N: usize> {
    /// Track label
    pub label: Label,
    /// State estimate (weighted mean of components)
    pub state: StateVector<T, N>,
    /// State covariance (from highest-weight component)
    pub covariance: StateCovariance<T, N>,
    /// Existence probability
    pub existence: T,
}

/// Extracts target estimates from an LMB filter state using MAP cardinality.
#[cfg(feature = "alloc")]
pub fn extract_lmb_estimates<T: RealField + Float + Copy, const N: usize, Phase>(
    state: &LmbFilterState<T, N, Phase>,
) -> Vec<LmbEstimate<T, N>> {
    let existence_probs = state.tracks.existence_probabilities();
    let result = map_cardinality_estimate(&existence_probs);

    result
        .track_indices
        .iter()
        .filter_map(|&idx| {
            let track = &state.tracks.tracks[idx];
            track.best_estimate().map(|best| LmbEstimate {
                label: track.label,
                state: track.weighted_mean(),
                covariance: best.covariance,
                existence: track.existence,
            })
        })
        .collect()
}

/// Extracts target estimates using a simple existence threshold.
#[cfg(feature = "alloc")]
pub fn extract_lmb_estimates_threshold<T: RealField + Float + Copy, const N: usize, Phase>(
    state: &LmbFilterState<T, N, Phase>,
    threshold: T,
) -> Vec<LmbEstimate<T, N>> {
    state
        .tracks
        .iter()
        .filter(|track| track.existence >= threshold)
        .filter_map(|track| {
            track.best_estimate().map(|best| LmbEstimate {
                label: track.label,
                state: track.weighted_mean(),
                covariance: best.covariance,
                existence: track.existence,
            })
        })
        .collect()
}

// ============================================================================
// LMB Filter (with models)
// ============================================================================

/// Complete LMB filter with models.
///
/// This struct combines the filter state with all necessary models
/// for a complete tracking system.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct LmbFilter<T, Trans, Obs, Clutter, Birth, const N: usize, const M: usize>
where
    T: RealField,
    Trans: TransitionModel<T, N>,
    Obs: ObservationModel<T, N, M>,
    Clutter: ClutterModel<T, M>,
    Birth: LabeledBirthModel<T, N>,
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
    LmbFilter<T, Trans, Obs, Clutter, Birth, N, M>
where
    T: RealField + Float + Copy,
    Trans: TransitionModel<T, N>,
    Obs: ObservationModel<T, N, M>,
    Clutter: ClutterModel<T, M>,
    Birth: LabeledBirthModel<T, N>,
{
    /// Creates a new LMB filter with the specified models.
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
    pub fn initial_state(&self) -> LmbFilterState<T, N, Updated> {
        LmbFilterState::new()
    }

    /// Creates an initial filter state from prior tracks.
    pub fn initial_state_from(&self, tracks: Vec<LmbTrack<T, N>>) -> LmbFilterState<T, N, Updated> {
        LmbFilterState::from_tracks(tracks)
    }

    /// Runs one complete predict-update cycle.
    pub fn step(
        &self,
        state: LmbFilterState<T, N, Updated>,
        measurements: &[Measurement<T, M>],
        dt: T,
    ) -> (LmbFilterState<T, N, Updated>, UpdateStats) {
        let predicted = state.predict(&self.transition, &self.birth, dt);
        predicted.update(measurements, &self.observation, &self.clutter)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::gaussian::GaussianState;
    use crate::types::spaces::StateCovariance;

    #[cfg(feature = "alloc")]
    #[test]
    fn test_lmb_filter_state_creation() {
        let state: LmbFilterState<f64, 4, Updated> = LmbFilterState::new();
        assert_eq!(state.num_tracks(), 0);
        assert_eq!(state.time_step, 0);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_lmb_filter_state_from_tracks() {
        let label = Label::new(0, 0);
        let mean: StateVector<f64, 4> = StateVector::from_array([1.0, 2.0, 0.5, 0.5]);
        let cov: StateCovariance<f64, 4> = StateCovariance::identity();
        let state = GaussianState::new(1.0, mean, cov);

        let track = LmbTrack::new(label, 0.9, state);
        let filter_state = LmbFilterState::from_tracks(vec![track]);

        assert_eq!(filter_state.num_tracks(), 1);
        assert!((filter_state.expected_target_count() - 0.9).abs() < 1e-10);
    }
}
