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

use alloc::vec;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use super::cardinality::map_cardinality_estimate;
use super::types::{LmbTrack, LmbTrackSet};
use crate::models::{
    ClutterModel, NonlinearObservationModel, ObservationModel, TransitionModel,
    effective_clutter_intensity,
};
use crate::types::gaussian::GaussianState;
use crate::types::labels::{BernoulliTrack, Label, LabelGenerator};
use crate::types::phase::{Predicted, UpdateStats, Updated};
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
/// Extends the basic [`crate::models::BirthModel`] with the ability to generate
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
        #[allow(clippy::type_complexity)]
        let mut posteriors: Vec<Vec<Vec<(StateVector<T, N>, StateCovariance<T, N>, T)>>> =
            vec![vec![Vec::new(); n_meas]; n_tracks];
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
                // Sum likelihood over all components and retain each conditional mode.
                let mut total_likelihood = T::zero();

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

                        total_likelihood += weighted_likelihood;
                        posteriors[i][j].push((updated_mean, updated_cov, weighted_likelihood));
                    } else {
                        stats.singular_covariance_count += 1;
                    }
                }

                likelihood_matrix[i][j] = total_likelihood;
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
                    let measurement_likelihood = likelihood_matrix[i][j];
                    if measurement_likelihood > T::zero() {
                        for (mean, covariance, weighted_likelihood) in &posteriors[i][j] {
                            new_components.push(GaussianState {
                                weight: det_weight * *weighted_likelihood / measurement_likelihood,
                                mean: *mean,
                                covariance: *covariance,
                            });
                        }
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
    /// ```no_run
    /// use tracktor::prelude::*;
    /// use tracktor::types::phase::Predicted;
    ///
    /// let sensor = RangeBearingSensor::new(1.0, 0.01, 0.9);
    /// let clutter = UniformClutterRangeBearing::new(5.0, (0.0, 1000.0), (-3.14, 3.14));
    ///
    /// let measurements = vec![Measurement::from_array([500.0, 0.5])];
    /// // `predicted` would come from a prior predict() call
    /// # let predicted: LmbFilterState<f64, 4, Predicted> = unimplemented!();
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
        #[allow(clippy::type_complexity)]
        let mut posteriors: Vec<Vec<Vec<(StateVector<T, N>, StateCovariance<T, N>, T)>>> =
            vec![vec![Vec::new(); n_meas]; n_tracks];
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
                // Sum likelihood over all components and retain each conditional mode.
                let mut total_likelihood = T::zero();

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
                    let innovation = observation_model.innovation(measurement, &predicted_meas);

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

                        total_likelihood += weighted_likelihood;
                        posteriors[i][j].push((updated_mean, updated_cov, weighted_likelihood));
                    } else {
                        stats.singular_covariance_count += 1;
                    }
                }

                likelihood_matrix[i][j] = total_likelihood;
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
                    let measurement_likelihood = likelihood_matrix[i][j];
                    if measurement_likelihood > T::zero() {
                        for (mean, covariance, weighted_likelihood) in &posteriors[i][j] {
                            new_components.push(GaussianState {
                                weight: det_weight * *weighted_likelihood / measurement_likelihood,
                                mean: *mean,
                                covariance: *covariance,
                            });
                        }
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

#[cfg(feature = "alloc")]
fn log_add_exp<T: RealField + Float + Copy>(left: T, right: T) -> T {
    if left == T::infinity() || right == T::infinity() {
        return T::infinity();
    }
    if left == T::neg_infinity() {
        return right;
    }
    if right == T::neg_infinity() {
        return left;
    }

    let maximum = if left > right { left } else { right };
    let minimum = if left > right { right } else { left };
    maximum + Float::ln(T::one() + Float::exp(minimum - maximum))
}

#[cfg(feature = "alloc")]
fn log_positive<T: RealField + Float + Copy>(value: T) -> T {
    if value > T::zero() {
        Float::ln(value)
    } else {
        T::neg_infinity()
    }
}

#[cfg(feature = "alloc")]
fn subtract_logs<T: RealField + Float + Copy>(numerator: T, denominator: T) -> T {
    if numerator == T::neg_infinity() || denominator == T::infinity() {
        T::neg_infinity()
    } else if denominator == T::neg_infinity() {
        T::infinity()
    } else {
        numerator - denominator
    }
}

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

    // eta[i] = 1 - r * p_d. Work with q = eta * psi = r*p_d*L/kappa
    // rather than psi itself. Besides avoiding overflow, this remains defined
    // at the valid boundary r = p_d = 1 where eta is exactly zero.
    let eta: Vec<T> = existence_probs
        .iter()
        .zip(detection_probs.iter())
        .map(|(&r, &p_d)| T::one() - r * p_d)
        .collect();
    let mut log_q: Vec<Vec<T>> = vec![vec![T::neg_infinity(); n_meas]; n_tracks];
    for i in 0..n_tracks {
        let log_r = log_positive(existence_probs[i]);
        let log_p_d = log_positive(detection_probs[i]);
        for j in 0..n_meas {
            let kappa = effective_clutter_intensity(clutter_intensities[j]);
            let likelihood = likelihood_matrix[i][j];
            if likelihood > T::zero() && log_r != T::neg_infinity() && log_p_d != T::neg_infinity()
            {
                log_q[i][j] = log_r + log_p_d + Float::ln(likelihood) - Float::ln(kappa);
            }
        }
    }

    // log_sigma_mt[j][i] and log_sigma_tm[i][j] both start at ln(1).
    let mut log_sigma_mt: Vec<Vec<T>> = vec![vec![T::zero(); n_tracks]; n_meas];
    let mut log_sigma_tm: Vec<Vec<T>> = vec![vec![T::zero(); n_meas]; n_tracks];

    let mut lbp_converged = false;
    let mut lbp_iterations = 0usize;

    for iter in 0..max_iterations {
        lbp_iterations = iter + 1;
        let log_sigma_mt_old = log_sigma_mt.clone();

        let mut log_q_sigma: Vec<Vec<T>> = vec![vec![T::neg_infinity(); n_meas]; n_tracks];
        for i in 0..n_tracks {
            for j in 0..n_meas {
                log_q_sigma[i][j] = log_q[i][j] + log_sigma_mt[j][i];
            }
        }

        // Multiplying the usual message equation by eta gives
        // sigma_tm[i][j] = q[i][j] / (eta[i] + sum_{k != j} q[i][k]*sigma_mt[k][i]).
        // This is algebraically equivalent when eta > 0 and has the correct
        // limiting behavior when eta = 0.
        let mut prefix_meas = vec![T::neg_infinity(); n_meas + 1];
        let mut suffix_meas = vec![T::neg_infinity(); n_meas + 1];
        for i in 0..n_tracks {
            prefix_meas[0] = log_positive(eta[i]);
            suffix_meas[n_meas] = T::neg_infinity();
            for j in 0..n_meas {
                prefix_meas[j + 1] = log_add_exp(prefix_meas[j], log_q_sigma[i][j]);
            }
            for j in (0..n_meas).rev() {
                suffix_meas[j] = log_add_exp(log_q_sigma[i][j], suffix_meas[j + 1]);
            }
            for j in 0..n_meas {
                let log_denominator = log_add_exp(prefix_meas[j], suffix_meas[j + 1]);
                log_sigma_tm[i][j] = subtract_logs(log_q[i][j], log_denominator);
            }
        }

        // sigma_mt[j][i] = 1 / (1 + sum_{k != i} sigma_tm[k][j]).
        let mut prefix_tracks = vec![T::neg_infinity(); n_tracks + 1];
        let mut suffix_tracks = vec![T::neg_infinity(); n_tracks + 1];
        for j in 0..n_meas {
            prefix_tracks[0] = T::zero();
            suffix_tracks[n_tracks] = T::neg_infinity();
            for i in 0..n_tracks {
                prefix_tracks[i + 1] = log_add_exp(prefix_tracks[i], log_sigma_tm[i][j]);
            }
            for i in (0..n_tracks).rev() {
                suffix_tracks[i] = log_add_exp(log_sigma_tm[i][j], suffix_tracks[i + 1]);
            }
            for i in 0..n_tracks {
                let log_denominator = log_add_exp(prefix_tracks[i], suffix_tracks[i + 1]);
                log_sigma_mt[j][i] = -log_denominator;
            }
        }

        let mut max_delta = T::zero();
        for j in 0..n_meas {
            for i in 0..n_tracks {
                let current = Float::exp(log_sigma_mt[j][i]);
                let previous = Float::exp(log_sigma_mt_old[j][i]);
                let delta = ComplexField::abs(current - previous);
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

    let mut marginal_weights: Vec<Vec<T>> = Vec::with_capacity(n_tracks);
    let mut updated_existence: Vec<T> = Vec::with_capacity(n_tracks);

    for i in 0..n_tracks {
        let log_eta = log_positive(eta[i]);
        let log_phi = log_positive(existence_probs[i] * (T::one() - detection_probs[i]));
        let mut log_gamma = Vec::with_capacity(n_meas + 1);
        log_gamma.push(log_phi);
        for j in 0..n_meas {
            log_gamma.push(log_q[i][j] + log_sigma_mt[j][i]);
        }

        let log_gamma_sum = log_gamma
            .iter()
            .copied()
            .fold(T::neg_infinity(), log_add_exp);
        let weights = if Float::is_finite(log_gamma_sum) {
            log_gamma
                .iter()
                .map(|&log_weight| Float::exp(log_weight - log_gamma_sum))
                .collect()
        } else {
            let mut weights = vec![T::zero(); n_meas + 1];
            weights[0] = T::one();
            weights
        };
        marginal_weights.push(weights);

        // gamma_sum = phi + detections, while the existence denominator is
        // eta + detections. Form both sums in logs rather than overflowing B.
        let log_detection_sum = log_gamma
            .iter()
            .skip(1)
            .copied()
            .fold(T::neg_infinity(), log_add_exp);
        let log_denominator = log_add_exp(log_eta, log_detection_sum);
        let r_updated = if log_denominator == T::neg_infinity() {
            // Both numerator and denominator are zero. Preserve the prior at
            // this indeterminate model boundary (for example r = p_d = 1
            // with no supported association event).
            existence_probs[i]
        } else if log_gamma_sum == T::neg_infinity() {
            // A zero existence numerator over a positive denominator is an
            // exact zero posterior, not an indeterminate 0/0 case.
            T::zero()
        } else if Float::is_finite(log_gamma_sum) && Float::is_finite(log_denominator) {
            Float::exp(log_gamma_sum - log_denominator)
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
    use crate::models::{
        ConstantVelocity2D, PositionSensor2D, RangeBearingSensor, UniformClutter2D,
        UniformClutterRangeBearing,
    };
    use crate::types::gaussian::{GaussianMixture, GaussianState};
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

    #[cfg(feature = "alloc")]
    #[test]
    fn test_linear_update_retains_all_mixture_posterior_modes() {
        let components = GaussianMixture::from_components(vec![
            GaussianState::new(
                0.5,
                StateVector::from_array([45.0, 50.0, 0.0, 0.0]),
                StateCovariance::identity(),
            ),
            GaussianState::new(
                0.5,
                StateVector::from_array([55.0, 50.0, 0.0, 0.0]),
                StateCovariance::identity(),
            ),
        ]);
        let state = LmbFilterState::from_tracks(vec![LmbTrack::with_components(
            Label::new(0, 0),
            0.9,
            components,
        )]);
        let predicted = state.predict(&ConstantVelocity2D::new(0.0, 1.0), &NoBirthModel, 0.0);
        let (updated, _) = predicted.update(
            &[Measurement::from_array([50.0, 50.0])],
            &PositionSensor2D::new(10.0, 0.9),
            &UniformClutter2D::new(1.0, (0.0, 100.0), (0.0, 100.0)),
        );

        // Two missed-detection modes plus two measurement-conditioned modes.
        assert_eq!(updated.tracks.tracks[0].components.len(), 4);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_out_of_support_clutter_still_allows_lbp_target_association_f32() {
        let track = LmbTrack::new(
            Label::new(0, 0),
            0.9_f32,
            GaussianState::new(
                1.0,
                StateVector::from_array([20.0, 20.0, 0.0, 0.0]),
                StateCovariance::identity().scale(1.0e-8),
            ),
        );
        let predicted = LmbFilterState::from_tracks(vec![track]).predict(
            &ConstantVelocity2D::new(0.0, 1.0),
            &NoBirthModel,
            0.0,
        );
        let (updated, _) = predicted.update(
            &[Measurement::from_array([20.0, 20.0])],
            &PositionSensor2D::new(1.0e-8, 0.9),
            &UniformClutter2D::new(1.0, (0.0, 10.0), (0.0, 10.0)),
        );

        assert!(updated.tracks.tracks[0].existence.is_finite());
        assert!(updated.tracks.tracks[0].existence > 0.9);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_lbp_certain_track_and_detection_uses_measurement() {
        let clutter = UniformClutter2D::new(1.0_f64, (0.0, 1.0), (0.0, 1.0));
        let measurement = Measurement::from_array([0.5, 0.5]);
        let (weights, existence, converged, iterations) = loopy_belief_propagation(
            &[vec![0.25]],
            &[1.0],
            &[1.0],
            &[measurement],
            &clutter,
            20,
            1.0e-12,
        );

        assert!(converged);
        assert!(iterations > 0);
        assert_eq!(weights.len(), 1);
        assert_eq!(weights[0], vec![0.0, 1.0]);
        assert_eq!(existence, vec![1.0]);
        assert!(weights[0].iter().all(|weight| weight.is_finite()));
        assert!(existence.iter().all(|probability| probability.is_finite()));
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_lbp_zero_detection_likelihood_yields_zero_existence() {
        let clutter = UniformClutter2D::new(1.0_f64, (0.0, 1.0), (0.0, 1.0));
        let measurement = Measurement::from_array([0.5, 0.5]);
        let (weights, existence, _, _) = loopy_belief_propagation(
            &[vec![0.0]],
            &[1.0],
            &[0.5],
            &[measurement],
            &clutter,
            20,
            1.0e-12,
        );

        assert_eq!(weights, vec![vec![1.0, 0.0]]);
        assert_eq!(existence, vec![0.0]);
        assert!(weights[0].iter().all(|weight| weight.is_finite()));
        assert!(existence[0].is_finite());
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_lbp_zero_boundaries_and_no_measurements_are_finite() {
        let clutter = UniformClutter2D::new(1.0_f64, (0.0, 1.0), (0.0, 1.0));
        let measurement = Measurement::from_array([0.5, 0.5]);

        for (existence_probability, detection_probability, likelihood, expected_existence) in [
            (0.0, 1.0, 0.25, 0.0),
            (1.0, 0.0, 0.25, 1.0),
            (1.0, 1.0, 0.0, 1.0),
        ] {
            let (weights, existence, _, _) = loopy_belief_propagation(
                &[vec![likelihood]],
                &[detection_probability],
                &[existence_probability],
                &[measurement],
                &clutter,
                20,
                1.0e-12,
            );
            assert_eq!(weights[0], vec![1.0, 0.0]);
            assert_eq!(existence, vec![expected_existence]);
            assert!(weights[0].iter().all(|weight| weight.is_finite()));
        }

        let (weights, existence, converged, iterations) = loopy_belief_propagation::<_, 2, _>(
            &[Vec::new()],
            &[1.0],
            &[1.0],
            &[],
            &clutter,
            20,
            1.0e-12,
        );
        assert_eq!(weights, vec![vec![1.0]]);
        assert_eq!(existence, vec![1.0]);
        assert!(converged);
        assert_eq!(iterations, 0);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_ekf_update_retains_all_mixture_posterior_modes() {
        let components = GaussianMixture::from_components(vec![
            GaussianState::new(
                0.5,
                StateVector::from_array([50.0, -5.0, 0.0, 0.0]),
                StateCovariance::identity(),
            ),
            GaussianState::new(
                0.5,
                StateVector::from_array([50.0, 5.0, 0.0, 0.0]),
                StateCovariance::identity(),
            ),
        ]);
        let state = LmbFilterState::from_tracks(vec![LmbTrack::with_components(
            Label::new(0, 0),
            0.9,
            components,
        )]);
        let predicted = state.predict(&ConstantVelocity2D::new(0.0, 1.0), &NoBirthModel, 0.0);
        let (updated, _) = predicted.update_ekf(
            &[Measurement::from_array([50.25, 0.0])],
            &RangeBearingSensor::new(10.0, 0.2, 0.9),
            &UniformClutterRangeBearing::new(
                1.0,
                (0.0, 100.0),
                (-core::f64::consts::PI, core::f64::consts::PI),
            ),
        );

        assert_eq!(updated.tracks.tracks[0].components.len(), 4);
    }
}
