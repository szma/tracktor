//! Gaussian Mixture Cardinalized Probability Hypothesis Density (GM-CPHD) Filter
//!
//! Implementation of the GM-CPHD filter for multi-target tracking with improved
//! cardinality estimation compared to the standard PHD filter.
//!
//! The CPHD filter propagates a cardinality distribution alongside the intensity
//! function, providing more accurate estimates of the number of targets.
//!
//! Reference: Vo, B.-T., Vo, B.-N., & Cantoni, A. (2007). "Analytic Implementations
//! of the Cardinalized Probability Hypothesis Density Filter"

use ::core::marker::PhantomData;
use nalgebra::RealField;
use num_traits::Float;

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use crate::filters::lmb::cardinality::elementary_symmetric_function;
use crate::models::{BirthModel, ClutterModel, ObservationModel, TransitionModel};
use crate::types::gaussian::{GaussianMixture, GaussianState, innovation_likelihood};
use crate::types::phase::{Predicted, Updated};
use crate::types::spaces::{ComputeInnovation, Measurement, StateVector};
use crate::types::transforms::{compute_innovation_covariance, compute_kalman_gain, joseph_update};

// ============================================================================
// Update Statistics
// ============================================================================

/// Statistics from a CPHD filter update step, reporting any numerical issues.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, Default)]
pub struct CphdUpdateStats {
    /// Number of components where Kalman gain computation failed (singular innovation covariance)
    pub singular_covariance_count: usize,
    /// Number of components where likelihood computation returned zero or failed
    pub zero_likelihood_count: usize,
    /// Maximum cardinality considered in the distribution
    pub max_cardinality: usize,
    /// MAP (most likely) cardinality estimate
    pub map_cardinality: usize,
}

#[cfg(feature = "alloc")]
impl CphdUpdateStats {
    /// Returns true if any numerical issues were encountered.
    pub fn has_issues(&self) -> bool {
        self.singular_covariance_count > 0 || self.zero_likelihood_count > 0
    }
}

// ============================================================================
// CPHD Filter State
// ============================================================================

/// The state of a GM-CPHD filter at a particular phase.
///
/// The `Phase` parameter encodes whether this is a predicted or updated state,
/// ensuring correct operation ordering at compile time.
///
/// Unlike the PHD filter, the CPHD filter maintains an explicit cardinality
/// distribution `rho[n]` giving the probability of exactly `n` targets existing.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct CphdFilterState<T: RealField, const N: usize, Phase> {
    /// Gaussian mixture representing the PHD (intensity function)
    pub mixture: GaussianMixture<T, N>,
    /// Cardinality distribution: rho[n] = P(exactly n targets exist)
    /// Index 0 corresponds to 0 targets, index 1 to 1 target, etc.
    pub cardinality: Vec<T>,
    /// Current time step
    pub time_step: u32,
    /// Phase marker
    _phase: PhantomData<Phase>,
}

// Generic methods available on any phase
#[cfg(feature = "alloc")]
impl<T: RealField + Copy, const N: usize, Phase> CphdFilterState<T, N, Phase> {
    /// Returns the expected number of targets from the cardinality distribution.
    pub fn expected_target_count(&self) -> T {
        self.cardinality
            .iter()
            .enumerate()
            .fold(T::zero(), |acc, (n, &rho_n)| {
                acc + T::from_usize(n).unwrap() * rho_n
            })
    }

    /// Returns the MAP (Maximum A Posteriori) cardinality estimate.
    pub fn map_cardinality(&self) -> usize {
        self.cardinality
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Returns the cardinality variance.
    pub fn cardinality_variance(&self) -> T {
        let mean = self.expected_target_count();
        self.cardinality
            .iter()
            .enumerate()
            .fold(T::zero(), |acc, (n, &rho_n)| {
                let diff = T::from_usize(n).unwrap() - mean;
                acc + diff * diff * rho_n
            })
    }
}

#[cfg(feature = "alloc")]
impl<T: RealField + Float + Copy, const N: usize> CphdFilterState<T, N, Updated> {
    /// Creates a new filter state in the updated phase with empty mixture.
    ///
    /// Initializes with cardinality distribution rho(0) = 1 (no targets).
    pub fn new() -> Self {
        Self {
            mixture: GaussianMixture::new(),
            cardinality: vec![T::one()],
            time_step: 0,
            _phase: PhantomData,
        }
    }

    /// Creates a filter state from an initial mixture.
    ///
    /// The cardinality distribution is initialized based on the total weight
    /// of the mixture using a Poisson approximation.
    pub fn from_mixture(mixture: GaussianMixture<T, N>) -> Self {
        let expected_count = mixture.total_weight();
        let cardinality = poisson_distribution(expected_count, mixture.len().max(10));

        Self {
            mixture,
            cardinality,
            time_step: 0,
            _phase: PhantomData,
        }
    }

    /// Creates a filter state from an initial mixture and cardinality distribution.
    pub fn from_mixture_and_cardinality(
        mixture: GaussianMixture<T, N>,
        cardinality: Vec<T>,
    ) -> Self {
        Self {
            mixture,
            cardinality,
            time_step: 0,
            _phase: PhantomData,
        }
    }

    /// Predicts the CPHD to the next time step.
    ///
    /// This method consumes the updated state and returns a predicted state.
    /// Both the intensity (mixture) and cardinality distribution are predicted.
    pub fn predict<Trans, Birth>(
        self,
        transition_model: &Trans,
        birth_model: &Birth,
        dt: T,
    ) -> CphdFilterState<T, N, Predicted>
    where
        Trans: TransitionModel<T, N>,
        Birth: BirthModel<T, N>,
    {
        let transition_matrix = transition_model.transition_matrix(dt);
        let process_noise = transition_model.process_noise(dt);

        let birth_components = birth_model.birth_components_vec();
        let birth_weight: T = birth_components
            .iter()
            .fold(T::zero(), |acc, c| acc + c.weight);

        let mut predicted =
            GaussianMixture::with_capacity(self.mixture.len() + birth_components.len());

        // Predict surviving components
        for component in self.mixture.iter() {
            let p_s = transition_model.survival_probability(&component.mean);
            let predicted_weight = component.weight * p_s;

            let predicted_component = GaussianState {
                weight: predicted_weight,
                mean: transition_matrix.apply_state(&component.mean),
                covariance: transition_matrix
                    .propagate_covariance(&component.covariance)
                    .add(&process_noise),
            };

            predicted.push(predicted_component);
        }

        // Add birth components
        for birth_component in birth_components {
            predicted.push(birth_component);
        }

        // Predict cardinality distribution
        // The predicted cardinality is a convolution of survival and birth processes
        let predicted_cardinality =
            predict_cardinality(&self.cardinality, transition_model, birth_weight);

        CphdFilterState {
            mixture: predicted,
            cardinality: predicted_cardinality,
            time_step: self.time_step + 1,
            _phase: PhantomData,
        }
    }
}

#[cfg(feature = "alloc")]
impl<T: RealField + Float + Copy, const N: usize> Default for CphdFilterState<T, N, Updated> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "alloc")]
impl<T: RealField + Float + Copy, const N: usize> CphdFilterState<T, N, Predicted> {
    /// Updates the CPHD with a set of measurements.
    ///
    /// This method consumes the predicted state and returns an updated state along
    /// with statistics about any numerical issues encountered during the update.
    ///
    /// The CPHD update differs from PHD in that it uses the cardinality distribution
    /// to properly normalize the component weights, leading to more accurate
    /// target number estimation.
    pub fn update_with_stats<const M: usize, Obs, Clutter>(
        self,
        measurements: &[Measurement<T, M>],
        observation_model: &Obs,
        clutter_model: &Clutter,
    ) -> (CphdFilterState<T, N, Updated>, CphdUpdateStats)
    where
        Obs: ObservationModel<T, N, M>,
        Clutter: ClutterModel<T, M>,
    {
        let obs_matrix = observation_model.observation_matrix();
        let meas_noise = observation_model.measurement_noise();
        let mut stats = CphdUpdateStats::default();

        let num_measurements = measurements.len();

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

        // Compute likelihood matrix: L[j][i] = likelihood of measurement j from component i
        let likelihood_matrix: Vec<Vec<T>> = measurements
            .iter()
            .map(|measurement| {
                self.mixture
                    .iter()
                    .enumerate()
                    .map(|(i, c)| {
                        let (p_d, ref predicted_meas, ref innovation_cov, _) = component_data[i];
                        let innovation = measurement.innovation(*predicted_meas);
                        let likelihood =
                            innovation_likelihood(&innovation, innovation_cov.as_matrix());

                        if likelihood <= T::zero() {
                            stats.zero_likelihood_count += 1;
                        }

                        p_d * c.weight * likelihood
                    })
                    .collect()
            })
            .collect();

        // Compute detection terms for each component
        let detection_terms: Vec<T> = self
            .mixture
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let p_d = component_data[i].0;
                p_d * c.weight
            })
            .collect();

        // Compute missed detection terms
        let missed_detection_terms: Vec<T> = self
            .mixture
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let p_d = component_data[i].0;
                (T::one() - p_d) * c.weight
            })
            .collect();

        // Compute clutter intensities
        let clutter_intensities: Vec<T> = measurements
            .iter()
            .map(|m| clutter_model.clutter_intensity(m))
            .collect();

        // Compute the CPHD update using Elementary Symmetric Functions
        let (updated_cardinality, upsilon_factors) = update_cardinality(
            &self.cardinality,
            &detection_terms,
            &missed_detection_terms,
            &likelihood_matrix,
            &clutter_intensities,
            num_measurements,
        );

        stats.max_cardinality = updated_cardinality.len().saturating_sub(1);
        stats.map_cardinality = updated_cardinality
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        // Build updated mixture
        let mut updated =
            GaussianMixture::with_capacity(self.mixture.len() * (num_measurements + 1));

        // Missed detection components (weighted by cardinality-corrected factor)
        let missed_detection_factor = if upsilon_factors.0 > T::zero() {
            upsilon_factors.1 / upsilon_factors.0
        } else {
            T::one()
        };

        for (i, component) in self.mixture.iter().enumerate() {
            let p_d = component_data[i].0;
            let missed_weight = component.weight * (T::one() - p_d) * missed_detection_factor;

            if missed_weight > T::zero() {
                updated.push(GaussianState {
                    weight: missed_weight,
                    mean: component.mean,
                    covariance: component.covariance,
                });
            }
        }

        // Detection components
        for (j, measurement) in measurements.iter().enumerate() {
            let clutter = clutter_intensities[j];

            // Sum of detection likelihoods for this measurement
            let detection_sum: T = likelihood_matrix[j]
                .iter()
                .fold(T::zero(), |acc, &x| acc + x);

            // Denominator for weight normalization
            let denom = clutter + detection_sum;

            if denom <= T::zero() {
                continue;
            }

            // Detection factor from cardinality update
            let detection_factor = if upsilon_factors.0 > T::zero() {
                upsilon_factors.2[j] / upsilon_factors.0
            } else {
                T::one()
            };

            for (i, component) in self.mixture.iter().enumerate() {
                let (_, ref predicted_meas, _, ref kalman_gain) = component_data[i];

                if let Some(gain) = kalman_gain {
                    let unnormalized_weight = likelihood_matrix[j][i];
                    let normalized_weight = (unnormalized_weight / denom) * detection_factor;

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
            }
        }

        (
            CphdFilterState {
                mixture: updated,
                cardinality: updated_cardinality,
                time_step: self.time_step,
                _phase: PhantomData,
            },
            stats,
        )
    }

    /// Updates the CPHD with a set of measurements.
    ///
    /// This method consumes the predicted state and returns an updated state.
    /// For access to numerical issue statistics, use `update_with_stats` instead.
    pub fn update<const M: usize, Obs, Clutter>(
        self,
        measurements: &[Measurement<T, M>],
        observation_model: &Obs,
        clutter_model: &Clutter,
    ) -> CphdFilterState<T, N, Updated>
    where
        Obs: ObservationModel<T, N, M>,
        Clutter: ClutterModel<T, M>,
    {
        self.update_with_stats(measurements, observation_model, clutter_model)
            .0
    }
}

// ============================================================================
// GM-CPHD Filter (with models)
// ============================================================================

/// Complete GM-CPHD filter with models.
///
/// This struct bundles the transition, observation, clutter, and birth models
/// together for convenient filter operation.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct GmCphdFilter<T, Trans, Obs, Clutter, Birth, const N: usize, const M: usize>
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
    GmCphdFilter<T, Trans, Obs, Clutter, Birth, N, M>
where
    T: RealField + Float + Copy,
    Trans: TransitionModel<T, N>,
    Obs: ObservationModel<T, N, M>,
    Clutter: ClutterModel<T, M>,
    Birth: BirthModel<T, N>,
{
    /// Creates a new GM-CPHD filter with the specified models.
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
    pub fn initial_state(&self) -> CphdFilterState<T, N, Updated> {
        CphdFilterState::new()
    }

    /// Creates an initial filter state from prior components.
    pub fn initial_state_from(
        &self,
        components: Vec<GaussianState<T, N>>,
    ) -> CphdFilterState<T, N, Updated> {
        CphdFilterState::from_mixture(GaussianMixture::from_components(components))
    }

    /// Creates an initial filter state from prior components and cardinality.
    pub fn initial_state_from_with_cardinality(
        &self,
        components: Vec<GaussianState<T, N>>,
        cardinality: Vec<T>,
    ) -> CphdFilterState<T, N, Updated> {
        CphdFilterState::from_mixture_and_cardinality(
            GaussianMixture::from_components(components),
            cardinality,
        )
    }

    /// Runs one complete predict-update cycle with statistics.
    ///
    /// Returns both the updated state and statistics about any numerical issues.
    pub fn step_with_stats(
        &self,
        state: CphdFilterState<T, N, Updated>,
        measurements: &[Measurement<T, M>],
        dt: T,
    ) -> (CphdFilterState<T, N, Updated>, CphdUpdateStats) {
        let predicted = state.predict(&self.transition, &self.birth, dt);
        predicted.update_with_stats(measurements, &self.observation, &self.clutter)
    }

    /// Runs one complete predict-update cycle.
    ///
    /// For access to numerical issue statistics, use `step_with_stats` instead.
    pub fn step(
        &self,
        state: CphdFilterState<T, N, Updated>,
        measurements: &[Measurement<T, M>],
        dt: T,
    ) -> CphdFilterState<T, N, Updated> {
        self.step_with_stats(state, measurements, dt).0
    }
}

// ============================================================================
// Cardinality Distribution Utilities
// ============================================================================

/// Generates a Poisson distribution with given mean, truncated at max_n.
#[cfg(feature = "alloc")]
fn poisson_distribution<T: RealField + Float + Copy>(mean: T, max_n: usize) -> Vec<T> {
    if mean <= T::zero() {
        let mut dist = vec![T::zero(); max_n + 1];
        dist[0] = T::one();
        return dist;
    }

    let mut dist = Vec::with_capacity(max_n + 1);
    let mut log_factorial = T::zero();

    for n in 0..=max_n {
        if n > 0 {
            log_factorial += Float::ln(T::from_usize(n).unwrap());
        }
        // P(N=n) = exp(-mean) * mean^n / n!
        let log_prob = -mean + T::from_usize(n).unwrap() * Float::ln(mean) - log_factorial;
        dist.push(Float::exp(log_prob));
    }

    // Normalize to ensure sum = 1
    let sum: T = dist.iter().fold(T::zero(), |acc, &x| acc + x);
    if sum > T::zero() {
        for p in &mut dist {
            *p /= sum;
        }
    }

    dist
}

/// Predicts the cardinality distribution through survival and birth.
#[cfg(feature = "alloc")]
fn predict_cardinality<
    T: RealField + Float + Copy,
    const N: usize,
    Trans: TransitionModel<T, N>,
>(
    cardinality: &[T],
    transition_model: &Trans,
    birth_weight: T,
) -> Vec<T> {
    // Get average survival probability (simplified - could be state-dependent)
    let p_s = transition_model.survival_probability(&StateVector::from_array([T::zero(); N]));

    let max_n = cardinality.len() + 10; // Allow for growth
    let mut predicted = vec![T::zero(); max_n];

    // Birth cardinality distribution (Poisson approximation)
    let birth_dist = poisson_distribution(birth_weight, max_n);

    // Survival cardinality distribution (binomial)
    // For each prior cardinality n, we have a binomial(n, p_s) survival distribution
    for (n, &rho_n) in cardinality.iter().enumerate() {
        if rho_n <= T::zero() {
            continue;
        }

        // Binomial survival: P(k survive | n exist) = C(n,k) * p_s^k * (1-p_s)^(n-k)
        let survival_dist = binomial_distribution(n, p_s, max_n);

        // Convolve survival with birth
        for (k_survive, &p_survive) in survival_dist.iter().enumerate() {
            for (k_birth, &p_birth) in birth_dist.iter().enumerate() {
                let k_total = k_survive + k_birth;
                if k_total < max_n {
                    predicted[k_total] += rho_n * p_survive * p_birth;
                }
            }
        }
    }

    // Normalize
    let sum: T = predicted.iter().fold(T::zero(), |acc, &x| acc + x);
    if sum > T::zero() {
        for p in &mut predicted {
            *p /= sum;
        }
    }

    // Trim trailing zeros
    while predicted.len() > 1
        && predicted
            .last()
            .is_some_and(|&x| x < T::from_f64(1e-15).unwrap())
    {
        predicted.pop();
    }

    predicted
}

/// Generates a binomial distribution B(n, p) truncated at max_k.
#[cfg(feature = "alloc")]
fn binomial_distribution<T: RealField + Float + Copy>(n: usize, p: T, max_k: usize) -> Vec<T> {
    let mut dist = vec![T::zero(); max_k.min(n + 1)];

    if n == 0 {
        if !dist.is_empty() {
            dist[0] = T::one();
        }
        return dist;
    }

    let q = T::one() - p;
    let eps = T::from_f64(1e-15).unwrap();
    let p_safe = if p < eps { eps } else { p };
    let q_safe = if q < eps { eps } else { q };

    // Use log-space for numerical stability
    let mut log_binom_coeff = T::zero(); // log(C(n, 0)) = 0
    let max_k_iter = n.min(max_k.saturating_sub(1));

    for (k, dist_k) in dist.iter_mut().enumerate().take(max_k_iter + 1) {
        if k > 0 {
            // log(C(n,k)) = log(C(n,k-1)) + log(n-k+1) - log(k)
            log_binom_coeff +=
                Float::ln(T::from_usize(n - k + 1).unwrap()) - Float::ln(T::from_usize(k).unwrap());
        }

        // P(X=k) = C(n,k) * p^k * q^(n-k)
        let log_prob = log_binom_coeff
            + T::from_usize(k).unwrap() * Float::ln(p_safe)
            + T::from_usize(n - k).unwrap() * Float::ln(q_safe);

        *dist_k = Float::exp(log_prob);
    }

    dist
}

/// Updates the cardinality distribution using the CPHD equations.
///
/// Returns the updated cardinality distribution and normalization factors
/// (upsilon_0, upsilon_1, upsilon_2[j]) for weight correction.
#[cfg(feature = "alloc")]
fn update_cardinality<T: RealField + Float + Copy>(
    cardinality: &[T],
    detection_terms: &[T],
    missed_detection_terms: &[T],
    likelihood_matrix: &[Vec<T>],
    clutter_intensities: &[T],
    num_measurements: usize,
) -> (Vec<T>, (T, T, Vec<T>)) {
    // Sum of missed detection terms
    let sum_missed: T = missed_detection_terms
        .iter()
        .fold(T::zero(), |acc, &x| acc + x);

    // For CPHD update, we need to compute ratios of elementary symmetric functions
    // The key insight is that weights are corrected by factors involving ESF ratios

    // Compute z_i = detection_term_i for ESF computation
    let z: Vec<T> = detection_terms.to_vec();

    // Elementary symmetric functions of detection terms
    let esf_detection = elementary_symmetric_function(&z);

    // Compute inner product terms for each measurement
    let inner_products: Vec<T> = likelihood_matrix
        .iter()
        .map(|row| row.iter().fold(T::zero(), |acc, &x| acc + x))
        .collect();

    // Build updated cardinality distribution
    let max_n = cardinality.len() + num_measurements;
    let mut updated_cardinality = vec![T::zero(); max_n];

    // Normalization factors for weight correction
    let mut upsilon_0 = T::zero();
    let mut upsilon_1 = T::zero();
    let mut upsilon_2 = vec![T::zero(); num_measurements];

    // For each possible number of targets n
    for (n, &rho_n) in cardinality.iter().enumerate() {
        if rho_n <= T::zero() {
            continue;
        }

        // Compute the likelihood of observing the measurements given n targets
        // This involves computing sums over all possible association hypotheses

        // For simplicity, use the approximation from the Vo & Cantoni paper
        // which factors the computation using ESF

        // Upsilon^0_n: probability of n targets with these measurements
        // Upsilon^1_n: for missed detection correction
        // Upsilon^2_n[j]: for detection correction for measurement j

        // Base term: contribution from missed detections
        let missed_power = if n < esf_detection.len() {
            esf_detection[n]
        } else {
            T::zero()
        };

        // Contribution to upsilon_0 (normalization)
        let contrib_0 = rho_n * missed_power;
        upsilon_0 += contrib_0;

        // Contribution to upsilon_1 (missed detection factor)
        if n > 0 && n - 1 < esf_detection.len() {
            let contrib_1 = rho_n * esf_detection[n - 1] * sum_missed;
            upsilon_1 += contrib_1;
        }

        // Contribution to upsilon_2[j] (detection factor for measurement j)
        for (j, &inner_prod) in inner_products.iter().enumerate() {
            if n > 0 && n - 1 < esf_detection.len() {
                let contrib_2 = rho_n * esf_detection[n - 1] * inner_prod / clutter_intensities[j];
                upsilon_2[j] += contrib_2;
            }
        }

        // Update cardinality distribution
        // The number of targets after update depends on associations
        for k in 0..=n.min(num_measurements) {
            let n_after = n; // Number of targets doesn't change in update step
            if n_after < updated_cardinality.len() {
                // Weight by measurement likelihood combination
                let factor = if k < esf_detection.len() {
                    esf_detection[k]
                } else {
                    T::zero()
                };
                updated_cardinality[n_after] += rho_n * factor;
            }
        }
    }

    // Normalize cardinality distribution
    let sum: T = updated_cardinality
        .iter()
        .fold(T::zero(), |acc, &x| acc + x);
    if sum > T::zero() {
        for p in &mut updated_cardinality {
            *p /= sum;
        }
    }

    // Trim trailing near-zero entries
    while updated_cardinality.len() > 1
        && updated_cardinality
            .last()
            .is_some_and(|&x| x < T::from_f64(1e-15).unwrap())
    {
        updated_cardinality.pop();
    }

    // Ensure upsilon_2 has correct size
    if upsilon_2.len() < num_measurements {
        upsilon_2.resize(num_measurements, T::zero());
    }

    (updated_cardinality, (upsilon_0, upsilon_1, upsilon_2))
}

// ============================================================================
// CPHD-Specific Extraction
// ============================================================================

/// Extracts targets using the MAP cardinality from the CPHD distribution.
///
/// This uses the cardinality distribution to determine how many targets
/// to extract, then selects the highest-weight components.
#[cfg(feature = "alloc")]
pub fn extract_by_map_cardinality<T: RealField + Float + Copy, const N: usize, Phase>(
    state: &CphdFilterState<T, N, Phase>,
) -> Vec<(StateVector<T, N>, T)> {
    let n_targets = state.map_cardinality();

    let mut components: Vec<_> = state.mixture.iter().map(|c| (c.mean, c.weight)).collect();

    // Sort by weight descending
    components.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));

    components.into_iter().take(n_targets).collect()
}

/// Extracts targets using the expected cardinality (rounded).
#[cfg(feature = "alloc")]
pub fn extract_by_expected_cardinality<T: RealField + Float + Copy, const N: usize, Phase>(
    state: &CphdFilterState<T, N, Phase>,
) -> Vec<(StateVector<T, N>, T)> {
    let expected = state.expected_target_count();
    let n_targets = Float::round(expected).to_usize().unwrap_or(0);

    let mut components: Vec<_> = state.mixture.iter().map(|c| (c.mean, c.weight)).collect();

    // Sort by weight descending
    components.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));

    components.into_iter().take(n_targets).collect()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{ConstantVelocity2D, FixedBirthModel, PositionSensor2D, UniformClutter2D};
    use crate::types::spaces::StateCovariance;

    #[cfg(feature = "alloc")]
    #[test]
    fn test_cphd_filter_creation() {
        let transition = ConstantVelocity2D::new(1.0_f64, 0.99);
        let observation = PositionSensor2D::new(1.0, 0.9);
        let clutter = UniformClutter2D::new(10.0, (0.0, 100.0), (0.0, 100.0));
        let birth = FixedBirthModel::<f64, 4>::new();

        let filter = GmCphdFilter::new(transition, observation, clutter, birth);
        let state = filter.initial_state();

        assert_eq!(state.mixture.len(), 0);
        assert_eq!(state.cardinality.len(), 1);
        assert!((state.cardinality[0] - 1.0).abs() < 1e-10);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_cphd_from_mixture() {
        let components = vec![
            GaussianState::new(
                1.0,
                StateVector::from_array([10.0, 20.0, 0.0, 0.0]),
                StateCovariance::identity(),
            ),
            GaussianState::new(
                1.0,
                StateVector::from_array([50.0, 60.0, 0.0, 0.0]),
                StateCovariance::identity(),
            ),
        ];

        let state: CphdFilterState<f64, 4, Updated> =
            CphdFilterState::from_mixture(GaussianMixture::from_components(components));

        assert_eq!(state.mixture.len(), 2);
        // Cardinality should be initialized based on total weight (2.0)
        assert!(state.cardinality.len() > 1);

        // Expected count should be close to 2
        let expected = state.expected_target_count();
        assert!(expected > 1.5 && expected < 2.5);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_cphd_predict() {
        let transition = ConstantVelocity2D::new(1.0_f64, 0.99);
        let observation = PositionSensor2D::new(1.0, 0.9);
        let clutter = UniformClutter2D::new(10.0, (0.0, 100.0), (0.0, 100.0));

        let mut birth = FixedBirthModel::<f64, 4>::new();
        birth.add_birth_location(
            0.1,
            StateVector::from_array([50.0, 50.0, 0.0, 0.0]),
            StateCovariance::identity(),
        );

        let filter = GmCphdFilter::new(transition, observation, clutter, birth);

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

        // Cardinality distribution should be predicted
        assert!(!predicted.cardinality.is_empty());
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_cphd_update() {
        let transition = ConstantVelocity2D::new(0.1_f64, 0.99);
        let observation = PositionSensor2D::new(1.0, 0.9);
        let clutter = UniformClutter2D::new(1.0, (0.0, 100.0), (0.0, 100.0));
        let birth = FixedBirthModel::<f64, 4>::new();

        let filter = GmCphdFilter::new(transition, observation, clutter, birth);

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
        let (updated, stats) =
            predicted.update_with_stats(&measurements, &filter.observation, &filter.clutter);

        // Should have components for missed detection and detection
        assert!(updated.mixture.len() >= 1);

        // Cardinality distribution should be updated
        assert!(!updated.cardinality.is_empty());

        // Stats should be populated
        assert_eq!(stats.singular_covariance_count, 0);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_poisson_distribution() {
        let dist = poisson_distribution(2.0_f64, 10);

        // Sum should be 1
        let sum: f64 = dist.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);

        // Mode should be at n=2
        let mode = dist
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        assert!(mode == 1 || mode == 2); // Mode of Poisson(2) is 1 or 2
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_binomial_distribution() {
        let dist = binomial_distribution(5_usize, 0.5_f64, 10);

        // Sum should be 1
        let sum: f64 = dist.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);

        // For B(5, 0.5), mode should be at n=2 or n=3
        let mode = dist
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        assert!(mode == 2 || mode == 3);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_extract_by_map_cardinality() {
        let components = vec![
            GaussianState::new(
                0.9,
                StateVector::from_array([10.0, 20.0, 0.0, 0.0]),
                StateCovariance::identity(),
            ),
            GaussianState::new(
                0.8,
                StateVector::from_array([50.0, 60.0, 0.0, 0.0]),
                StateCovariance::identity(),
            ),
            GaussianState::new(
                0.1,
                StateVector::from_array([100.0, 100.0, 0.0, 0.0]),
                StateCovariance::identity(),
            ),
        ];

        // Create state with cardinality peaked at 2
        let cardinality = vec![0.05, 0.15, 0.6, 0.15, 0.05];
        let state: CphdFilterState<f64, 4, Updated> = CphdFilterState::from_mixture_and_cardinality(
            GaussianMixture::from_components(components),
            cardinality,
        );

        let extracted = extract_by_map_cardinality(&state);

        // Should extract 2 targets (MAP cardinality)
        assert_eq!(extracted.len(), 2);

        // Should be the two highest-weight components
        assert!((extracted[0].0.index(0) - 10.0).abs() < 1e-10);
        assert!((extracted[1].0.index(0) - 50.0).abs() < 1e-10);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_cardinality_variance() {
        // Peaked distribution (low variance)
        let cardinality = vec![0.01, 0.04, 0.9, 0.04, 0.01];
        let state: CphdFilterState<f64, 4, Updated> = CphdFilterState::from_mixture_and_cardinality(
            GaussianMixture::new(),
            cardinality.clone(),
        );

        let variance = state.cardinality_variance();
        assert!(variance < 0.5); // Low variance for peaked distribution

        // Flat distribution (high variance)
        let flat_cardinality = vec![0.2, 0.2, 0.2, 0.2, 0.2];
        let flat_state: CphdFilterState<f64, 4, Updated> =
            CphdFilterState::from_mixture_and_cardinality(GaussianMixture::new(), flat_cardinality);

        let flat_variance = flat_state.cardinality_variance();
        assert!(flat_variance > variance); // Higher variance for flat distribution
    }
}
