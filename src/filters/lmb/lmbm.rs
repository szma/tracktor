//! LMBM Filter (Multi-Hypothesis)
//!
//! Implementation of the Labeled Multi-Bernoulli Multi-Hypothesis filter
//! for multi-target tracking with explicit hypothesis management.
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
use nalgebra::RealField;
use num_traits::Float;

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use super::filter::LabeledBirthModel;
use super::types::{LmbmHypothesis, LmbmState, PosteriorGrid};
use crate::models::{ClutterModel, ObservationModel, TransitionModel, effective_clutter_intensity};
use crate::types::gaussian::GaussianState;
use crate::types::labels::{BernoulliTrack, Label, LabelGenerator};
use crate::types::phase::{Predicted, UpdateStats, Updated};
use crate::types::spaces::{ComputeInnovation, Measurement, StateVector};
use crate::types::transforms::{compute_innovation_covariance, compute_kalman_gain, joseph_update};

// ============================================================================
// LMBM Filter State
// ============================================================================

#[cfg(feature = "alloc")]
fn negative_log_ratio_cost<T: RealField + Float + Copy>(
    numerator_factors: &[T],
    denominator: T,
    fallback: f64,
) -> f64 {
    if denominator <= T::zero() || numerator_factors.iter().any(|&factor| factor <= T::zero()) {
        return fallback;
    }

    let log_ratio = numerator_factors
        .iter()
        .fold(T::zero(), |sum, &factor| sum + Float::ln(factor))
        - Float::ln(denominator);
    let cost = -log_ratio;
    if Float::is_finite(cost) {
        cost.to_subset().unwrap_or(fallback)
    } else {
        fallback
    }
}

#[cfg(feature = "alloc")]
fn negative_log_probability<T: RealField + Float + Copy>(probability: T, fallback: f64) -> f64 {
    negative_log_ratio_cost(&[probability], T::one(), fallback)
}

/// The state of an LMBM filter at a particular phase.
///
/// Unlike the standard LMB filter which maintains a single set of tracks
/// with Gaussian mixture posteriors, LMBM maintains multiple hypotheses,
/// each representing a different data association history.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct LmbmFilterState<T: RealField, const N: usize, Phase> {
    /// LMBM state (mixture of hypotheses)
    pub state: LmbmState<T, N>,
    /// Label generator for new tracks
    pub label_gen: LabelGenerator,
    /// Current time step
    pub time_step: u32,
    /// Phase marker
    _phase: PhantomData<Phase>,
}

#[cfg(feature = "alloc")]
impl<T: RealField + Float + Copy, const N: usize> LmbmFilterState<T, N, Updated> {
    /// Creates a new LMBM filter state with a single empty hypothesis.
    pub fn new() -> Self {
        Self {
            state: LmbmState::with_empty_hypothesis(),
            label_gen: LabelGenerator::new(),
            time_step: 0,
            _phase: PhantomData,
        }
    }

    /// Creates an LMBM filter state from initial hypotheses.
    pub fn from_hypotheses(hypotheses: Vec<LmbmHypothesis<T, N>>) -> Self {
        Self {
            state: LmbmState { hypotheses },
            label_gen: LabelGenerator::new(),
            time_step: 0,
            _phase: PhantomData,
        }
    }

    /// Returns the number of hypotheses.
    pub fn num_hypotheses(&self) -> usize {
        self.state.num_hypotheses()
    }

    /// Predicts the LMBM density to the next time step.
    pub fn predict<Trans, Birth>(
        mut self,
        transition_model: &Trans,
        birth_model: &Birth,
        dt: T,
    ) -> LmbmFilterState<T, N, Predicted>
    where
        Trans: TransitionModel<T, N>,
        Birth: LabeledBirthModel<T, N>,
    {
        let transition_matrix = transition_model.transition_matrix(dt);
        let process_noise = transition_model.process_noise(dt);

        // Advance label generator
        self.label_gen.advance_time();

        // Generate birth tracks (same for all hypotheses)
        let birth_tracks = birth_model.birth_tracks(&mut self.label_gen);

        // Predict each hypothesis
        for hypothesis in &mut self.state.hypotheses {
            // Predict existing tracks
            for track in &mut hypothesis.tracks {
                // Update existence with survival probability
                let p_s = transition_model.survival_probability(&track.state.mean);
                track.existence *= p_s;

                // Predict state
                track.state.mean = transition_matrix.apply_state(&track.state.mean);
                track.state.covariance = transition_matrix
                    .propagate_covariance(&track.state.covariance)
                    .add(&process_noise);
            }

            // Add birth tracks to each hypothesis
            for bt in &birth_tracks {
                hypothesis.tracks.push(bt.clone());
            }
        }

        LmbmFilterState {
            state: self.state,
            label_gen: self.label_gen,
            time_step: self.time_step + 1,
            _phase: PhantomData,
        }
    }
}

#[cfg(feature = "alloc")]
impl<T: RealField + Float + Copy, const N: usize> Default for LmbmFilterState<T, N, Updated> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "alloc")]
impl<T: RealField + Float + Copy, const N: usize> LmbmFilterState<T, N, Predicted> {
    /// Updates the LMBM density with measurements using Murty's algorithm.
    ///
    /// Generates top-k assignments and creates new hypotheses for each.
    pub fn update<const M: usize, Obs, Clutter>(
        self,
        measurements: &[Measurement<T, M>],
        observation_model: &Obs,
        clutter_model: &Clutter,
        k_best: usize,
        max_hypotheses: usize,
    ) -> (LmbmFilterState<T, N, Updated>, UpdateStats)
    where
        Obs: ObservationModel<T, N, M>,
        Clutter: ClutterModel<T, M>,
    {
        let mut stats = UpdateStats::default();
        let n_meas = measurements.len();

        if self.state.hypotheses.is_empty() {
            return (
                LmbmFilterState {
                    state: LmbmState::with_empty_hypothesis(),
                    label_gen: self.label_gen,
                    time_step: self.time_step,
                    _phase: PhantomData,
                },
                stats,
            );
        }

        let obs_matrix = observation_model.observation_matrix();
        let meas_noise = observation_model.measurement_noise();

        let mut new_hypotheses: Vec<LmbmHypothesis<T, N>> = Vec::new();

        // Process each input hypothesis
        for hypothesis in &self.state.hypotheses {
            let n_tracks = hypothesis.tracks.len();

            if n_tracks == 0 {
                new_hypotheses.push(hypothesis.clone());
                continue;
            }

            if n_meas == 0 {
                // The only event is that every Bernoulli produces no
                // detection. Its probability and each conditional existence
                // must both be reflected in the resulting hypothesis.
                let mut updated_tracks = Vec::with_capacity(n_tracks);
                let mut log_weight_delta = T::zero();
                let mut possible = true;
                for track in &hypothesis.tracks {
                    let p_d = observation_model.detection_probability(&track.state.mean);
                    let miss_probability = T::one() - track.existence * p_d;
                    if miss_probability <= T::zero() {
                        possible = false;
                        break;
                    }
                    log_weight_delta += Float::ln(miss_probability);
                    updated_tracks.push(BernoulliTrack {
                        label: track.label,
                        existence: super::updaters::existence_update_miss(track.existence, p_d),
                        state: track.state.clone(),
                    });
                }
                if possible {
                    new_hypotheses.push(LmbmHypothesis {
                        log_weight: hypothesis.log_weight + log_weight_delta,
                        tracks: updated_tracks,
                    });
                }
                continue;
            }

            // Each track chooses either a shared measurement column or its
            // own private miss column. This makes misses explicit even for
            // square track/measurement inputs while preserving measurement
            // exclusivity through the assignment solver.
            let miss_column_offset = n_meas;
            let forbidden_cost = 1e20_f64;
            let mut cost_matrix =
                crate::assignment::CostMatrix::filled(n_tracks, n_meas + n_tracks, forbidden_cost);
            let mut posteriors = PosteriorGrid::new(n_tracks, n_meas);
            let mut detection_probs: Vec<T> = Vec::with_capacity(n_tracks);

            for (i, track) in hypothesis.tracks.iter().enumerate() {
                let p_d = observation_model.detection_probability(&track.state.mean);
                detection_probs.push(p_d);

                for (j, measurement) in measurements.iter().enumerate() {
                    let predicted_meas = obs_matrix.observe(&track.state.mean);
                    let innovation = measurement.innovation(predicted_meas);
                    let innovation_cov = compute_innovation_covariance(
                        &track.state.covariance,
                        &obs_matrix,
                        &meas_noise,
                    );

                    let likelihood = crate::types::gaussian::innovation_likelihood(
                        &innovation,
                        innovation_cov.as_matrix(),
                    );

                    // A detection event has probability
                    // r * p_d * L(z|x) / kappa(z). Form its cost as a
                    // difference of logs so the quotient cannot overflow.
                    let kappa =
                        effective_clutter_intensity(clutter_model.clutter_intensity(measurement));

                    if let Some(kalman_gain) =
                        compute_kalman_gain(&track.state.covariance, &obs_matrix, &innovation_cov)
                    {
                        let correction = kalman_gain.correct(&innovation);
                        let updated_mean = StateVector::from_svector(
                            track.state.mean.as_svector() + correction.as_svector(),
                        );
                        let updated_cov = joseph_update(
                            &track.state.covariance,
                            &kalman_gain,
                            &obs_matrix,
                            &meas_noise,
                        );
                        posteriors.set(i, j, (updated_mean, updated_cov, likelihood));
                        cost_matrix.set(
                            i,
                            j,
                            negative_log_ratio_cost(
                                &[track.existence, p_d, likelihood],
                                kappa,
                                forbidden_cost,
                            ),
                        );
                    } else {
                        stats.singular_covariance_count += 1;
                    }
                }

                let miss_probability = T::one() - track.existence * p_d;
                cost_matrix.set(
                    i,
                    miss_column_offset + i,
                    negative_log_probability(miss_probability, forbidden_cost),
                );
            }

            // Generate k-best assignments using Murty's algorithm
            let assignments = generate_k_best_assignments(&cost_matrix, k_best);

            // Create new hypotheses from semantically valid assignments.
            // Murty may eventually enumerate a forbidden private-column or
            // zero-probability event; those are matrix padding, not events.
            for (assignment, cost) in assignments {
                let mut updated_tracks = Vec::with_capacity(n_tracks);
                let mut valid = true;

                for (i, track) in hypothesis.tracks.iter().enumerate() {
                    let p_d = detection_probs[i];
                    let Some(column) = assignment.get(i).copied().flatten() else {
                        valid = false;
                        break;
                    };

                    let (updated_state, existence) = if column < n_meas {
                        let Some((mean, cov, likelihood)) = posteriors.get(i, column) else {
                            valid = false;
                            break;
                        };
                        if track.existence <= T::zero()
                            || p_d <= T::zero()
                            || *likelihood <= T::zero()
                        {
                            valid = false;
                            break;
                        }

                        // Conditioning on this detection event proves that the
                        // Bernoulli exists.
                        (GaussianState::new(T::one(), *mean, *cov), T::one())
                    } else if column == miss_column_offset + i {
                        let miss_probability = T::one() - track.existence * p_d;
                        if miss_probability <= T::zero() {
                            valid = false;
                            break;
                        }
                        (
                            track.state.clone(),
                            super::updaters::existence_update_miss(track.existence, p_d),
                        )
                    } else {
                        valid = false;
                        break;
                    };

                    updated_tracks.push(BernoulliTrack {
                        label: track.label,
                        existence,
                        state: updated_state,
                    });
                }

                if valid {
                    new_hypotheses.push(LmbmHypothesis {
                        log_weight: hypothesis.log_weight - T::from(cost).unwrap(),
                        tracks: updated_tracks,
                    });
                }
            }
        }

        // Prune and normalize hypotheses
        let mut new_state = LmbmState {
            hypotheses: new_hypotheses,
        };
        new_state.normalize_log_weights();
        new_state.keep_top_k(max_hypotheses);

        (
            LmbmFilterState {
                state: new_state,
                label_gen: self.label_gen,
                time_step: self.time_step,
                _phase: PhantomData,
            },
            stats,
        )
    }

    /// Returns the expected number of targets.
    pub fn expected_target_count(&self) -> T {
        // Compute marginal existence and sum
        let marginals = self.state.marginal_existence();
        marginals.iter().fold(T::zero(), |acc, (_, r)| acc + *r)
    }
}

// ============================================================================
// K-Best Assignments (Murty's Algorithm)
// ============================================================================

/// Generates k-best assignments using Murty's algorithm.
///
/// Returns a vector of (assignment, cost) pairs, where assignment[i] is
/// the measurement index assigned to track i (None for miss).
#[cfg(feature = "alloc")]
fn generate_k_best_assignments(
    cost_matrix: &crate::assignment::CostMatrix,
    k: usize,
) -> Vec<(Vec<Option<usize>>, f64)> {
    use crate::assignment::murty_k_best;

    murty_k_best(cost_matrix, k)
        .into_iter()
        .map(|assignment| (assignment.mapping, assignment.cost))
        .collect()
}

// ============================================================================
// LMBM Filter (with models)
// ============================================================================

/// Complete LMBM filter with models.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct LmbmFilter<T, Trans, Obs, Clutter, Birth, const N: usize, const M: usize>
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
    /// Number of best assignments to consider
    pub k_best: usize,
    /// Maximum number of hypotheses
    pub max_hypotheses: usize,
    _marker: PhantomData<T>,
}

#[cfg(feature = "alloc")]
impl<T, Trans, Obs, Clutter, Birth, const N: usize, const M: usize>
    LmbmFilter<T, Trans, Obs, Clutter, Birth, N, M>
where
    T: RealField + Float + Copy,
    Trans: TransitionModel<T, N>,
    Obs: ObservationModel<T, N, M>,
    Clutter: ClutterModel<T, M>,
    Birth: LabeledBirthModel<T, N>,
{
    /// Creates a new LMBM filter with the specified models.
    pub fn new(
        transition: Trans,
        observation: Obs,
        clutter: Clutter,
        birth: Birth,
        k_best: usize,
        max_hypotheses: usize,
    ) -> Self {
        Self {
            transition,
            observation,
            clutter,
            birth,
            k_best,
            max_hypotheses,
            _marker: PhantomData,
        }
    }

    /// Creates an initial filter state.
    pub fn initial_state(&self) -> LmbmFilterState<T, N, Updated> {
        LmbmFilterState::new()
    }

    /// Runs one complete predict-update cycle.
    pub fn step(
        &self,
        state: LmbmFilterState<T, N, Updated>,
        measurements: &[Measurement<T, M>],
        dt: T,
    ) -> (LmbmFilterState<T, N, Updated>, UpdateStats) {
        let predicted = state.predict(&self.transition, &self.birth, dt);
        predicted.update(
            measurements,
            &self.observation,
            &self.clutter,
            self.k_best,
            self.max_hypotheses,
        )
    }
}

// ============================================================================
// State Extraction
// ============================================================================

/// Extracts the best hypothesis from an LMBM filter state.
#[cfg(feature = "alloc")]
pub fn extract_best_hypothesis<T: RealField + Float + Copy, const N: usize, Phase>(
    state: &LmbmFilterState<T, N, Phase>,
) -> Option<&LmbmHypothesis<T, N>> {
    state
        .state
        .hypotheses
        .iter()
        .max_by(|a, b| a.log_weight.partial_cmp(&b.log_weight).unwrap())
}

/// Extracts marginal state estimates from an LMBM filter.
#[cfg(feature = "alloc")]
pub fn extract_lmbm_estimates<T: RealField + Float + Copy, const N: usize, Phase>(
    state: &LmbmFilterState<T, N, Phase>,
    existence_threshold: T,
) -> Vec<(Label, StateVector<T, N>, T)> {
    use alloc::collections::BTreeMap;

    if state.state.hypotheses.is_empty() {
        return Vec::new();
    }

    // Normalize hypothesis weights
    let max_log = state
        .state
        .hypotheses
        .iter()
        .map(|h| h.log_weight)
        .fold(T::neg_infinity(), |a, b| if a > b { a } else { b });

    let weights: Vec<T> = state
        .state
        .hypotheses
        .iter()
        .map(|h| Float::exp(h.log_weight - max_log))
        .collect();
    let total_weight: T = weights.iter().fold(T::zero(), |acc, &w| acc + w);

    // Compute marginal states
    let mut state_accum: BTreeMap<Label, (StateVector<T, N>, T)> = BTreeMap::new();

    for (hypothesis, &weight) in state.state.hypotheses.iter().zip(&weights) {
        let normalized_weight = weight / total_weight;
        for track in &hypothesis.tracks {
            let entry = state_accum
                .entry(track.label)
                .or_insert((StateVector::zeros(), T::zero()));

            let w = normalized_weight * track.existence;
            entry.0 = StateVector::from_svector(
                entry.0.as_svector() + track.state.mean.as_svector().scale(w),
            );
            entry.1 += w;
        }
    }

    // Normalize and filter by threshold
    state_accum
        .into_iter()
        .filter_map(|(label, (state_sum, existence))| {
            if existence >= existence_threshold {
                let normalized_state =
                    StateVector::from_svector(state_sum.as_svector().scale(T::one() / existence));
                Some((label, normalized_state, existence))
            } else {
                None
            }
        })
        .collect()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "alloc")]
    #[test]
    fn test_lmbm_filter_state_creation() {
        let state: LmbmFilterState<f64, 4, Updated> = LmbmFilterState::new();
        assert_eq!(state.num_hypotheses(), 1);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_out_of_support_clutter_still_allows_lmbm_target_association_f32() {
        use crate::filters::lmb::NoBirthModel;
        use crate::models::{ConstantVelocity2D, PositionSensor2D, UniformClutter2D};
        use crate::types::spaces::StateCovariance;

        let track = BernoulliTrack::new(
            Label::new(0, 0),
            0.9_f32,
            GaussianState::new(
                1.0,
                StateVector::from_array([20.0, 20.0, 0.0, 0.0]),
                StateCovariance::identity().scale(1.0e-8),
            ),
        );
        let predicted = LmbmFilterState::from_hypotheses(vec![LmbmHypothesis::new(
            0.0,
            vec![track],
        )])
        .predict(&ConstantVelocity2D::new(0.0, 1.0), &NoBirthModel, 0.0);
        let (updated, _) = predicted.update(
            &[
                Measurement::from_array([20.0, 20.0]),
                Measurement::from_array([5.0, 5.0]),
            ],
            &PositionSensor2D::new(1.0e-8, 0.9),
            &UniformClutter2D::new(1.0, (0.0, 10.0), (0.0, 10.0)),
            2,
            2,
        );

        let best = extract_best_hypothesis(&updated).unwrap();
        assert!(*best.tracks[0].state.mean.index(0) > 18.0);
    }

    #[cfg(feature = "alloc")]
    fn predicted_state(
        hypotheses: Vec<LmbmHypothesis<f64, 4>>,
    ) -> LmbmFilterState<f64, 4, Predicted> {
        LmbmFilterState {
            state: LmbmState { hypotheses },
            label_gen: LabelGenerator::new(),
            time_step: 0,
            _phase: PhantomData,
        }
    }

    #[cfg(feature = "alloc")]
    fn test_track(label: u32, existence: f64, x: f64) -> BernoulliTrack<f64, 4> {
        use crate::types::spaces::StateCovariance;

        BernoulliTrack::new(
            Label::new(0, label),
            existence,
            GaussianState::new(
                1.0,
                StateVector::from_array([x, 0.0, 0.0, 0.0]),
                StateCovariance::identity(),
            ),
        )
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_one_track_one_measurement_has_detection_and_miss_hypotheses() {
        use crate::models::{PositionSensor2D, UniformClutter2D};

        let predicted = predicted_state(vec![LmbmHypothesis::new(
            0.0,
            vec![test_track(0, 0.6, 0.0)],
        )]);
        let (updated, _) = predicted.update(
            &[Measurement::from_array([0.0, 0.0])],
            &PositionSensor2D::new(1.0, 0.5),
            &UniformClutter2D::new(1.0, (-5.0, 5.0), (-5.0, 5.0)),
            2,
            2,
        );

        assert_eq!(updated.state.hypotheses.len(), 2);
        assert!(
            updated
                .state
                .hypotheses
                .iter()
                .any(|hypothesis| hypothesis.tracks[0].existence == 1.0)
        );
        assert!(
            updated
                .state
                .hypotheses
                .iter()
                .any(|hypothesis| (hypothesis.tracks[0].existence - 3.0 / 7.0).abs() < 1.0e-12)
        );
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_zero_existence_cannot_produce_detection_hypothesis() {
        use crate::models::{PositionSensor2D, UniformClutter2D};

        let predicted = predicted_state(vec![LmbmHypothesis::new(
            0.0,
            vec![test_track(0, 0.0, 0.0)],
        )]);
        let (updated, _) = predicted.update(
            &[Measurement::from_array([0.0, 0.0])],
            &PositionSensor2D::new(1.0, 0.5),
            &UniformClutter2D::new(1.0, (-5.0, 5.0), (-5.0, 5.0)),
            2,
            2,
        );

        assert_eq!(updated.state.hypotheses.len(), 1);
        assert_eq!(updated.state.hypotheses[0].tracks[0].existence, 0.0);
        assert_eq!(updated.state.hypotheses[0].log_weight, 0.0);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_unequal_existence_detection_odds_match_event_probabilities() {
        use crate::models::{PositionSensor2D, UniformClutter2D};

        let predicted = predicted_state(vec![LmbmHypothesis::new(
            0.0,
            vec![test_track(0, 0.2, 0.0), test_track(1, 0.8, 0.0)],
        )]);
        let (updated, _) = predicted.update(
            &[Measurement::from_array([0.0, 0.0])],
            &PositionSensor2D::new(1.0, 0.5),
            &UniformClutter2D::new(1.0, (-5.0, 5.0), (-5.0, 5.0)),
            3,
            3,
        );

        assert_eq!(updated.state.hypotheses.len(), 3);
        let assigned_zero = updated
            .state
            .hypotheses
            .iter()
            .find(|hypothesis| hypothesis.tracks[0].existence == 1.0)
            .unwrap();
        let assigned_one = updated
            .state
            .hypotheses
            .iter()
            .find(|hypothesis| hypothesis.tracks[1].existence == 1.0)
            .unwrap();

        // L and kappa are common, so the event odds are
        // [r0 * (1-r1*p_d)] / [(1-r0*p_d) * r1].
        let expected_log_odds = Float::ln((0.2 * 0.6) / (0.9 * 0.8));
        let actual_log_odds = assigned_zero.log_weight - assigned_one.log_weight;
        assert!((actual_log_odds - expected_log_odds).abs() < 1.0e-12);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_no_measurement_weights_include_miss_event_probabilities() {
        use crate::models::{PositionSensor2D, UniformClutter2D};

        let predicted = predicted_state(vec![
            LmbmHypothesis::new(0.0, vec![test_track(0, 0.2, 0.0)]),
            LmbmHypothesis::new(0.0, vec![test_track(0, 0.8, 0.0)]),
        ]);
        let (updated, _) = predicted.update(
            &[],
            &PositionSensor2D::new(1.0, 0.5),
            &UniformClutter2D::new(1.0, (-5.0, 5.0), (-5.0, 5.0)),
            2,
            2,
        );

        assert_eq!(updated.state.hypotheses.len(), 2);
        let low_existence = updated
            .state
            .hypotheses
            .iter()
            .find(|hypothesis| hypothesis.tracks[0].existence < 0.2)
            .unwrap();
        let high_existence = updated
            .state
            .hypotheses
            .iter()
            .find(|hypothesis| hypothesis.tracks[0].existence > 0.6)
            .unwrap();
        let actual_log_odds = low_existence.log_weight - high_existence.log_weight;
        assert!((actual_log_odds - Float::ln(0.9_f64 / 0.6)).abs() < 1.0e-12);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_detection_conditioned_existence_is_one() {
        use crate::models::{PositionSensor2D, UniformClutter2D};

        let predicted = predicted_state(vec![LmbmHypothesis::new(
            0.0,
            vec![test_track(0, 0.01, 0.0)],
        )]);
        let (updated, _) = predicted.update(
            &[Measurement::from_array([0.0, 0.0])],
            &PositionSensor2D::new(1.0, 0.1),
            &UniformClutter2D::new(1.0, (-5.0, 5.0), (-5.0, 5.0)),
            2,
            2,
        );

        let detection = updated
            .state
            .hypotheses
            .iter()
            .find(|hypothesis| hypothesis.tracks[0].state.covariance.as_matrix()[(0, 0)] < 0.75)
            .unwrap();
        assert_eq!(detection.tracks[0].existence, 1.0);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_lmbm_hypothesis_creation() {
        let label = Label::new(0, 0);
        let mean: StateVector<f64, 4> = StateVector::from_array([1.0, 2.0, 0.0, 0.0]);
        let cov: crate::types::spaces::StateCovariance<f64, 4> =
            crate::types::spaces::StateCovariance::identity();
        let state = GaussianState::new(1.0, mean, cov);
        let track = BernoulliTrack::new(label, 0.9, state);

        let hypothesis = LmbmHypothesis::new(0.0, vec![track]);
        assert_eq!(hypothesis.num_tracks(), 1);
    }
}
