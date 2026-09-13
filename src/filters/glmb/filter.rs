//! GLMB Filter Implementation
//!
//! Generalized Labeled Multi-Bernoulli filter for multi-target tracking.
//!
//! This module implements the delta-GLMB filter, which maintains a mixture of
//! hypotheses where each hypothesis represents a specific combination of:
//! - Label set I (which tracks exist)
//! - Association history ξ (which measurements were associated to which tracks)
//!
//! # Type Safety
//!
//! The filter uses phase markers (`Predicted`/`Updated`) to ensure correct
//! operation ordering at compile time:
//! - `predict()` consumes an `Updated` state and returns a `Predicted` state
//! - `update()` consumes a `Predicted` state and returns an `Updated` state
//!
//! # Reference
//!
//! Vo, B.-T., & Vo, B.-N. (2013). "Labeled Random Finite Sets and
//! Multi-Object Conjugate Priors"

#![allow(clippy::type_complexity)]

use core::marker::PhantomData;
use nalgebra::{ComplexField, RealField};
use num_traits::Float;

use alloc::vec;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use super::types::{GlmbDensity, GlmbHypothesis, GlmbTrack, GlmbTruncationConfig};
use crate::filters::lmb::LabeledBirthModel;
use crate::models::effective_clutter_intensity;
use crate::models::{ClutterModel, NonlinearObservationModel, ObservationModel, TransitionModel};
use crate::types::gaussian::GaussianState;
use crate::types::labels::{Label, LabelGenerator};
use crate::types::phase::{Predicted, UpdateStats, Updated};
use crate::types::spaces::{ComputeInnovation, Measurement, StateCovariance, StateVector};
use crate::types::transforms::{compute_innovation_covariance, compute_kalman_gain, joseph_update};

// ============================================================================
// GLMB Filter State
// ============================================================================

/// The state of a GLMB filter at a particular phase.
///
/// The `Phase` parameter encodes whether this is a predicted or updated state,
/// ensuring correct operation ordering at compile time.
///
/// # Type Parameters
///
/// - `T`: Scalar type (e.g., `f64`)
/// - `N`: State dimension
/// - `Phase`: Either `Predicted` or `Updated`
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct GlmbFilterState<T: RealField, const N: usize, Phase> {
    /// GLMB density
    pub density: GlmbDensity<T, N>,
    /// Label generator for new tracks
    pub label_gen: LabelGenerator,
    /// Current time step
    pub time_step: u32,
    /// Phase marker
    _phase: PhantomData<Phase>,
}

#[cfg(feature = "alloc")]
impl<T: RealField + Float + Copy, const N: usize> GlmbFilterState<T, N, Updated> {
    /// Creates a new filter state with a single empty hypothesis.
    pub fn new() -> Self {
        Self {
            density: GlmbDensity::with_empty_hypothesis(),
            label_gen: LabelGenerator::new(),
            time_step: 0,
            _phase: PhantomData,
        }
    }

    /// Creates a filter state from initial hypotheses.
    pub fn from_hypotheses(hypotheses: Vec<GlmbHypothesis<T, N>>) -> Self {
        Self {
            density: GlmbDensity { hypotheses },
            label_gen: LabelGenerator::new(),
            time_step: 0,
            _phase: PhantomData,
        }
    }

    /// Creates a filter state from initial tracks in a single hypothesis.
    pub fn from_tracks(tracks: Vec<GlmbTrack<T, N>>) -> Self {
        let associations = vec![None; tracks.len()];
        let hypothesis = GlmbHypothesis::new(T::zero(), tracks, associations);
        Self::from_hypotheses(vec![hypothesis])
    }

    /// Returns the number of hypotheses.
    #[inline]
    pub fn num_hypotheses(&self) -> usize {
        self.density.num_hypotheses()
    }

    /// Returns the expected number of targets (sum of marginal existences).
    pub fn expected_target_count(&self) -> T {
        self.density
            .marginal_existence()
            .iter()
            .fold(T::zero(), |acc, (_, r)| acc + *r)
    }

    /// Returns the MAP (most likely) cardinality.
    pub fn map_cardinality(&self) -> usize {
        self.density.map_cardinality()
    }

    /// Joint prediction and update in a single step (fast GLMB algorithm).
    ///
    /// This implements the efficient algorithm from:
    /// "A fast implementation of the generalized labeled multi-Bernoulli filter
    /// with joint prediction and update" (Vo & Vo)
    ///
    /// Key optimizations over separate predict() + update():
    /// 1. Unique tracks across all hypotheses are collected and predicted once
    /// 2. Posteriors are computed once per unique (track, measurement) pair
    /// 3. Hypotheses sharing the same track set share computation
    ///
    /// # Arguments
    ///
    /// - `measurements`: Measurements at current time step
    /// - `transition_model`: State transition model
    /// - `observation_model`: Linear observation model
    /// - `clutter_model`: Clutter model for false alarm density
    /// - `birth_model`: Birth model for new track generation
    /// - `config`: Update configuration (truncation, k_best, dt)
    pub fn joint_predict_and_update<const M: usize, Trans, Obs, Clutter, Birth>(
        mut self,
        measurements: &[Measurement<T, M>],
        transition_model: &Trans,
        observation_model: &Obs,
        clutter_model: &Clutter,
        birth_model: &Birth,
        config: &super::types::GlmbUpdateConfig<T>,
    ) -> (GlmbFilterState<T, N, Updated>, UpdateStats)
    where
        Trans: TransitionModel<T, N>,
        Obs: ObservationModel<T, N, M>,
        Clutter: ClutterModel<T, M>,
        Birth: LabeledBirthModel<T, N>,
    {
        use alloc::collections::BTreeMap;

        let mut stats = UpdateStats::default();

        // Advance label generator
        self.label_gen.advance_time();

        // Step 1: Collect all unique tracks across all hypotheses and predict them
        let mut unique_tracks: BTreeMap<Label, GlmbTrack<T, N>> = BTreeMap::new();

        for hypothesis in &self.density.hypotheses {
            for track in &hypothesis.tracks {
                unique_tracks
                    .entry(track.label)
                    .or_insert_with(|| track.clone());
            }
        }

        // Predict each unique track
        let transition_matrix = transition_model.transition_matrix(config.dt);
        let process_noise = transition_model.process_noise(config.dt);

        let mut predicted_tracks: BTreeMap<Label, (GlmbTrack<T, N>, T)> = BTreeMap::new();

        for (label, track) in unique_tracks {
            let p_s = transition_model.survival_probability(&track.state.mean);

            let predicted_mean = transition_matrix.apply_state(&track.state.mean);
            let predicted_cov = transition_matrix
                .propagate_covariance(&track.state.covariance)
                .add(&process_noise);

            let predicted_track = GlmbTrack {
                label,
                state: GaussianState::new(track.state.weight, predicted_mean, predicted_cov),
            };

            predicted_tracks.insert(label, (predicted_track, p_s));
        }

        // Step 2: Generate birth tracks
        let mut birth_label_gen = self.label_gen.clone();
        let birth_tracks = birth_model.birth_tracks(&mut birth_label_gen);

        // Step 3: Build list of all tracks (predicted existing + birth)
        let obs_matrix = observation_model.observation_matrix();
        let meas_noise = observation_model.measurement_noise();

        let mut all_tracks: Vec<(GlmbTrack<T, N>, bool, T)> = predicted_tracks
            .values()
            .map(|(t, p_s)| (t.clone(), false, *p_s))
            .collect();

        for bt in &birth_tracks {
            all_tracks.push((GlmbTrack::from_bernoulli(bt), true, bt.existence));
        }

        // Step 4: Compute posteriors for all (track, measurement) pairs once
        let n_tracks = all_tracks.len();
        let n_meas = measurements.len();

        // posteriors[track_idx][meas_idx] = Some((mean, cov, likelihood))
        let mut posteriors: Vec<Vec<Option<(StateVector<T, N>, StateCovariance<T, N>, T)>>> =
            vec![vec![None; n_meas]; n_tracks];

        let mut detection_probs: Vec<T> = Vec::with_capacity(n_tracks);

        for (i, (track, _, _)) in all_tracks.iter().enumerate() {
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

                if likelihood <= T::zero() {
                    stats.zero_likelihood_count += 1;
                    continue;
                }

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

                    posteriors[i][j] = Some((updated_mean, updated_cov, likelihood));
                } else {
                    stats.singular_covariance_count += 1;
                }
            }
        }

        // Step 5: Build label-to-index mapping
        let label_to_idx: BTreeMap<Label, usize> = all_tracks
            .iter()
            .enumerate()
            .map(|(i, (t, _, _))| (t.label, i))
            .collect();

        // Step 6: Group hypotheses by track set
        // Hypotheses with the same set of surviving tracks can share the cost matrix
        let mut hypothesis_groups: BTreeMap<Vec<Label>, Vec<(usize, T)>> = BTreeMap::new();

        for (hyp_idx, hypothesis) in self.density.hypotheses.iter().enumerate() {
            let mut track_labels: Vec<Label> =
                hypothesis.tracks.iter().map(|track| track.label).collect();
            track_labels.sort();

            hypothesis_groups
                .entry(track_labels)
                .or_default()
                .push((hyp_idx, hypothesis.log_weight));
        }

        // Step 7: Process each hypothesis group
        let mut new_hypotheses: Vec<GlmbHypothesis<T, N>> = Vec::new();

        for (track_labels, group) in hypothesis_groups {
            // Build cost matrix for this track set + birth tracks
            // Rows: existing tracks from this group + all birth tracks
            // Columns: measurements + private absent and missed columns

            let n_existing = track_labels.len();
            let n_birth = birth_tracks.len();
            let n_group_tracks = n_existing + n_birth;

            if n_group_tracks == 0 && n_meas == 0 {
                // Empty hypothesis
                for (_, log_weight) in &group {
                    new_hypotheses.push(GlmbHypothesis::empty(*log_weight));
                }
                continue;
            }

            if n_group_tracks == 0 {
                // The common clutter set likelihood is factored out of every
                // hypothesis; association costs below use target/kappa ratios.
                for (_, log_weight) in &group {
                    new_hypotheses.push(GlmbHypothesis::empty(*log_weight));
                }
                continue;
            }

            // Map group track indices to global indices
            let mut group_track_indices: Vec<usize> = track_labels
                .iter()
                .filter_map(|l| label_to_idx.get(l).copied())
                .collect();

            // Add birth track indices
            for i in 0..n_birth {
                let birth_idx = n_tracks - n_birth + i;
                group_track_indices.push(birth_idx);
            }

            // Build cost matrix
            // Each row has a private absent/death column and a private
            // present-but-missed column in addition to shared measurements.
            let n_cols = n_meas + 2 * n_group_tracks;
            let mut cost_matrix = crate::assignment::CostMatrix::zeros(n_group_tracks, n_cols);
            let large_cost = 1e10_f64;

            for (row, &global_idx) in group_track_indices.iter().enumerate() {
                let (_, _, existence_probability) = &all_tracks[global_idx];
                let p_d = detection_probs[global_idx];

                // Detection columns
                for (j, measurement) in measurements.iter().enumerate() {
                    let kappa =
                        effective_clutter_intensity(clutter_model.clutter_intensity(measurement));

                    let cost = if let Some((_, _, likelihood)) = posteriors[global_idx][j] {
                        if likelihood > T::zero() {
                            let ratio = p_d * likelihood * *existence_probability / kappa;
                            if ratio > T::zero() {
                                -ComplexField::ln(ratio).to_subset().unwrap_or(large_cost)
                            } else {
                                large_cost
                            }
                        } else {
                            large_cost
                        }
                    } else {
                        large_cost
                    };

                    cost_matrix.set(row, j, cost);
                }

                let absent_probability = T::one() - *existence_probability;
                let absent_col = n_meas + row;
                cost_matrix.set(
                    row,
                    absent_col,
                    -ComplexField::ln(absent_probability + T::from_f64(1e-300).unwrap())
                        .to_subset()
                        .unwrap_or(large_cost),
                );

                let missed_probability = *existence_probability * (T::one() - p_d);
                let missed_col = n_meas + n_group_tracks + row;
                cost_matrix.set(
                    row,
                    missed_col,
                    -ComplexField::ln(missed_probability + T::from_f64(1e-300).unwrap())
                        .to_subset()
                        .unwrap_or(large_cost),
                );

                // Private columns can only be selected by their own row.
                for other_row in 0..n_group_tracks {
                    if other_row != row {
                        cost_matrix.set(row, n_meas + other_row, large_cost);
                        cost_matrix.set(row, n_meas + n_group_tracks + other_row, large_cost);
                    }
                }
            }

            // Run Murty's k-best
            let assignments =
                crate::assignment::hungarian::murty_k_best(&cost_matrix, config.k_best);

            // Generate hypotheses from assignments for each input hypothesis in this group
            for (_hyp_idx, base_log_weight) in &group {
                for assignment in &assignments {
                    let mut output_tracks: Vec<GlmbTrack<T, N>> = Vec::new();
                    let mut associations: Vec<Option<usize>> = Vec::new();

                    for (row, col_opt) in assignment.mapping.iter().enumerate() {
                        let col = match col_opt {
                            Some(c) => *c,
                            None => continue,
                        };
                        let global_idx = group_track_indices[row];
                        let (track, _, _) = &all_tracks[global_idx];

                        if col < n_meas {
                            if let Some((updated_mean, updated_cov, _)) =
                                &posteriors[global_idx][col]
                            {
                                output_tracks.push(GlmbTrack {
                                    label: track.label,
                                    state: GaussianState::new(
                                        track.state.weight,
                                        *updated_mean,
                                        *updated_cov,
                                    ),
                                });
                                associations.push(Some(col));
                            }
                        } else if col >= n_meas + n_group_tracks {
                            // The track exists/survives but was not detected.
                            output_tracks.push(track.clone());
                            associations.push(None);
                        }
                        // The first private block denotes non-birth/death and
                        // therefore contributes no output track.
                    }

                    new_hypotheses.push(GlmbHypothesis {
                        log_weight: *base_log_weight - T::from(assignment.cost).unwrap(),
                        tracks: output_tracks,
                        current_association: associations,
                    });
                }
            }
        }

        // Apply truncation
        let mut new_density = GlmbDensity {
            hypotheses: new_hypotheses,
        };
        new_density.normalize_log_weights();
        new_density.prune_by_weight(config.truncation.log_weight_threshold);

        if let Some(k_per_card) = config.truncation.max_per_cardinality {
            new_density.keep_top_k_per_cardinality(k_per_card);
        }

        new_density.keep_top_k(config.truncation.max_hypotheses);

        // Ensure at least one hypothesis exists
        if new_density.is_empty() {
            new_density.push(GlmbHypothesis::empty(T::zero()));
        }

        (
            GlmbFilterState {
                density: new_density,
                label_gen: birth_label_gen,
                time_step: self.time_step + 1,
                _phase: PhantomData,
            },
            stats,
        )
    }

    /// Predicts the GLMB density to the next time step.
    ///
    /// This method consumes the updated state and returns a predicted state.
    /// It applies the transition model to each track and updates survival
    /// probabilities.
    ///
    /// Note: Birth tracks are incorporated during the update step (joint
    /// prediction-update formulation).
    pub fn predict<Trans, Birth>(
        mut self,
        transition_model: &Trans,
        _birth_model: &Birth, // Used in update, kept here for API consistency
        dt: T,
    ) -> GlmbFilterState<T, N, Predicted>
    where
        Trans: TransitionModel<T, N>,
        Birth: LabeledBirthModel<T, N>,
    {
        let transition_matrix = transition_model.transition_matrix(dt);
        let process_noise = transition_model.process_noise(dt);

        // Advance label generator to new time step
        self.label_gen.advance_time();

        // Branch each labeled track into its survival and death alternatives.
        let mut predicted_hypotheses = Vec::new();
        for hypothesis in &self.density.hypotheses {
            let mut branches = vec![GlmbHypothesis::empty(hypothesis.log_weight)];

            for track in &hypothesis.tracks {
                let p_s = transition_model.survival_probability(&track.state.mean);
                assert!(
                    p_s >= T::zero() && p_s <= T::one(),
                    "Survival probability must be in [0, 1]"
                );

                let predicted_track = GlmbTrack {
                    label: track.label,
                    state: GaussianState::new(
                        track.state.weight,
                        transition_matrix.apply_state(&track.state.mean),
                        transition_matrix
                            .propagate_covariance(&track.state.covariance)
                            .add(&process_noise),
                    ),
                };

                let mut next = Vec::with_capacity(branches.len() * 2);
                for branch in branches {
                    let death_probability = T::one() - p_s;
                    if death_probability > T::zero() {
                        let mut death = branch.clone();
                        death.log_weight += ComplexField::ln(death_probability);
                        next.push(death);
                    }
                    if p_s > T::zero() {
                        let mut survival = branch;
                        survival.log_weight += ComplexField::ln(p_s);
                        survival.tracks.push(predicted_track.clone());
                        survival.current_association.push(None);
                        next.push(survival);
                    }
                }
                branches = next;
            }

            predicted_hypotheses.extend(branches);
        }

        let mut density = GlmbDensity {
            hypotheses: predicted_hypotheses,
        };
        density.normalize_log_weights();

        GlmbFilterState {
            density,
            label_gen: self.label_gen,
            time_step: self.time_step + 1,
            _phase: PhantomData,
        }
    }
}

#[cfg(feature = "alloc")]
impl<T: RealField + Float + Copy, const N: usize> Default for GlmbFilterState<T, N, Updated> {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// GLMB Update Implementation
// ============================================================================

#[cfg(feature = "alloc")]
impl<T: RealField + Float + Copy, const N: usize> GlmbFilterState<T, N, Predicted> {
    /// Returns the expected number of targets (before update).
    pub fn expected_target_count(&self) -> T {
        self.density
            .marginal_existence()
            .iter()
            .fold(T::zero(), |acc, (_, r)| acc + *r)
    }

    /// Updates the GLMB density with measurements using joint prediction-update.
    ///
    /// Implements the Vo & Vo (2013) algorithm using Murty's k-best for
    /// hypothesis generation.
    ///
    /// # Arguments
    ///
    /// - `measurements`: Measurements at current time step
    /// - `observation_model`: Linear observation model
    /// - `clutter_model`: Clutter model for false alarm density
    /// - `birth_model`: Birth model for new track generation
    /// - `truncation`: Truncation configuration
    /// - `k_best`: Number of best assignments to generate per hypothesis
    pub fn update<const M: usize, Obs, Clutter, Birth>(
        self,
        measurements: &[Measurement<T, M>],
        observation_model: &Obs,
        clutter_model: &Clutter,
        birth_model: &Birth,
        truncation: &GlmbTruncationConfig<T>,
        k_best: usize,
    ) -> (GlmbFilterState<T, N, Updated>, UpdateStats)
    where
        Obs: ObservationModel<T, N, M>,
        Clutter: ClutterModel<T, M>,
        Birth: LabeledBirthModel<T, N>,
    {
        let mut stats = UpdateStats::default();

        let obs_matrix = observation_model.observation_matrix();
        let meas_noise = observation_model.measurement_noise();

        // Generate birth tracks for this timestep
        let mut birth_label_gen = self.label_gen.clone();
        let birth_tracks = birth_model.birth_tracks(&mut birth_label_gen);

        let mut new_hypotheses: Vec<GlmbHypothesis<T, N>> = Vec::new();

        let ctx = HypothesisContext {
            measurements,
            birth_tracks: &birth_tracks,
            meas_noise: &meas_noise,
            k_best,
        };

        // Process each input hypothesis
        for hypothesis in &self.density.hypotheses {
            let generated = generate_hypotheses_from_assignment(
                hypothesis,
                &ctx,
                observation_model,
                clutter_model,
                &obs_matrix,
                &mut stats,
            );

            new_hypotheses.extend(generated);
        }

        // Apply truncation
        let mut new_density = GlmbDensity {
            hypotheses: new_hypotheses,
        };
        new_density.normalize_log_weights();
        new_density.prune_by_weight(truncation.log_weight_threshold);

        if let Some(k_per_card) = truncation.max_per_cardinality {
            new_density.keep_top_k_per_cardinality(k_per_card);
        }

        new_density.keep_top_k(truncation.max_hypotheses);

        // Ensure at least one hypothesis exists
        if new_density.is_empty() {
            new_density.push(GlmbHypothesis::empty(T::zero()));
        }

        (
            GlmbFilterState {
                density: new_density,
                label_gen: birth_label_gen,
                time_step: self.time_step,
                _phase: PhantomData,
            },
            stats,
        )
    }

    /// Updates the GLMB density with measurements using EKF-style linearization.
    ///
    /// This method is for nonlinear observation models where the observation
    /// function h(x) and its Jacobian depend on the state.
    ///
    /// # Arguments
    ///
    /// - `measurements`: Measurements in the sensor's native coordinate system
    /// - `observation_model`: Nonlinear observation model
    /// - `clutter_model`: Clutter model in the same coordinate system as measurements
    /// - `birth_model`: Birth model for new track generation
    /// - `truncation`: Truncation configuration
    /// - `k_best`: Number of best assignments to generate per hypothesis
    pub fn update_ekf<const M: usize, Obs, Clutter, Birth>(
        self,
        measurements: &[Measurement<T, M>],
        observation_model: &Obs,
        clutter_model: &Clutter,
        birth_model: &Birth,
        truncation: &GlmbTruncationConfig<T>,
        k_best: usize,
    ) -> (GlmbFilterState<T, N, Updated>, UpdateStats)
    where
        Obs: NonlinearObservationModel<T, N, M>,
        Clutter: ClutterModel<T, M>,
        Birth: LabeledBirthModel<T, N>,
    {
        let mut stats = UpdateStats::default();

        let meas_noise = observation_model.measurement_noise();

        // Generate birth tracks for this timestep
        let mut birth_label_gen = self.label_gen.clone();
        let birth_tracks = birth_model.birth_tracks(&mut birth_label_gen);

        let mut new_hypotheses: Vec<GlmbHypothesis<T, N>> = Vec::new();

        let ctx = HypothesisContext {
            measurements,
            birth_tracks: &birth_tracks,
            meas_noise: &meas_noise,
            k_best,
        };

        // Process each input hypothesis
        for hypothesis in &self.density.hypotheses {
            let generated = generate_hypotheses_ekf(
                hypothesis,
                &ctx,
                observation_model,
                clutter_model,
                &mut stats,
            );

            new_hypotheses.extend(generated);
        }

        // Apply truncation
        let mut new_density = GlmbDensity {
            hypotheses: new_hypotheses,
        };
        new_density.normalize_log_weights();
        new_density.prune_by_weight(truncation.log_weight_threshold);

        if let Some(k_per_card) = truncation.max_per_cardinality {
            new_density.keep_top_k_per_cardinality(k_per_card);
        }

        new_density.keep_top_k(truncation.max_hypotheses);

        // Ensure at least one hypothesis exists
        if new_density.is_empty() {
            new_density.push(GlmbHypothesis::empty(T::zero()));
        }

        (
            GlmbFilterState {
                density: new_density,
                label_gen: birth_label_gen,
                time_step: self.time_step,
                _phase: PhantomData,
            },
            stats,
        )
    }
}

// ============================================================================
// Hypothesis Generation (Linear)
// ============================================================================

/// Pre-computed context shared across hypothesis generation.
#[cfg(feature = "alloc")]
struct HypothesisContext<'a, T: RealField, const N: usize, const M: usize> {
    measurements: &'a [Measurement<T, M>],
    birth_tracks: &'a [crate::types::labels::BernoulliTrack<T, N>],
    meas_noise: &'a crate::types::spaces::MeasurementCovariance<T, M>,
    k_best: usize,
}

/// Generates new hypotheses from a single input hypothesis using Murty's algorithm.
#[cfg(feature = "alloc")]
fn generate_hypotheses_from_assignment<
    T: RealField + Float + Copy,
    const N: usize,
    const M: usize,
    Obs: ObservationModel<T, N, M>,
    Clutter: ClutterModel<T, M>,
>(
    hypothesis: &GlmbHypothesis<T, N>,
    ctx: &HypothesisContext<'_, T, N, M>,
    observation_model: &Obs,
    clutter_model: &Clutter,
    obs_matrix: &crate::types::transforms::ObservationMatrix<T, M, N>,
    stats: &mut UpdateStats,
) -> Vec<GlmbHypothesis<T, N>> {
    let measurements = ctx.measurements;
    let birth_tracks = ctx.birth_tracks;
    let meas_noise = ctx.meas_noise;
    let k_best = ctx.k_best;
    let n_meas = measurements.len();

    // Build augmented track list: existing + potential births
    let mut augmented_tracks: Vec<(GlmbTrack<T, N>, bool, T)> = hypothesis
        .tracks
        .iter()
        .map(|t| (t.clone(), false, T::one()))
        .collect();

    let n_existing = augmented_tracks.len();

    // Add birth tracks
    for bt in birth_tracks {
        augmented_tracks.push((GlmbTrack::from_bernoulli(bt), true, bt.existence));
    }

    let n_total = augmented_tracks.len();

    if n_total == 0 && n_meas == 0 {
        // No tracks, no births, no measurements - return empty hypothesis
        return vec![GlmbHypothesis {
            log_weight: hypothesis.log_weight,
            tracks: Vec::new(),
            current_association: Vec::new(),
        }];
    }

    if n_total == 0 {
        // The common clutter set likelihood is factored out of every
        // hypothesis; association costs below use target/kappa ratios.
        return vec![GlmbHypothesis::empty(hypothesis.log_weight)];
    }

    // Precompute posteriors and likelihoods
    let mut posteriors: Vec<Vec<Option<(StateVector<T, N>, StateCovariance<T, N>, T)>>> =
        vec![vec![None; n_meas]; n_total];

    let mut detection_probs: Vec<T> = Vec::with_capacity(n_total);

    for (i, (track, _, _)) in augmented_tracks.iter().enumerate() {
        let p_d = observation_model.detection_probability(&track.state.mean);
        detection_probs.push(p_d);

        for (j, measurement) in measurements.iter().enumerate() {
            let predicted_meas = obs_matrix.observe(&track.state.mean);
            let innovation = measurement.innovation(predicted_meas);
            let innovation_cov =
                compute_innovation_covariance(&track.state.covariance, obs_matrix, meas_noise);

            let likelihood = crate::types::gaussian::innovation_likelihood(
                &innovation,
                innovation_cov.as_matrix(),
            );

            if likelihood <= T::zero() {
                stats.zero_likelihood_count += 1;
                continue;
            }

            if let Some(kalman_gain) =
                compute_kalman_gain(&track.state.covariance, obs_matrix, &innovation_cov)
            {
                let correction = kalman_gain.correct(&innovation);
                let updated_mean = StateVector::from_svector(
                    track.state.mean.as_svector() + correction.as_svector(),
                );
                let updated_cov = joseph_update(
                    &track.state.covariance,
                    &kalman_gain,
                    obs_matrix,
                    meas_noise,
                );

                posteriors[i][j] = Some((updated_mean, updated_cov, likelihood));
            } else {
                stats.singular_covariance_count += 1;
            }
        }
    }

    // Build cost matrix
    // Rows: tracks (existing + birth)
    // Columns: measurements + private absence/miss alternatives.
    let n_cols = n_meas + 2 * n_total;
    let mut cost_matrix = crate::assignment::CostMatrix::zeros(n_total, n_cols);
    let large_cost = 1e10_f64;

    for (i, (_track, is_birth, r_birth)) in augmented_tracks.iter().enumerate() {
        let p_d = detection_probs[i];

        // Detection columns
        for (j, measurement) in measurements.iter().enumerate() {
            let kappa = effective_clutter_intensity(clutter_model.clutter_intensity(measurement));

            let cost = if let Some((_, _, likelihood)) = posteriors[i][j] {
                if likelihood > T::zero() {
                    let ratio = if *is_birth {
                        p_d * likelihood * *r_birth / kappa
                    } else {
                        p_d * likelihood / kappa
                    };
                    if ratio > T::zero() {
                        -ComplexField::ln(ratio).to_subset().unwrap_or(large_cost)
                    } else {
                        large_cost
                    }
                } else {
                    large_cost
                }
            } else {
                large_cost
            };

            cost_matrix.set(i, j, cost);
        }

        // The first private column is absence for a birth and a miss for an
        // existing track. A birth also needs a distinct born-but-missed option.
        let primary_col = n_meas + i;
        let primary_probability = if *is_birth {
            T::one() - *r_birth
        } else {
            T::one() - p_d
        };
        cost_matrix.set(
            i,
            primary_col,
            -ComplexField::ln(primary_probability + T::from_f64(1e-300).unwrap())
                .to_subset()
                .unwrap_or(large_cost),
        );

        let birth_missed_col = n_meas + n_total + i;
        if *is_birth {
            let birth_missed_probability = *r_birth * (T::one() - p_d);
            cost_matrix.set(
                i,
                birth_missed_col,
                -ComplexField::ln(birth_missed_probability + T::from_f64(1e-300).unwrap())
                    .to_subset()
                    .unwrap_or(large_cost),
            );
        } else {
            cost_matrix.set(i, birth_missed_col, large_cost);
        }

        // Private columns can only be selected by their own row.
        for other in 0..n_total {
            if other != i {
                cost_matrix.set(i, n_meas + other, large_cost);
                cost_matrix.set(i, n_meas + n_total + other, large_cost);
            }
        }
    }

    // Generate k-best assignments using Murty's algorithm
    let assignments = crate::assignment::murty_k_best(&cost_matrix, k_best);

    // Create hypotheses from assignments
    let mut result: Vec<GlmbHypothesis<T, N>> = Vec::with_capacity(assignments.len());

    for assignment in assignments {
        let mut output_tracks: Vec<GlmbTrack<T, N>> = Vec::new();
        let mut associations: Vec<Option<usize>> = Vec::new();

        for (i, &col_opt) in assignment.mapping.iter().enumerate() {
            let col = match col_opt {
                Some(c) => c,
                None => continue,
            };

            let (track, is_birth, _) = &augmented_tracks[i];
            if col < n_meas {
                if let Some((mean, covariance, _)) = &posteriors[i][col] {
                    output_tracks.push(GlmbTrack {
                        label: track.label,
                        state: GaussianState::new(T::one(), *mean, *covariance),
                    });
                    associations.push(Some(col));
                }
            } else if !*is_birth || col >= n_meas + n_total {
                // Existing-track miss, or a birth that occurred but was missed.
                output_tracks.push(track.clone());
                associations.push(None);
            }
            // A birth assigned to its first private column did not occur.
        }

        // The assignment cost is -ln of the complete association
        // contribution, so subtract it exactly once.
        let new_log_weight = hypothesis.log_weight - T::from(assignment.cost).unwrap();

        result.push(GlmbHypothesis {
            log_weight: new_log_weight,
            tracks: output_tracks,
            current_association: associations,
        });
    }

    if result.is_empty() {
        // Fallback: return miss detection for all existing tracks
        let output_tracks = hypothesis.tracks.clone();
        let associations = vec![None; output_tracks.len()];
        let mut log_weight_delta = T::zero();

        for p_d in detection_probs.iter().take(n_existing) {
            log_weight_delta += ComplexField::ln(T::one() - *p_d + T::from_f64(1e-300).unwrap());
        }

        result.push(GlmbHypothesis {
            log_weight: hypothesis.log_weight + log_weight_delta,
            tracks: output_tracks,
            current_association: associations,
        });
    }

    result
}

// ============================================================================
// Hypothesis Generation (EKF)
// ============================================================================

/// Generates new hypotheses using EKF-style linearization for nonlinear sensors.
#[cfg(feature = "alloc")]
fn generate_hypotheses_ekf<
    T: RealField + Float + Copy,
    const N: usize,
    const M: usize,
    Obs: NonlinearObservationModel<T, N, M>,
    Clutter: ClutterModel<T, M>,
>(
    hypothesis: &GlmbHypothesis<T, N>,
    ctx: &HypothesisContext<'_, T, N, M>,
    observation_model: &Obs,
    clutter_model: &Clutter,
    stats: &mut UpdateStats,
) -> Vec<GlmbHypothesis<T, N>> {
    let measurements = ctx.measurements;
    let birth_tracks = ctx.birth_tracks;
    let meas_noise = ctx.meas_noise;
    let k_best = ctx.k_best;
    let n_meas = measurements.len();

    // Build augmented track list
    let mut augmented_tracks: Vec<(GlmbTrack<T, N>, bool, T)> = hypothesis
        .tracks
        .iter()
        .map(|t| (t.clone(), false, T::one()))
        .collect();

    let n_existing = augmented_tracks.len();

    for bt in birth_tracks {
        augmented_tracks.push((GlmbTrack::from_bernoulli(bt), true, bt.existence));
    }

    let n_total = augmented_tracks.len();

    if n_total == 0 && n_meas == 0 {
        return vec![GlmbHypothesis {
            log_weight: hypothesis.log_weight,
            tracks: Vec::new(),
            current_association: Vec::new(),
        }];
    }

    if n_total == 0 {
        // Keep the same factored common-clutter convention as nonempty rows.
        return vec![GlmbHypothesis::empty(hypothesis.log_weight)];
    }

    // Precompute posteriors using EKF linearization
    let mut posteriors: Vec<Vec<Option<(StateVector<T, N>, StateCovariance<T, N>, T)>>> =
        vec![vec![None; n_meas]; n_total];

    let mut detection_probs: Vec<T> = Vec::with_capacity(n_total);

    for (i, (track, _, _)) in augmented_tracks.iter().enumerate() {
        let p_d = observation_model.detection_probability(&track.state.mean);
        detection_probs.push(p_d);

        for (j, measurement) in measurements.iter().enumerate() {
            // EKF: Get Jacobian at current state
            let obs_matrix = match observation_model.jacobian_at(&track.state.mean) {
                Some(h) => h,
                None => {
                    stats.singular_covariance_count += 1;
                    continue;
                }
            };

            // EKF: Predicted measurement using nonlinear function
            let predicted_meas = observation_model.observe(&track.state.mean);
            let innovation = observation_model.innovation(measurement, &predicted_meas);

            let innovation_cov =
                compute_innovation_covariance(&track.state.covariance, &obs_matrix, meas_noise);

            let likelihood = crate::types::gaussian::innovation_likelihood(
                &innovation,
                innovation_cov.as_matrix(),
            );

            if likelihood <= T::zero() {
                stats.zero_likelihood_count += 1;
                continue;
            }

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
                    meas_noise,
                );

                posteriors[i][j] = Some((updated_mean, updated_cov, likelihood));
            } else {
                stats.singular_covariance_count += 1;
            }
        }
    }

    // Build cost matrix (same private alternatives as the linear case).
    let n_cols = n_meas + 2 * n_total;
    let mut cost_matrix = crate::assignment::CostMatrix::zeros(n_total, n_cols);
    let large_cost = 1e10_f64;

    for (i, (_track, is_birth, r_birth)) in augmented_tracks.iter().enumerate() {
        let p_d = detection_probs[i];

        for (j, measurement) in measurements.iter().enumerate() {
            let kappa = effective_clutter_intensity(clutter_model.clutter_intensity(measurement));

            let cost = if let Some((_, _, likelihood)) = posteriors[i][j] {
                if likelihood > T::zero() {
                    let ratio = if *is_birth {
                        p_d * likelihood * *r_birth / kappa
                    } else {
                        p_d * likelihood / kappa
                    };
                    if ratio > T::zero() {
                        -ComplexField::ln(ratio).to_subset().unwrap_or(large_cost)
                    } else {
                        large_cost
                    }
                } else {
                    large_cost
                }
            } else {
                large_cost
            };

            cost_matrix.set(i, j, cost);
        }

        // The first private column is absence for a birth and a miss for an
        // existing track. A birth also needs a distinct born-but-missed option.
        let primary_col = n_meas + i;
        let primary_probability = if *is_birth {
            T::one() - *r_birth
        } else {
            T::one() - p_d
        };
        cost_matrix.set(
            i,
            primary_col,
            -ComplexField::ln(primary_probability + T::from_f64(1e-300).unwrap())
                .to_subset()
                .unwrap_or(large_cost),
        );

        let birth_missed_col = n_meas + n_total + i;
        if *is_birth {
            let birth_missed_probability = *r_birth * (T::one() - p_d);
            cost_matrix.set(
                i,
                birth_missed_col,
                -ComplexField::ln(birth_missed_probability + T::from_f64(1e-300).unwrap())
                    .to_subset()
                    .unwrap_or(large_cost),
            );
        } else {
            cost_matrix.set(i, birth_missed_col, large_cost);
        }

        // Private columns can only be selected by their own row.
        for other in 0..n_total {
            if other != i {
                cost_matrix.set(i, n_meas + other, large_cost);
                cost_matrix.set(i, n_meas + n_total + other, large_cost);
            }
        }
    }

    let assignments = crate::assignment::murty_k_best(&cost_matrix, k_best);

    let mut result: Vec<GlmbHypothesis<T, N>> = Vec::with_capacity(assignments.len());

    for assignment in assignments {
        let mut output_tracks: Vec<GlmbTrack<T, N>> = Vec::new();
        let mut associations: Vec<Option<usize>> = Vec::new();

        for (i, &col_opt) in assignment.mapping.iter().enumerate() {
            let col = match col_opt {
                Some(c) => c,
                None => continue,
            };

            let (track, is_birth, _) = &augmented_tracks[i];
            if col < n_meas {
                if let Some((mean, covariance, _)) = &posteriors[i][col] {
                    output_tracks.push(GlmbTrack {
                        label: track.label,
                        state: GaussianState::new(T::one(), *mean, *covariance),
                    });
                    associations.push(Some(col));
                }
            } else if !*is_birth || col >= n_meas + n_total {
                // Existing-track miss, or a birth that occurred but was missed.
                output_tracks.push(track.clone());
                associations.push(None);
            }
            // A birth assigned to its first private column did not occur.
        }

        // The assignment cost is -ln of the complete association
        // contribution, so subtract it exactly once.
        let new_log_weight = hypothesis.log_weight - T::from(assignment.cost).unwrap();

        result.push(GlmbHypothesis {
            log_weight: new_log_weight,
            tracks: output_tracks,
            current_association: associations,
        });
    }

    if result.is_empty() {
        let output_tracks = hypothesis.tracks.clone();
        let associations = vec![None; output_tracks.len()];
        let mut log_weight_delta = T::zero();

        for p_d in detection_probs.iter().take(n_existing) {
            log_weight_delta += ComplexField::ln(T::one() - *p_d + T::from_f64(1e-300).unwrap());
        }

        result.push(GlmbHypothesis {
            log_weight: hypothesis.log_weight + log_weight_delta,
            tracks: output_tracks,
            current_association: associations,
        });
    }

    result
}

// ============================================================================
// State Extraction
// ============================================================================

/// Extracted state from the GLMB filter.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct GlmbEstimate<T: RealField, const N: usize> {
    /// Track label
    pub label: Label,
    /// State estimate
    pub state: StateVector<T, N>,
    /// State covariance
    pub covariance: StateCovariance<T, N>,
}

/// Extracts states from the best (highest weight) hypothesis.
#[cfg(feature = "alloc")]
pub fn extract_best_hypothesis<T: RealField + Float + Copy, const N: usize, Phase>(
    state: &GlmbFilterState<T, N, Phase>,
) -> Vec<GlmbEstimate<T, N>> {
    state
        .density
        .best_hypothesis()
        .map(|h| {
            h.tracks
                .iter()
                .map(|t| GlmbEstimate {
                    label: t.label,
                    state: t.state.mean,
                    covariance: t.state.covariance,
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Extracts marginal states with existence probability above threshold.
#[cfg(feature = "alloc")]
pub fn extract_marginal_states<T: RealField + Float + Copy, const N: usize, Phase>(
    state: &GlmbFilterState<T, N, Phase>,
    existence_threshold: T,
) -> Vec<(Label, StateVector<T, N>, T)> {
    state
        .density
        .marginal_states()
        .into_iter()
        .filter(|(_, _, r)| *r >= existence_threshold)
        .collect()
}

/// Extracts states using MAP cardinality.
///
/// Returns the states from the hypothesis closest to the MAP cardinality
/// among the highest-weight hypotheses.
#[cfg(feature = "alloc")]
pub fn extract_map_cardinality<T: RealField + Float + Copy, const N: usize, Phase>(
    state: &GlmbFilterState<T, N, Phase>,
) -> Vec<GlmbEstimate<T, N>> {
    let map_card = state.density.map_cardinality();

    // Find the best hypothesis with the MAP cardinality
    state
        .density
        .hypotheses
        .iter()
        .filter(|h| h.cardinality() == map_card)
        .max_by(|a, b| a.log_weight.partial_cmp(&b.log_weight).unwrap())
        .map(|h| {
            h.tracks
                .iter()
                .map(|t| GlmbEstimate {
                    label: t.label,
                    state: t.state.mean,
                    covariance: t.state.covariance,
                })
                .collect()
        })
        .unwrap_or_else(|| extract_best_hypothesis(state))
}

// ============================================================================
// Complete GLMB Filter
// ============================================================================

/// Complete GLMB filter with models.
///
/// This struct combines the filter state with all necessary models
/// for a complete tracking system.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct GlmbFilter<T, Trans, Obs, Clutter, Birth, const N: usize, const M: usize>
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
    /// Truncation configuration
    pub truncation: GlmbTruncationConfig<T>,
    /// Number of best assignments to generate per hypothesis
    pub k_best: usize,
    _marker: PhantomData<T>,
}

#[cfg(feature = "alloc")]
impl<T, Trans, Obs, Clutter, Birth, const N: usize, const M: usize>
    GlmbFilter<T, Trans, Obs, Clutter, Birth, N, M>
where
    T: RealField + Float + Copy,
    Trans: TransitionModel<T, N>,
    Obs: ObservationModel<T, N, M>,
    Clutter: ClutterModel<T, M>,
    Birth: LabeledBirthModel<T, N>,
{
    /// Creates a new GLMB filter with default truncation settings.
    pub fn new(
        transition: Trans,
        observation: Obs,
        clutter: Clutter,
        birth: Birth,
        k_best: usize,
    ) -> Self {
        Self {
            transition,
            observation,
            clutter,
            birth,
            truncation: GlmbTruncationConfig::default_config(),
            k_best,
            _marker: PhantomData,
        }
    }

    /// Creates a GLMB filter with custom truncation settings.
    pub fn with_truncation(
        transition: Trans,
        observation: Obs,
        clutter: Clutter,
        birth: Birth,
        truncation: GlmbTruncationConfig<T>,
        k_best: usize,
    ) -> Self {
        Self {
            transition,
            observation,
            clutter,
            birth,
            truncation,
            k_best,
            _marker: PhantomData,
        }
    }

    /// Creates initial filter state.
    pub fn initial_state(&self) -> GlmbFilterState<T, N, Updated> {
        GlmbFilterState::new()
    }

    /// Creates initial filter state from prior tracks.
    pub fn initial_state_from(
        &self,
        tracks: Vec<GlmbTrack<T, N>>,
    ) -> GlmbFilterState<T, N, Updated> {
        GlmbFilterState::from_tracks(tracks)
    }

    /// Runs one complete predict-update cycle.
    pub fn step(
        &self,
        state: GlmbFilterState<T, N, Updated>,
        measurements: &[Measurement<T, M>],
        dt: T,
    ) -> (GlmbFilterState<T, N, Updated>, UpdateStats) {
        let predicted = state.predict(&self.transition, &self.birth, dt);
        predicted.update(
            measurements,
            &self.observation,
            &self.clutter,
            &self.birth,
            &self.truncation,
            self.k_best,
        )
    }

    /// Runs one complete cycle using the fast joint predict-update algorithm.
    ///
    /// This is more efficient than `step()` when there are many hypotheses,
    /// as it shares computation across hypotheses with the same track set.
    ///
    /// See: "A fast implementation of the generalized labeled multi-Bernoulli
    /// filter with joint prediction and update" (Vo & Vo)
    pub fn step_joint(
        &self,
        state: GlmbFilterState<T, N, Updated>,
        measurements: &[Measurement<T, M>],
        dt: T,
    ) -> (GlmbFilterState<T, N, Updated>, UpdateStats) {
        let config = super::types::GlmbUpdateConfig::new(self.truncation.clone(), self.k_best, dt);
        state.joint_predict_and_update(
            measurements,
            &self.transition,
            &self.observation,
            &self.clutter,
            &self.birth,
            &config,
        )
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::labels::BernoulliTrack;
    use crate::types::spaces::StateCovariance;

    #[cfg(feature = "alloc")]
    #[derive(Clone, Copy)]
    struct CertainBirth;

    #[cfg(feature = "alloc")]
    impl LabeledBirthModel<f64, 4> for CertainBirth {
        fn birth_tracks(&self, label_gen: &mut LabelGenerator) -> Vec<BernoulliTrack<f64, 4>> {
            vec![BernoulliTrack::new(
                label_gen.next_label(),
                1.0,
                GaussianState::new(
                    1.0,
                    StateVector::from_array([20.0, 20.0, 0.0, 0.0]),
                    StateCovariance::identity(),
                ),
            )]
        }

        fn expected_birth_count(&self) -> f64 {
            1.0
        }
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_glmb_filter_state_creation() {
        let state: GlmbFilterState<f64, 4, Updated> = GlmbFilterState::new();
        assert_eq!(state.num_hypotheses(), 1);
        assert_eq!(state.time_step, 0);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_glmb_filter_state_from_tracks() {
        let label = Label::new(0, 0);
        let mean: StateVector<f64, 4> = StateVector::from_array([1.0, 2.0, 0.5, 0.5]);
        let cov: StateCovariance<f64, 4> = StateCovariance::identity();
        let state = GaussianState::new(1.0, mean, cov);

        let track = GlmbTrack::new(label, state);
        let filter_state = GlmbFilterState::from_tracks(vec![track]);

        assert_eq!(filter_state.num_hypotheses(), 1);
        assert_eq!(filter_state.density.hypotheses[0].cardinality(), 1);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_extract_best_hypothesis() {
        let label = Label::new(0, 0);
        let mean: StateVector<f64, 4> = StateVector::from_array([1.0, 2.0, 0.0, 0.0]);
        let cov: StateCovariance<f64, 4> = StateCovariance::identity();
        let track = GlmbTrack::new(label, GaussianState::new(1.0, mean, cov));

        let hypothesis = GlmbHypothesis::new(0.0, vec![track], vec![None]);
        let state = GlmbFilterState::from_hypotheses(vec![hypothesis]);

        let estimates = extract_best_hypothesis(&state);
        assert_eq!(estimates.len(), 1);
        assert_eq!(estimates[0].label, Label::new(0, 0));
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_prediction_branches_actual_survival_and_death_probabilities() {
        use crate::filters::lmb::NoBirthModel;
        use crate::models::ConstantVelocity2D;

        let transition = ConstantVelocity2D::new(0.0_f64, 0.25);
        let track = GlmbTrack::new(
            Label::new(0, 0),
            GaussianState::new(
                1.0,
                StateVector::from_array([10.0, 20.0, 0.0, 0.0]),
                StateCovariance::identity(),
            ),
        );
        let initial = GlmbFilterState::from_tracks(vec![track]);
        let predicted = initial.clone().predict(&transition, &NoBirthModel, 1.0);

        assert_eq!(predicted.density.hypotheses.len(), 2);
        let death = predicted
            .density
            .hypotheses
            .iter()
            .find(|hypothesis| hypothesis.tracks.is_empty())
            .unwrap();
        let survival = predicted
            .density
            .hypotheses
            .iter()
            .find(|hypothesis| hypothesis.tracks.len() == 1)
            .unwrap();
        assert!((Float::exp(death.log_weight - survival.log_weight) - 3.0).abs() < 1e-12);
        assert!((predicted.expected_target_count() - 0.25).abs() < 1e-12);
        assert_eq!(survival.tracks[0].state.weight, 1.0);

        let (joint, _) = initial.joint_predict_and_update(
            &[],
            &transition,
            &crate::models::PositionSensor2D::new(1.0, 0.0),
            &crate::models::UniformClutter2D::new(1.0, (0.0, 100.0), (0.0, 100.0)),
            &NoBirthModel,
            &crate::filters::glmb::types::GlmbUpdateConfig::new(
                GlmbTruncationConfig::new(-100.0, 10, None),
                10,
                1.0,
            ),
        );
        assert!((joint.expected_target_count() - 0.25).abs() < 1e-12);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_assignment_contribution_is_applied_once() {
        use crate::filters::lmb::NoBirthModel;
        use crate::models::{ConstantVelocity2D, PositionSensor2D, UniformClutter2D};

        let transition = ConstantVelocity2D::new(0.0_f64, 1.0);
        let observation = PositionSensor2D::new(1.0, 0.5);
        let clutter = UniformClutter2D::new(1.0, (0.0, 100.0), (0.0, 100.0));
        let track = GlmbTrack::new(
            Label::new(0, 0),
            GaussianState::new(
                1.0,
                StateVector::from_array([50.0, 50.0, 0.0, 0.0]),
                StateCovariance::identity(),
            ),
        );
        let predicted =
            GlmbFilterState::from_tracks(vec![track]).predict(&transition, &NoBirthModel, 0.0);
        let (updated, _) = predicted.update(
            &[Measurement::from_array([50.0, 50.0])],
            &observation,
            &clutter,
            &NoBirthModel,
            &GlmbTruncationConfig::new(-100.0, 10, None),
            10,
        );

        let detected = updated
            .density
            .hypotheses
            .iter()
            .find(|hypothesis| hypothesis.current_association == [Some(0)])
            .unwrap();
        let missed = updated
            .density
            .hypotheses
            .iter()
            .find(|hypothesis| hypothesis.current_association == [None])
            .unwrap();
        let actual_ratio = Float::exp(detected.log_weight - missed.log_weight);
        let expected_ratio = (1.0 / (4.0 * core::f64::consts::PI)) / 0.0001;
        assert!((actual_ratio / expected_ratio - 1.0).abs() < 1e-10);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_ekf_update_wraps_range_bearing_innovation() {
        use crate::filters::lmb::NoBirthModel;
        use crate::models::{ConstantVelocity2D, RangeBearingSensor, UniformClutterRangeBearing};

        let transition = ConstantVelocity2D::new(0.0_f64, 1.0);
        let track = GlmbTrack::new(
            Label::new(0, 0),
            GaussianState::new(
                1.0,
                StateVector::from_array([-100.0, 0.1, 0.0, 0.0]),
                StateCovariance::identity(),
            ),
        );
        let predicted =
            GlmbFilterState::from_tracks(vec![track]).predict(&transition, &NoBirthModel, 0.0);
        let (updated, _) = predicted.update_ekf(
            &[Measurement::from_array([
                100.0,
                -core::f64::consts::PI + 0.001,
            ])],
            &RangeBearingSensor::new(1.0, 0.1, 0.9),
            &UniformClutterRangeBearing::new(
                1.0,
                (0.0, 50.0),
                (-core::f64::consts::PI, core::f64::consts::PI),
            ),
            &NoBirthModel,
            &GlmbTruncationConfig::new(-100.0, 10, None),
            10,
        );

        assert!(updated.density.hypotheses.iter().any(|hypothesis| {
            hypothesis.current_association == [Some(0)] && hypothesis.tracks.len() == 1
        }));
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_certain_birth_is_retained_when_born_but_missed() {
        use crate::models::{PositionSensor2D, RangeBearingSensor, UniformClutter2D};

        let transition = crate::models::ConstantVelocity2D::new(0.0_f64, 1.0);
        let predicted = GlmbFilterState::new().predict(&transition, &CertainBirth, 0.0);
        let (linear, _) = predicted.clone().update(
            &[],
            &PositionSensor2D::new(1.0, 0.0),
            &UniformClutter2D::new(1.0, (0.0, 100.0), (0.0, 100.0)),
            &CertainBirth,
            &GlmbTruncationConfig::new(-100.0, 10, None),
            10,
        );
        assert_eq!(extract_best_hypothesis(&linear).len(), 1);

        let (ekf, _) = predicted.update_ekf(
            &[],
            &RangeBearingSensor::new(1.0, 0.1, 0.0),
            &crate::models::UniformClutterRangeBearing::new(
                1.0,
                (0.0, 100.0),
                (-core::f64::consts::PI, core::f64::consts::PI),
            ),
            &CertainBirth,
            &GlmbTruncationConfig::new(-100.0, 10, None),
            10,
        );
        assert_eq!(extract_best_hypothesis(&ekf).len(), 1);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_zero_track_and_nonempty_hypotheses_share_factored_clutter_convention() {
        use crate::filters::lmb::NoBirthModel;
        use crate::models::{ConstantVelocity2D, PositionSensor2D, UniformClutter2D};

        let track = GlmbTrack::new(
            Label::new(0, 0),
            GaussianState::new(
                1.0,
                StateVector::from_array([20.0, 20.0, 0.0, 0.0]),
                StateCovariance::identity(),
            ),
        );
        let initial = GlmbFilterState::from_hypotheses(vec![
            GlmbHypothesis::empty(0.0),
            GlmbHypothesis::new(0.0, vec![track], vec![None]),
        ]);
        let predicted = initial.predict(&ConstantVelocity2D::new(0.0, 1.0), &NoBirthModel, 0.0);
        let (updated, _) = predicted.update(
            &[Measurement::from_array([20.0, 20.0])],
            &PositionSensor2D::new(1.0, 0.0),
            &UniformClutter2D::new(1.0, (0.0, 100.0), (0.0, 100.0)),
            &NoBirthModel,
            &GlmbTruncationConfig::new(-100.0, 10, None),
            10,
        );

        let empty = updated
            .density
            .hypotheses
            .iter()
            .find(|hypothesis| hypothesis.tracks.is_empty())
            .unwrap();
        let nonempty = updated
            .density
            .hypotheses
            .iter()
            .find(|hypothesis| hypothesis.tracks.len() == 1)
            .unwrap();
        assert!(Float::abs(empty.log_weight - nonempty.log_weight) < 1e-12);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_out_of_support_clutter_still_allows_target_association() {
        use crate::filters::lmb::NoBirthModel;
        use crate::models::{ConstantVelocity2D, PositionSensor2D, UniformClutter2D};

        let transition = ConstantVelocity2D::new(0.0_f64, 1.0);
        let track = GlmbTrack::new(
            Label::new(0, 0),
            GaussianState::new(
                1.0,
                StateVector::from_array([20.0, 20.0, 0.0, 0.0]),
                StateCovariance::identity(),
            ),
        );
        let initial = GlmbFilterState::from_tracks(vec![track]);
        let measurement = Measurement::from_array([20.0, 20.0]);
        let observation = PositionSensor2D::new(1.0, 0.9);
        let clutter = UniformClutter2D::new(1.0, (0.0, 10.0), (0.0, 10.0));
        let truncation = GlmbTruncationConfig::new(-100.0, 10, None);

        let (standalone, _) = initial
            .clone()
            .predict(&transition, &NoBirthModel, 0.0)
            .update(
                &[measurement],
                &observation,
                &clutter,
                &NoBirthModel,
                &truncation,
                10,
            );
        let standalone_best = standalone
            .density
            .hypotheses
            .iter()
            .max_by(|left, right| left.log_weight.partial_cmp(&right.log_weight).unwrap())
            .unwrap();
        assert_eq!(standalone_best.current_association, [Some(0)]);

        let (joint, _) = initial.joint_predict_and_update(
            &[measurement],
            &transition,
            &observation,
            &clutter,
            &NoBirthModel,
            &crate::filters::glmb::types::GlmbUpdateConfig::new(truncation, 10, 0.0),
        );
        let joint_best = joint
            .density
            .hypotheses
            .iter()
            .max_by(|left, right| left.log_weight.partial_cmp(&right.log_weight).unwrap())
            .unwrap();
        assert_eq!(joint_best.current_association, [Some(0)]);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_joint_predict_and_update() {
        use crate::filters::lmb::NoBirthModel;
        use crate::models::{ConstantVelocity2D, PositionSensor2D, UniformClutter2D};

        // Create models
        let transition = ConstantVelocity2D::new(1.0, 0.99);
        let observation = PositionSensor2D::new(10.0, 0.95);
        let clutter = UniformClutter2D::new(1.0, (0.0, 100.0), (0.0, 100.0));
        let birth = NoBirthModel;
        let truncation = GlmbTruncationConfig::default_config();

        // Create initial state with one track
        let label = Label::new(0, 0);
        let mean: StateVector<f64, 4> = StateVector::from_array([10.0, 20.0, 1.0, 0.5]);
        let cov: StateCovariance<f64, 4> = StateCovariance::identity();
        let track = GlmbTrack::new(label, GaussianState::new(1.0, mean, cov));
        let state = GlmbFilterState::from_tracks(vec![track]);

        // Create measurement near expected position after dt=1.0
        // Expected position: [10+1, 20+0.5] = [11, 20.5]
        let measurement = Measurement::from_array([11.5, 20.8]);

        // Run joint predict and update
        let config = crate::filters::glmb::types::GlmbUpdateConfig::new(truncation, 10, 1.0);
        let (new_state, stats) = state.joint_predict_and_update(
            &[measurement],
            &transition,
            &observation,
            &clutter,
            &birth,
            &config,
        );

        // Should still have hypotheses
        assert!(new_state.num_hypotheses() > 0);
        // Time step should advance
        assert_eq!(new_state.time_step, 1);

        // Extract best hypothesis
        let estimates = extract_best_hypothesis(&new_state);

        // Should have detected the track
        assert!(!estimates.is_empty(), "Track should be detected");

        if !estimates.is_empty() {
            // State should be updated towards measurement
            let est = &estimates[0];
            // Position should be between predicted and measurement
            let x = est.state.as_slice()[0];
            let y = est.state.as_slice()[1];
            assert!(x > 10.0 && x < 12.0);
            assert!(y > 20.0 && y < 21.0);
        }

        // Stats object should be returned (no specific field to check for hypotheses count)
        let _ = stats;
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_step_vs_step_joint_equivalence() {
        use crate::filters::lmb::NoBirthModel;
        use crate::models::{ConstantVelocity2D, PositionSensor2D, UniformClutter2D};

        // Create filter
        let transition = ConstantVelocity2D::new(1.0, 0.99);
        let observation = PositionSensor2D::new(10.0, 0.95);
        let clutter = UniformClutter2D::new(1.0, (0.0, 100.0), (0.0, 100.0));
        let birth = NoBirthModel;

        let filter = GlmbFilter::new(transition, observation, clutter, birth, 5);

        // Create identical initial states
        let label = Label::new(0, 0);
        let mean: StateVector<f64, 4> = StateVector::from_array([10.0, 20.0, 1.0, 0.5]);
        let cov: StateCovariance<f64, 4> = StateCovariance::identity();

        let track1 = GlmbTrack::new(label, GaussianState::new(1.0, mean, cov));
        let state1 = GlmbFilterState::from_tracks(vec![track1]);

        let track2 = GlmbTrack::new(label, GaussianState::new(1.0, mean, cov));
        let state2 = GlmbFilterState::from_tracks(vec![track2]);

        // Same measurement
        let measurement = Measurement::from_array([11.5, 20.8]);

        // Run both methods
        let (result1, _) = filter.step(state1, &[measurement], 1.0);
        let (result2, _) = filter.step_joint(state2, &[measurement], 1.0);

        // Both should have similar number of hypotheses
        // (May not be exactly equal due to different processing order)
        assert!(result1.num_hypotheses() > 0);
        assert!(result2.num_hypotheses() > 0);

        // Extract estimates - should be similar
        let est1 = extract_best_hypothesis(&result1);
        let est2 = extract_best_hypothesis(&result2);

        assert_eq!(est1.len(), est2.len());

        if !est1.is_empty() {
            // States should be very close (both processed same measurement)
            let x1 = est1[0].state.as_slice()[0];
            let x2 = est2[0].state.as_slice()[0];
            let y1 = est1[0].state.as_slice()[1];
            let y2 = est2[0].state.as_slice()[1];
            let diff_x = (x1 - x2).abs();
            let diff_y = (y1 - y2).abs();
            assert!(diff_x < 0.1, "X positions differ: {} vs {}", x1, x2);
            assert!(diff_y < 0.1, "Y positions differ: {} vs {}", y1, y2);
        }
    }
}
