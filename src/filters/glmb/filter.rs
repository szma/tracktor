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

use core::marker::PhantomData;
use nalgebra::{ComplexField, RealField};
use num_traits::Float;

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use super::types::{GlmbDensity, GlmbHypothesis, GlmbTrack, GlmbTruncationConfig};
use crate::filters::lmb::LabeledBirthModel;
use crate::filters::phd::{Predicted, UpdateStats, Updated};
use crate::models::{ClutterModel, NonlinearObservationModel, ObservationModel, TransitionModel};
use crate::types::gaussian::GaussianState;
use crate::types::labels::{Label, LabelGenerator};
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

    /// Creates an Updated state from components (for internal use).
    pub(crate) fn from_components(
        density: GlmbDensity<T, N>,
        label_gen: LabelGenerator,
        time_step: u32,
    ) -> Self {
        Self {
            density,
            label_gen,
            time_step,
            _phase: PhantomData,
        }
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
    /// - `truncation`: Truncation configuration
    /// - `k_best`: Number of best assignments to generate per hypothesis group
    /// - `dt`: Time step
    pub fn joint_predict_and_update<const M: usize, Trans, Obs, Clutter, Birth>(
        mut self,
        measurements: &[Measurement<T, M>],
        transition_model: &Trans,
        observation_model: &Obs,
        clutter_model: &Clutter,
        birth_model: &Birth,
        truncation: &GlmbTruncationConfig<T>,
        k_best: usize,
        dt: T,
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
        let transition_matrix = transition_model.transition_matrix(dt);
        let process_noise = transition_model.process_noise(dt);

        let mut predicted_tracks: BTreeMap<Label, (GlmbTrack<T, N>, T)> = BTreeMap::new();

        for (label, track) in unique_tracks {
            let p_s = transition_model.survival_probability(&track.state.mean);

            if p_s < T::from_f64(1e-10).unwrap() {
                // Track dies with high probability - skip
                continue;
            }

            let predicted_mean = transition_matrix.apply_state(&track.state.mean);
            let predicted_cov = transition_matrix
                .propagate_covariance(&track.state.covariance)
                .add(&process_noise);

            let predicted_track = GlmbTrack {
                label,
                state: GaussianState::new(track.state.weight * p_s, predicted_mean, predicted_cov),
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
            // Get the labels of tracks that survived prediction
            let mut track_labels: Vec<Label> = hypothesis
                .tracks
                .iter()
                .filter_map(|t| {
                    if predicted_tracks.contains_key(&t.label) {
                        Some(t.label)
                    } else {
                        None
                    }
                })
                .collect();
            track_labels.sort();

            // Compute log weight adjustment for dead tracks
            let mut death_log_weight = T::zero();
            for track in &hypothesis.tracks {
                if !predicted_tracks.contains_key(&track.label) {
                    // Track died
                    let p_s = T::from_f64(1e-10).unwrap();
                    death_log_weight = death_log_weight
                        + ComplexField::ln(T::one() - p_s + T::from_f64(1e-300).unwrap());
                }
            }

            let adjusted_log_weight = hypothesis.log_weight + death_log_weight;

            hypothesis_groups
                .entry(track_labels)
                .or_insert_with(Vec::new)
                .push((hyp_idx, adjusted_log_weight));
        }

        // Step 7: Process each hypothesis group
        let mut new_hypotheses: Vec<GlmbHypothesis<T, N>> = Vec::new();

        for (track_labels, group) in hypothesis_groups {
            // Build cost matrix for this track set + birth tracks
            // Rows: existing tracks from this group + all birth tracks
            // Columns: measurements + miss columns

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
                // No tracks, only clutter
                let clutter_log_weight: T = measurements
                    .iter()
                    .map(|z| ComplexField::ln(clutter_model.clutter_intensity(z)))
                    .fold(T::zero(), |acc, x| acc + x);

                for (_, log_weight) in &group {
                    new_hypotheses.push(GlmbHypothesis {
                        log_weight: *log_weight + clutter_log_weight,
                        tracks: Vec::new(),
                        current_association: Vec::new(),
                    });
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
            let n_cols = n_meas + n_group_tracks;
            let mut cost_matrix = crate::assignment::CostMatrix::zeros(n_group_tracks, n_cols);
            let large_cost = 1e10_f64;

            for (row, &global_idx) in group_track_indices.iter().enumerate() {
                let (_, is_birth, r_birth) = &all_tracks[global_idx];
                let p_d = detection_probs[global_idx];

                // Detection columns
                for (j, measurement) in measurements.iter().enumerate() {
                    let kappa = clutter_model.clutter_intensity(measurement);

                    let cost = if let Some((_, _, likelihood)) = posteriors[global_idx][j] {
                        if likelihood > T::zero() && kappa > T::zero() {
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

                    cost_matrix.set(row, j, cost);
                }

                // Miss column
                let miss_col = n_meas + row;
                let miss_cost = if *is_birth {
                    // Birth track not born - cost of not detecting birth
                    -ComplexField::ln(T::one() - *r_birth + T::from_f64(1e-300).unwrap())
                        .to_subset()
                        .unwrap_or(large_cost)
                } else {
                    // Existing track missed detection
                    -ComplexField::ln(T::one() - p_d + T::from_f64(1e-300).unwrap())
                        .to_subset()
                        .unwrap_or(large_cost)
                };
                cost_matrix.set(row, miss_col, miss_cost);

                // Set other miss columns to infinity (each track has its own miss column)
                for other_row in 0..n_group_tracks {
                    if other_row != row {
                        cost_matrix.set(row, n_meas + other_row, large_cost);
                    }
                }
            }

            // Run Murty's k-best
            let assignments = crate::assignment::hungarian::murty_k_best(&cost_matrix, k_best);

            // Generate hypotheses from assignments for each input hypothesis in this group
            for (_hyp_idx, base_log_weight) in &group {
                for assignment in &assignments {
                    let mut output_tracks: Vec<GlmbTrack<T, N>> = Vec::new();
                    let mut associations: Vec<Option<usize>> = Vec::new();
                    let mut log_weight_delta = T::zero();

                    for (row, col_opt) in assignment.mapping.iter().enumerate() {
                        let col = match col_opt {
                            Some(c) => *c,
                            None => continue, // Unassigned row, skip
                        };
                        let global_idx = group_track_indices[row];
                        let (track, is_birth, r_birth) = &all_tracks[global_idx];
                        let p_d = detection_probs[global_idx];

                        if col < n_meas {
                            // Detection
                            if let Some((updated_mean, updated_cov, likelihood)) =
                                &posteriors[global_idx][col]
                            {
                                let kappa = clutter_model.clutter_intensity(&measurements[col]);

                                let weight_contrib = if *is_birth {
                                    p_d * *likelihood * *r_birth / kappa
                                } else {
                                    p_d * *likelihood / kappa
                                };

                                log_weight_delta = log_weight_delta
                                    + ComplexField::ln(
                                        weight_contrib + T::from_f64(1e-300).unwrap(),
                                    );

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
                        } else {
                            // Miss detection
                            if *is_birth {
                                // Birth track not instantiated - skip
                                log_weight_delta = log_weight_delta
                                    + ComplexField::ln(
                                        T::one() - *r_birth + T::from_f64(1e-300).unwrap(),
                                    );
                            } else {
                                // Existing track missed
                                log_weight_delta = log_weight_delta
                                    + ComplexField::ln(
                                        T::one() - p_d + T::from_f64(1e-300).unwrap(),
                                    );

                                output_tracks.push(track.clone());
                                associations.push(None);
                            }
                        }
                    }

                    let new_log_weight =
                        *base_log_weight + log_weight_delta - T::from(assignment.cost).unwrap();

                    new_hypotheses.push(GlmbHypothesis {
                        log_weight: new_log_weight,
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

        // Predict each hypothesis
        for hypothesis in &mut self.density.hypotheses {
            // Predict existing tracks with survival
            let mut tracks_to_remove: Vec<usize> = Vec::new();

            for (idx, track) in hypothesis.tracks.iter_mut().enumerate() {
                let p_s = transition_model.survival_probability(&track.state.mean);

                // Check if track should be removed due to very low survival
                if p_s < T::from_f64(1e-10).unwrap() {
                    tracks_to_remove.push(idx);
                    continue;
                }

                // Predict state
                track.state.mean = transition_matrix.apply_state(&track.state.mean);
                track.state.covariance = transition_matrix
                    .propagate_covariance(&track.state.covariance)
                    .add(&process_noise);

                // Weight contribution from survival
                track.state.weight = track.state.weight * p_s;
            }

            // Remove dead tracks (iterate in reverse to preserve indices)
            for &idx in tracks_to_remove.iter().rev() {
                hypothesis.tracks.remove(idx);
                hypothesis.current_association.remove(idx);
            }

            // Update log weight for track deaths
            for _ in &tracks_to_remove {
                let p_s = T::from_f64(1e-10).unwrap(); // Survival prob was below this
                hypothesis.log_weight = hypothesis.log_weight
                    + ComplexField::ln(T::one() - p_s + T::from_f64(1e-300).unwrap());
            }

            // Clear old associations (they belong to previous timestep)
            for assoc in &mut hypothesis.current_association {
                *assoc = None;
            }
        }

        GlmbFilterState {
            density: self.density,
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
    /// Creates a Predicted state from components (for internal use).
    pub(crate) fn from_components(
        density: GlmbDensity<T, N>,
        label_gen: LabelGenerator,
        time_step: u32,
    ) -> Self {
        Self {
            density,
            label_gen,
            time_step,
            _phase: PhantomData,
        }
    }

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

        // Process each input hypothesis
        for hypothesis in &self.density.hypotheses {
            let generated = generate_hypotheses_from_assignment(
                hypothesis,
                measurements,
                &birth_tracks,
                observation_model,
                clutter_model,
                &obs_matrix,
                &meas_noise,
                k_best,
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

        // Process each input hypothesis
        for hypothesis in &self.density.hypotheses {
            let generated = generate_hypotheses_ekf(
                hypothesis,
                measurements,
                &birth_tracks,
                observation_model,
                clutter_model,
                &meas_noise,
                k_best,
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
    measurements: &[Measurement<T, M>],
    birth_tracks: &[crate::types::labels::BernoulliTrack<T, N>],
    observation_model: &Obs,
    clutter_model: &Clutter,
    obs_matrix: &crate::types::transforms::ObservationMatrix<T, M, N>,
    meas_noise: &crate::types::spaces::MeasurementCovariance<T, M>,
    k_best: usize,
    stats: &mut UpdateStats,
) -> Vec<GlmbHypothesis<T, N>> {
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
        // No tracks but have measurements - they're all clutter
        let clutter_log_weight: T = measurements
            .iter()
            .map(|z| ComplexField::ln(clutter_model.clutter_intensity(z)))
            .fold(T::zero(), |acc, x| acc + x);

        return vec![GlmbHypothesis {
            log_weight: hypothesis.log_weight + clutter_log_weight,
            tracks: Vec::new(),
            current_association: Vec::new(),
        }];
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
    // Columns: measurements + miss columns (one per track)
    let n_cols = n_meas + n_total;
    let mut cost_matrix = crate::assignment::CostMatrix::zeros(n_total, n_cols);
    let large_cost = 1e10_f64;

    for (i, (_track, is_birth, r_birth)) in augmented_tracks.iter().enumerate() {
        let p_d = detection_probs[i];

        // Detection columns
        for (j, measurement) in measurements.iter().enumerate() {
            let kappa = clutter_model.clutter_intensity(measurement);

            let cost = if let Some((_, _, likelihood)) = posteriors[i][j] {
                if likelihood > T::zero() && kappa > T::zero() {
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

        // Miss column for this track
        let miss_col = n_meas + i;
        let miss_cost = if *is_birth {
            // Birth track not born
            let one_minus_r = T::one() - *r_birth;
            if one_minus_r > T::from_f64(1e-300).unwrap() {
                -ComplexField::ln(one_minus_r).to_subset().unwrap_or(0.0)
            } else {
                large_cost // r_birth ≈ 1 means must be born
            }
        } else {
            // Existing track missed detection
            let one_minus_pd = T::one() - p_d;
            if one_minus_pd > T::from_f64(1e-300).unwrap() {
                -ComplexField::ln(one_minus_pd)
                    .to_subset()
                    .unwrap_or(large_cost)
            } else {
                large_cost // p_d ≈ 1 means must be detected
            }
        };
        cost_matrix.set(i, miss_col, miss_cost);

        // Set infinite cost for other miss columns
        for k in 0..n_total {
            if k != i {
                cost_matrix.set(i, n_meas + k, 1e20);
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
        let mut log_weight_delta = T::zero();

        for (i, &col_opt) in assignment.mapping.iter().enumerate() {
            let col = match col_opt {
                Some(c) => c,
                None => continue,
            };

            let (track, is_birth, r_birth) = &augmented_tracks[i];
            let p_d = detection_probs[i];

            if col < n_meas {
                // Track detected with measurement col
                if let Some((mean, cov, likelihood)) = &posteriors[i][col] {
                    output_tracks.push(GlmbTrack {
                        label: track.label,
                        state: GaussianState::new(T::one(), *mean, *cov),
                    });
                    associations.push(Some(col));

                    // Weight contribution
                    let kappa = clutter_model.clutter_intensity(&measurements[col]);
                    if *is_birth {
                        log_weight_delta = log_weight_delta
                            + ComplexField::ln(p_d * *likelihood * *r_birth / kappa);
                    } else {
                        log_weight_delta =
                            log_weight_delta + ComplexField::ln(p_d * *likelihood / kappa);
                    }
                }
            } else {
                // Miss column
                if *is_birth {
                    // Birth track NOT born - don't add to output
                    log_weight_delta = log_weight_delta
                        + ComplexField::ln(T::one() - *r_birth + T::from_f64(1e-300).unwrap());
                } else {
                    // Existing track missed
                    output_tracks.push(track.clone());
                    associations.push(None);

                    log_weight_delta = log_weight_delta
                        + ComplexField::ln(T::one() - p_d + T::from_f64(1e-300).unwrap());
                }
            }
        }

        let new_log_weight =
            hypothesis.log_weight + log_weight_delta - T::from(assignment.cost).unwrap();

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
            log_weight_delta =
                log_weight_delta + ComplexField::ln(T::one() - *p_d + T::from_f64(1e-300).unwrap());
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
    measurements: &[Measurement<T, M>],
    birth_tracks: &[crate::types::labels::BernoulliTrack<T, N>],
    observation_model: &Obs,
    clutter_model: &Clutter,
    meas_noise: &crate::types::spaces::MeasurementCovariance<T, M>,
    k_best: usize,
    stats: &mut UpdateStats,
) -> Vec<GlmbHypothesis<T, N>> {
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
        let clutter_log_weight: T = measurements
            .iter()
            .map(|z| ComplexField::ln(clutter_model.clutter_intensity(z)))
            .fold(T::zero(), |acc, x| acc + x);

        return vec![GlmbHypothesis {
            log_weight: hypothesis.log_weight + clutter_log_weight,
            tracks: Vec::new(),
            current_association: Vec::new(),
        }];
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
            let innovation = measurement.innovation(predicted_meas);

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

    // Build cost matrix (same structure as linear case)
    let n_cols = n_meas + n_total;
    let mut cost_matrix = crate::assignment::CostMatrix::zeros(n_total, n_cols);
    let large_cost = 1e10_f64;

    for (i, (_track, is_birth, r_birth)) in augmented_tracks.iter().enumerate() {
        let p_d = detection_probs[i];

        for (j, measurement) in measurements.iter().enumerate() {
            let kappa = clutter_model.clutter_intensity(measurement);

            let cost = if let Some((_, _, likelihood)) = posteriors[i][j] {
                if likelihood > T::zero() && kappa > T::zero() {
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

        let miss_col = n_meas + i;
        let miss_cost = if *is_birth {
            let one_minus_r = T::one() - *r_birth;
            if one_minus_r > T::from_f64(1e-300).unwrap() {
                -ComplexField::ln(one_minus_r).to_subset().unwrap_or(0.0)
            } else {
                large_cost
            }
        } else {
            let one_minus_pd = T::one() - p_d;
            if one_minus_pd > T::from_f64(1e-300).unwrap() {
                -ComplexField::ln(one_minus_pd)
                    .to_subset()
                    .unwrap_or(large_cost)
            } else {
                large_cost
            }
        };
        cost_matrix.set(i, miss_col, miss_cost);

        for k in 0..n_total {
            if k != i {
                cost_matrix.set(i, n_meas + k, 1e20);
            }
        }
    }

    let assignments = crate::assignment::murty_k_best(&cost_matrix, k_best);

    let mut result: Vec<GlmbHypothesis<T, N>> = Vec::with_capacity(assignments.len());

    for assignment in assignments {
        let mut output_tracks: Vec<GlmbTrack<T, N>> = Vec::new();
        let mut associations: Vec<Option<usize>> = Vec::new();
        let mut log_weight_delta = T::zero();

        for (i, &col_opt) in assignment.mapping.iter().enumerate() {
            let col = match col_opt {
                Some(c) => c,
                None => continue,
            };

            let (track, is_birth, r_birth) = &augmented_tracks[i];
            let p_d = detection_probs[i];

            if col < n_meas {
                if let Some((mean, cov, likelihood)) = &posteriors[i][col] {
                    output_tracks.push(GlmbTrack {
                        label: track.label,
                        state: GaussianState::new(T::one(), *mean, *cov),
                    });
                    associations.push(Some(col));

                    let kappa = clutter_model.clutter_intensity(&measurements[col]);
                    if *is_birth {
                        log_weight_delta = log_weight_delta
                            + ComplexField::ln(p_d * *likelihood * *r_birth / kappa);
                    } else {
                        log_weight_delta =
                            log_weight_delta + ComplexField::ln(p_d * *likelihood / kappa);
                    }
                }
            } else {
                if *is_birth {
                    log_weight_delta = log_weight_delta
                        + ComplexField::ln(T::one() - *r_birth + T::from_f64(1e-300).unwrap());
                } else {
                    output_tracks.push(track.clone());
                    associations.push(None);

                    log_weight_delta = log_weight_delta
                        + ComplexField::ln(T::one() - p_d + T::from_f64(1e-300).unwrap());
                }
            }
        }

        let new_log_weight =
            hypothesis.log_weight + log_weight_delta - T::from(assignment.cost).unwrap();

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
            log_weight_delta =
                log_weight_delta + ComplexField::ln(T::one() - *p_d + T::from_f64(1e-300).unwrap());
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
        state.joint_predict_and_update(
            measurements,
            &self.transition,
            &self.observation,
            &self.clutter,
            &self.birth,
            &self.truncation,
            self.k_best,
            dt,
        )
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::spaces::StateCovariance;

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
        let (new_state, stats) = state.joint_predict_and_update(
            &[measurement],
            &transition,
            &observation,
            &clutter,
            &birth,
            &truncation,
            10,
            1.0,
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
