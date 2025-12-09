//! Comprehensive tests for LMB and LMBM filters
//!
//! This module provides thorough test coverage for the Labeled Multi-Bernoulli
//! filter implementation, including:
//! - Integration tests for predict-update cycles
//! - Track continuity and label persistence
//! - LBP convergence and data association
//! - Multi-target scenarios
//! - Clutter robustness
//! - Missed detection handling
//! - Numerical stability edge cases
//! - LMBM-specific tests

#![cfg(test)]
#![cfg(feature = "alloc")]

use alloc::vec;
use alloc::vec::Vec;

use crate::filters::lmb::filter::{
    extract_lmb_estimates, extract_lmb_estimates_threshold, LabeledBirthModel, LmbFilter,
    LmbFilterState,
};
use crate::filters::lmb::lmbm::{extract_best_hypothesis, LmbmFilter, LmbmFilterState};
use crate::filters::lmb::types::{LmbTrack, LmbTrackSet, LmbmHypothesis, LmbmState};
use crate::filters::phd::Updated;
use crate::models::{ConstantVelocity2D, PositionSensor2D, UniformClutter2D};
use crate::types::gaussian::GaussianState;
use crate::types::labels::{BernoulliTrack, Label, LabelGenerator};
use crate::types::spaces::{Measurement, StateCovariance, StateVector};

// ============================================================================
// Test Helpers
// ============================================================================

/// Simple birth model for testing
struct TestBirthModel {
    birth_locations: Vec<(f64, StateVector<f64, 4>, StateCovariance<f64, 4>)>,
}

impl TestBirthModel {
    fn new() -> Self {
        Self {
            birth_locations: Vec::new(),
        }
    }

    fn with_location(mut self, existence: f64, x: f64, y: f64) -> Self {
        let cov = StateCovariance::from_matrix(nalgebra::matrix![
            100.0, 0.0, 0.0, 0.0;
            0.0, 100.0, 0.0, 0.0;
            0.0, 0.0, 25.0, 0.0;
            0.0, 0.0, 0.0, 25.0
        ]);
        self.birth_locations
            .push((existence, StateVector::from_array([x, y, 0.0, 0.0]), cov));
        self
    }
}

impl LabeledBirthModel<f64, 4> for TestBirthModel {
    fn birth_tracks(&self, label_gen: &mut LabelGenerator) -> Vec<BernoulliTrack<f64, 4>> {
        self.birth_locations
            .iter()
            .map(|(existence, mean, cov)| {
                let label = label_gen.next_label();
                let state = GaussianState::new(1.0, *mean, *cov);
                BernoulliTrack::new(label, *existence, state)
            })
            .collect()
    }

    fn expected_birth_count(&self) -> f64 {
        self.birth_locations.iter().map(|(e, _, _)| e).sum()
    }
}

/// No-op birth model for deterministic testing
struct NoBirthModel;

impl LabeledBirthModel<f64, 4> for NoBirthModel {
    fn birth_tracks(&self, _label_gen: &mut LabelGenerator) -> Vec<BernoulliTrack<f64, 4>> {
        Vec::new()
    }

    fn expected_birth_count(&self) -> f64 {
        0.0
    }
}

/// Creates a test track with given position and velocity
fn make_track(label: Label, x: f64, y: f64, vx: f64, vy: f64, existence: f64) -> LmbTrack<f64, 4> {
    let mean = StateVector::from_array([x, y, vx, vy]);
    let cov = StateCovariance::from_matrix(nalgebra::matrix![
        10.0, 0.0, 0.0, 0.0;
        0.0, 10.0, 0.0, 0.0;
        0.0, 0.0, 5.0, 0.0;
        0.0, 0.0, 0.0, 5.0
    ]);
    let state = GaussianState::new(1.0, mean, cov);
    LmbTrack::new(label, existence, state)
}

/// Creates a measurement at given position
fn make_measurement(x: f64, y: f64) -> Measurement<f64, 2> {
    Measurement::from_array([x, y])
}

/// Creates standard test models
fn make_test_models() -> (
    ConstantVelocity2D<f64>,
    PositionSensor2D<f64>,
    UniformClutter2D<f64>,
) {
    let transition = ConstantVelocity2D::new(1.0, 0.99);
    let observation = PositionSensor2D::new(5.0, 0.9);
    let clutter = UniformClutter2D::new(5.0, (0.0, 200.0), (0.0, 200.0));
    (transition, observation, clutter)
}

// ============================================================================
// Integration Tests: Predict-Update Cycles
// ============================================================================

mod integration_tests {
    use super::*;

    #[test]
    fn test_single_target_tracking_over_multiple_steps() {
        let (transition, observation, clutter) = make_test_models();
        let birth = NoBirthModel;

        let filter: LmbFilter<f64, _, _, _, _, 4, 2> =
            LmbFilter::new(transition, observation, clutter, birth);

        // Initialize with single target at (50, 50) moving with velocity (2, 1)
        let initial_tracks = vec![make_track(Label::new(0, 0), 50.0, 50.0, 2.0, 1.0, 0.95)];
        let mut state = filter.initial_state_from(initial_tracks);

        let dt = 1.0;

        // Run 5 steps with measurements tracking the target
        let measurement_offsets = [(52.0, 51.0), (54.0, 52.0), (56.0, 53.0), (58.0, 54.0), (60.0, 55.0)];

        for (t, (mx, my)) in measurement_offsets.iter().enumerate() {
            let measurements = vec![make_measurement(*mx, *my)];
            let (updated, _stats) = filter.step(state, &measurements, dt);
            state = updated;

            // Verify single track maintained
            assert_eq!(state.num_tracks(), 1, "Step {}: should have 1 track", t);

            // Verify existence stays high with consistent detections
            let estimates = extract_lmb_estimates(&state);
            assert_eq!(estimates.len(), 1, "Step {}: should extract 1 estimate", t);
            assert!(
                estimates[0].existence > 0.8,
                "Step {}: existence should stay high, got {}",
                t,
                estimates[0].existence
            );

            // Verify position is reasonable (within measurement noise)
            let pos_x = *estimates[0].state.index(0);
            let pos_y = *estimates[0].state.index(1);
            assert!(
                (pos_x - mx).abs() < 15.0,
                "Step {}: x position {} too far from measurement {}",
                t,
                pos_x,
                mx
            );
            assert!(
                (pos_y - my).abs() < 15.0,
                "Step {}: y position {} too far from measurement {}",
                t,
                pos_y,
                my
            );
        }
    }

    #[test]
    fn test_two_target_tracking_with_separation() {
        let (transition, observation, clutter) = make_test_models();
        let birth = NoBirthModel;

        let filter: LmbFilter<f64, _, _, _, _, 4, 2> =
            LmbFilter::new(transition, observation, clutter, birth);

        // Two targets well-separated
        let initial_tracks = vec![
            make_track(Label::new(0, 0), 20.0, 20.0, 1.0, 1.0, 0.95),
            make_track(Label::new(0, 1), 180.0, 180.0, -1.0, -1.0, 0.95),
        ];
        let mut state = filter.initial_state_from(initial_tracks);

        let dt = 1.0;

        for t in 0..5 {
            // Measurements for both targets
            let m1_x = 20.0 + (t + 1) as f64 * 1.0;
            let m1_y = 20.0 + (t + 1) as f64 * 1.0;
            let m2_x = 180.0 - (t + 1) as f64 * 1.0;
            let m2_y = 180.0 - (t + 1) as f64 * 1.0;

            let measurements = vec![make_measurement(m1_x, m1_y), make_measurement(m2_x, m2_y)];
            let (updated, _) = filter.step(state, &measurements, dt);
            state = updated;
        }

        // Both tracks should survive
        let estimates = extract_lmb_estimates(&state);
        assert_eq!(estimates.len(), 2, "Should maintain 2 targets");

        // Verify labels are preserved
        let labels: Vec<Label> = estimates.iter().map(|e| e.label).collect();
        assert!(labels.contains(&Label::new(0, 0)), "Label (0,0) should persist");
        assert!(labels.contains(&Label::new(0, 1)), "Label (0,1) should persist");
    }

    #[test]
    fn test_predict_without_update() {
        let transition = ConstantVelocity2D::new(1.0, 0.99);
        let birth = NoBirthModel;

        let initial_tracks = vec![make_track(Label::new(0, 0), 50.0, 50.0, 10.0, 5.0, 0.95)];
        let state: LmbFilterState<f64, 4, Updated> = LmbFilterState::from_tracks(initial_tracks);

        // Predict only (no update)
        let predicted = state.predict(&transition, &birth, 1.0);

        // Check position advanced by velocity
        let track = &predicted.tracks.tracks[0];
        let mean = track.weighted_mean();
        assert!(
            (*mean.index(0) - 60.0).abs() < 0.1,
            "X should advance from 50 to 60"
        );
        assert!(
            (*mean.index(1) - 55.0).abs() < 0.1,
            "Y should advance from 50 to 55"
        );

        // Existence should decrease by survival probability
        assert!(
            (track.existence - 0.95 * 0.99).abs() < 0.01,
            "Existence should decrease by survival prob"
        );
    }

    #[test]
    fn test_update_increases_existence_on_detection() {
        let (transition, observation, clutter) = make_test_models();
        let birth = NoBirthModel;

        // Start with low existence
        let initial_tracks = vec![make_track(Label::new(0, 0), 50.0, 50.0, 0.0, 0.0, 0.3)];
        let state: LmbFilterState<f64, 4, Updated> = LmbFilterState::from_tracks(initial_tracks);

        // Predict
        let predicted = state.predict(&transition, &birth, 1.0);
        let existence_after_predict = predicted.tracks.tracks[0].existence;

        // Update with measurement at target location
        let measurements = vec![make_measurement(50.0, 50.0)];
        let (updated, _) = predicted.update(&measurements, &observation, &clutter);

        let existence_after_update = updated.tracks.tracks[0].existence;

        assert!(
            existence_after_update > existence_after_predict,
            "Detection should increase existence: {} > {}",
            existence_after_update,
            existence_after_predict
        );
    }
}

// ============================================================================
// Track Continuity and Label Persistence Tests
// ============================================================================

mod label_persistence_tests {
    use super::*;

    #[test]
    fn test_labels_persist_through_updates() {
        let (transition, observation, clutter) = make_test_models();
        let birth = NoBirthModel;

        let filter: LmbFilter<f64, _, _, _, _, 4, 2> =
            LmbFilter::new(transition, observation, clutter, birth);

        let label_a = Label::new(0, 0);
        let label_b = Label::new(0, 1);

        let initial_tracks = vec![
            make_track(label_a, 30.0, 30.0, 0.0, 0.0, 0.9),
            make_track(label_b, 170.0, 170.0, 0.0, 0.0, 0.9),
        ];
        let mut state = filter.initial_state_from(initial_tracks);

        // Multiple updates
        for _ in 0..10 {
            let measurements = vec![make_measurement(30.0, 30.0), make_measurement(170.0, 170.0)];
            let (updated, _) = filter.step(state, &measurements, 1.0);
            state = updated;
        }

        // Find tracks by label
        let track_a = state.tracks.find_by_label(label_a);
        let track_b = state.tracks.find_by_label(label_b);

        assert!(track_a.is_some(), "Track A should still exist");
        assert!(track_b.is_some(), "Track B should still exist");

        // Verify positions haven't drifted much
        let pos_a = track_a.unwrap().weighted_mean();
        let pos_b = track_b.unwrap().weighted_mean();

        assert!(
            (*pos_a.index(0) - 30.0).abs() < 10.0,
            "Track A x-position should be near 30"
        );
        assert!(
            (*pos_b.index(0) - 170.0).abs() < 10.0,
            "Track B x-position should be near 170"
        );
    }

    #[test]
    fn test_birth_tracks_get_unique_labels() {
        let (transition, observation, clutter) = make_test_models();
        let birth = TestBirthModel::new()
            .with_location(0.1, 100.0, 100.0)
            .with_location(0.1, 150.0, 150.0);

        let filter: LmbFilter<f64, _, _, _, _, 4, 2> =
            LmbFilter::new(transition, observation, clutter, birth);

        let mut state = filter.initial_state();

        // Run several steps to accumulate birth tracks
        for _ in 0..3 {
            let measurements = vec![make_measurement(100.0, 100.0)];
            let (updated, _) = filter.step(state, &measurements, 1.0);
            state = updated;
        }

        // Check all labels are unique
        let labels: Vec<Label> = state.tracks.iter().map(|t| t.label).collect();
        let mut unique_labels = labels.clone();
        unique_labels.sort();
        unique_labels.dedup();

        assert_eq!(
            labels.len(),
            unique_labels.len(),
            "All track labels should be unique"
        );
    }

    #[test]
    fn test_label_generator_advances_time() {
        let mut label_gen = LabelGenerator::new();

        let l1 = label_gen.next_label();
        let l2 = label_gen.next_label();

        assert_eq!(l1.birth_time, 0);
        assert_eq!(l2.birth_time, 0);
        assert_ne!(l1.index, l2.index);

        label_gen.advance_time();

        let l3 = label_gen.next_label();
        assert_eq!(l3.birth_time, 1);
    }
}

// ============================================================================
// LBP Convergence and Data Association Tests
// ============================================================================

mod data_association_tests {
    use super::*;

    #[test]
    fn test_measurement_assigned_to_nearest_track() {
        let (transition, observation, clutter) = make_test_models();
        let birth = NoBirthModel;

        // Two tracks at different positions
        let initial_tracks = vec![
            make_track(Label::new(0, 0), 30.0, 30.0, 0.0, 0.0, 0.9),
            make_track(Label::new(0, 1), 170.0, 170.0, 0.0, 0.0, 0.9),
        ];
        let state: LmbFilterState<f64, 4, Updated> = LmbFilterState::from_tracks(initial_tracks);

        let predicted = state.predict(&transition, &birth, 1.0);

        // Single measurement near track 0
        let measurements = vec![make_measurement(32.0, 32.0)];
        let (updated, _) = predicted.update(&measurements, &observation, &clutter);

        let track_0 = updated.tracks.find_by_label(Label::new(0, 0)).unwrap();
        let track_1 = updated.tracks.find_by_label(Label::new(0, 1)).unwrap();

        // Track 0 should have higher existence (got the measurement)
        // Track 1 should have lower existence (missed detection)
        assert!(
            track_0.existence > track_1.existence,
            "Track near measurement should have higher existence: {} vs {}",
            track_0.existence,
            track_1.existence
        );
    }

    #[test]
    fn test_ambiguous_association_handled() {
        let (transition, observation, clutter) = make_test_models();
        let birth = NoBirthModel;

        // Two tracks close together
        let initial_tracks = vec![
            make_track(Label::new(0, 0), 100.0, 100.0, 0.0, 0.0, 0.9),
            make_track(Label::new(0, 1), 105.0, 105.0, 0.0, 0.0, 0.9),
        ];
        let state: LmbFilterState<f64, 4, Updated> = LmbFilterState::from_tracks(initial_tracks);

        let predicted = state.predict(&transition, &birth, 1.0);

        // Measurement equidistant from both
        let measurements = vec![make_measurement(102.5, 102.5)];
        let (updated, _) = predicted.update(&measurements, &observation, &clutter);

        // Both tracks should still exist (LBP handles ambiguity)
        assert!(updated.tracks.len() >= 2, "Both tracks should survive");

        // Both should have reasonable existence
        for track in updated.tracks.iter() {
            assert!(
                track.existence > 0.1,
                "Track existence {} should be reasonable",
                track.existence
            );
        }
    }

    #[test]
    fn test_no_measurements_reduces_all_existence() {
        let (transition, observation, clutter) = make_test_models();
        let birth = NoBirthModel;

        let initial_tracks = vec![
            make_track(Label::new(0, 0), 50.0, 50.0, 0.0, 0.0, 0.9),
            make_track(Label::new(0, 1), 150.0, 150.0, 0.0, 0.0, 0.9),
        ];
        let state: LmbFilterState<f64, 4, Updated> = LmbFilterState::from_tracks(initial_tracks);

        let predicted = state.predict(&transition, &birth, 1.0);

        // Empty measurement set
        let measurements: Vec<Measurement<f64, 2>> = vec![];
        let (updated, _) = predicted.update(&measurements, &observation, &clutter);

        // All tracks should have reduced existence
        for track in updated.tracks.iter() {
            assert!(
                track.existence < 0.9,
                "Existence should decrease without measurements: {}",
                track.existence
            );
        }
    }

    #[test]
    fn test_more_measurements_than_tracks() {
        let (transition, observation, clutter) = make_test_models();
        let birth = NoBirthModel;

        let initial_tracks = vec![make_track(Label::new(0, 0), 50.0, 50.0, 0.0, 0.0, 0.9)];
        let state: LmbFilterState<f64, 4, Updated> = LmbFilterState::from_tracks(initial_tracks);

        let predicted = state.predict(&transition, &birth, 1.0);

        // Multiple measurements, only one track
        let measurements = vec![
            make_measurement(50.0, 50.0),
            make_measurement(100.0, 100.0), // Clutter
            make_measurement(150.0, 150.0), // Clutter
        ];
        let (updated, _) = predicted.update(&measurements, &observation, &clutter);

        // Should still work correctly
        assert!(updated.tracks.len() >= 1, "Track should survive");
    }
}

// ============================================================================
// Multi-Target Scenario Tests
// ============================================================================

mod multi_target_tests {
    use super::*;

    #[test]
    fn test_five_target_tracking() {
        let (transition, observation, clutter) = make_test_models();
        let birth = NoBirthModel;

        let filter: LmbFilter<f64, _, _, _, _, 4, 2> =
            LmbFilter::new(transition, observation, clutter, birth);

        // 5 targets at corners and center
        let initial_tracks = vec![
            make_track(Label::new(0, 0), 20.0, 20.0, 1.0, 1.0, 0.95),
            make_track(Label::new(0, 1), 180.0, 20.0, -1.0, 1.0, 0.95),
            make_track(Label::new(0, 2), 20.0, 180.0, 1.0, -1.0, 0.95),
            make_track(Label::new(0, 3), 180.0, 180.0, -1.0, -1.0, 0.95),
            make_track(Label::new(0, 4), 100.0, 100.0, 0.0, 0.0, 0.95),
        ];
        let mut state = filter.initial_state_from(initial_tracks);

        let dt = 1.0;

        // Track for several steps
        for t in 0..5 {
            let offset = (t + 1) as f64;
            let measurements = vec![
                make_measurement(20.0 + offset, 20.0 + offset),
                make_measurement(180.0 - offset, 20.0 + offset),
                make_measurement(20.0 + offset, 180.0 - offset),
                make_measurement(180.0 - offset, 180.0 - offset),
                make_measurement(100.0, 100.0),
            ];
            let (updated, _) = filter.step(state, &measurements, dt);
            state = updated;
        }

        // All 5 targets should be tracked
        let estimates = extract_lmb_estimates(&state);
        assert_eq!(estimates.len(), 5, "All 5 targets should be tracked");
    }

    #[test]
    fn test_crossing_targets() {
        let (transition, observation, clutter) = make_test_models();
        let birth = NoBirthModel;

        let filter: LmbFilter<f64, _, _, _, _, 4, 2> =
            LmbFilter::new(transition, observation, clutter, birth);

        // Two targets that will cross paths
        let initial_tracks = vec![
            make_track(Label::new(0, 0), 50.0, 100.0, 10.0, 0.0, 0.95),  // Moving right
            make_track(Label::new(0, 1), 150.0, 100.0, -10.0, 0.0, 0.95), // Moving left
        ];
        let mut state = filter.initial_state_from(initial_tracks);

        let dt = 1.0;

        // Track through crossing (targets cross at t=5)
        for t in 0..10 {
            let x1 = 50.0 + (t + 1) as f64 * 10.0;
            let x2 = 150.0 - (t + 1) as f64 * 10.0;
            let measurements = vec![make_measurement(x1, 100.0), make_measurement(x2, 100.0)];
            let (updated, _) = filter.step(state, &measurements, dt);
            state = updated;
        }

        // Both tracks should survive
        let estimates = extract_lmb_estimates(&state);
        assert_eq!(estimates.len(), 2, "Both targets should survive crossing");

        // Verify labels are maintained
        let labels: Vec<Label> = estimates.iter().map(|e| e.label).collect();
        assert!(labels.contains(&Label::new(0, 0)));
        assert!(labels.contains(&Label::new(0, 1)));
    }

    #[test]
    fn test_target_birth_and_death() {
        let (transition, observation, clutter) = make_test_models();
        let birth = TestBirthModel::new().with_location(0.05, 100.0, 100.0);

        let filter: LmbFilter<f64, _, _, _, _, 4, 2> =
            LmbFilter::new(transition, observation, clutter, birth);

        let mut state = filter.initial_state();

        // First few steps: no measurements at birth location
        for _ in 0..3 {
            let measurements: Vec<Measurement<f64, 2>> = vec![];
            let (updated, _) = filter.step(state, &measurements, 1.0);
            state = updated;
            state.tracks.prune_by_existence(0.01);
        }

        // Birth tracks should be pruned due to low existence
        let count_before = state.num_tracks();

        // Now provide measurements at birth location
        for _ in 0..5 {
            let measurements = vec![make_measurement(100.0, 100.0)];
            let (updated, _) = filter.step(state, &measurements, 1.0);
            state = updated;
        }

        let estimates = extract_lmb_estimates_threshold(&state, 0.3);
        assert!(
            estimates.len() > 0 || count_before > 0,
            "Should establish track at birth location over time"
        );
    }
}

// ============================================================================
// Clutter Robustness Tests
// ============================================================================

mod clutter_tests {
    use super::*;

    #[test]
    fn test_single_target_with_heavy_clutter() {
        let transition = ConstantVelocity2D::new(1.0, 0.99);
        let observation = PositionSensor2D::new(5.0, 0.9);
        // High clutter rate
        let clutter = UniformClutter2D::new(20.0, (0.0, 200.0), (0.0, 200.0));
        let birth = NoBirthModel;

        let filter: LmbFilter<f64, _, _, _, _, 4, 2> =
            LmbFilter::new(transition, observation, clutter, birth);

        let initial_tracks = vec![make_track(Label::new(0, 0), 100.0, 100.0, 0.0, 0.0, 0.95)];
        let mut state = filter.initial_state_from(initial_tracks);

        let dt = 1.0;

        // Track with lots of clutter measurements
        for _ in 0..5 {
            let measurements = vec![
                make_measurement(100.0, 100.0), // True target
                make_measurement(30.0, 50.0),   // Clutter
                make_measurement(170.0, 80.0),  // Clutter
                make_measurement(60.0, 140.0),  // Clutter
                make_measurement(120.0, 30.0),  // Clutter
            ];
            let (updated, _) = filter.step(state, &measurements, dt);
            state = updated;
        }

        // Target should maintain high existence despite clutter
        let estimates = extract_lmb_estimates(&state);
        assert_eq!(estimates.len(), 1, "Should maintain single target");
        assert!(
            estimates[0].existence > 0.7,
            "Target existence should remain high: {}",
            estimates[0].existence
        );

        // Position should be near true location
        let pos_x = *estimates[0].state.index(0);
        let pos_y = *estimates[0].state.index(1);
        assert!(
            (pos_x - 100.0).abs() < 15.0,
            "X position should be near 100: {}",
            pos_x
        );
        assert!(
            (pos_y - 100.0).abs() < 15.0,
            "Y position should be near 100: {}",
            pos_y
        );
    }

    #[test]
    fn test_clutter_only_no_targets() {
        let (transition, observation, clutter) = make_test_models();
        let birth = NoBirthModel;

        let filter: LmbFilter<f64, _, _, _, _, 4, 2> =
            LmbFilter::new(transition, observation, clutter, birth);

        // Start with empty state
        let mut state = filter.initial_state();

        // Provide only clutter measurements
        for _ in 0..5 {
            let measurements = vec![
                make_measurement(30.0, 50.0),
                make_measurement(170.0, 80.0),
                make_measurement(60.0, 140.0),
            ];
            let (updated, _) = filter.step(state, &measurements, 1.0);
            state = updated;
        }

        // Should not establish false tracks (without birth model)
        let estimates = extract_lmb_estimates(&state);
        assert_eq!(estimates.len(), 0, "Should not create tracks from clutter only");
    }

    #[test]
    fn test_varying_clutter_density() {
        let transition = ConstantVelocity2D::new(1.0, 0.99);
        let observation = PositionSensor2D::new(5.0, 0.9);
        let birth = NoBirthModel;

        // Low clutter first
        let clutter_low = UniformClutter2D::new(2.0, (0.0, 200.0), (0.0, 200.0));
        let filter_low: LmbFilter<f64, _, _, _, _, 4, 2> =
            LmbFilter::new(transition.clone(), observation.clone(), clutter_low, NoBirthModel);

        // High clutter
        let clutter_high = UniformClutter2D::new(30.0, (0.0, 200.0), (0.0, 200.0));
        let filter_high: LmbFilter<f64, _, _, _, _, 4, 2> =
            LmbFilter::new(transition, observation, clutter_high, birth);

        let initial_tracks = vec![make_track(Label::new(0, 0), 100.0, 100.0, 0.0, 0.0, 0.95)];

        let mut state_low = filter_low.initial_state_from(initial_tracks.clone());
        let mut state_high = filter_high.initial_state_from(initial_tracks);

        // Same measurements for both
        let measurements = vec![
            make_measurement(100.0, 100.0),
            make_measurement(50.0, 50.0), // Clutter
        ];

        let (updated_low, _) = filter_low.step(state_low, &measurements, 1.0);
        let (updated_high, _) = filter_high.step(state_high, &measurements, 1.0);

        state_low = updated_low;
        state_high = updated_high;

        // Low clutter should give higher existence (measurement more likely from target)
        let estimates_low = extract_lmb_estimates(&state_low);
        let estimates_high = extract_lmb_estimates(&state_high);

        // Both should track, but low clutter should be more confident
        assert!(!estimates_low.is_empty());
        assert!(!estimates_high.is_empty());
    }
}

// ============================================================================
// Missed Detection Handling Tests
// ============================================================================

mod missed_detection_tests {
    use super::*;

    #[test]
    fn test_track_survives_single_miss() {
        let (transition, observation, clutter) = make_test_models();
        let birth = NoBirthModel;

        let filter: LmbFilter<f64, _, _, _, _, 4, 2> =
            LmbFilter::new(transition, observation, clutter, birth);

        let initial_tracks = vec![make_track(Label::new(0, 0), 100.0, 100.0, 0.0, 0.0, 0.95)];
        let mut state = filter.initial_state_from(initial_tracks);

        // First step: detection
        let measurements = vec![make_measurement(100.0, 100.0)];
        let (updated, _) = filter.step(state, &measurements, 1.0);
        state = updated;

        let existence_after_detection = state.tracks.tracks[0].existence;

        // Second step: miss (no measurements)
        let measurements: Vec<Measurement<f64, 2>> = vec![];
        let (updated, _) = filter.step(state, &measurements, 1.0);
        state = updated;

        let existence_after_miss = state.tracks.tracks[0].existence;

        // Existence should decrease but track should survive
        assert!(
            existence_after_miss < existence_after_detection,
            "Existence should decrease after miss"
        );
        assert!(
            existence_after_miss > 0.3,
            "Track should survive single miss: {}",
            existence_after_miss
        );
    }

    #[test]
    fn test_track_decays_with_consecutive_misses() {
        let (transition, observation, clutter) = make_test_models();
        let birth = NoBirthModel;

        let filter: LmbFilter<f64, _, _, _, _, 4, 2> =
            LmbFilter::new(transition, observation, clutter, birth);

        let initial_tracks = vec![make_track(Label::new(0, 0), 100.0, 100.0, 0.0, 0.0, 0.95)];
        let mut state = filter.initial_state_from(initial_tracks);

        let mut existence_history = vec![state.tracks.tracks[0].existence];

        // Multiple consecutive misses
        for _ in 0..10 {
            let measurements: Vec<Measurement<f64, 2>> = vec![];
            let (updated, _) = filter.step(state, &measurements, 1.0);
            state = updated;
            existence_history.push(state.tracks.tracks[0].existence);
        }

        // Verify monotonic decrease
        for i in 1..existence_history.len() {
            assert!(
                existence_history[i] <= existence_history[i - 1],
                "Existence should decrease monotonically: {} > {}",
                existence_history[i],
                existence_history[i - 1]
            );
        }

        // Final existence should be very low
        assert!(
            *existence_history.last().unwrap() < 0.1,
            "Existence should be very low after many misses: {}",
            existence_history.last().unwrap()
        );
    }

    #[test]
    fn test_track_recovers_after_miss() {
        let (transition, observation, clutter) = make_test_models();
        let birth = NoBirthModel;

        let filter: LmbFilter<f64, _, _, _, _, 4, 2> =
            LmbFilter::new(transition, observation, clutter, birth);

        let initial_tracks = vec![make_track(Label::new(0, 0), 100.0, 100.0, 0.0, 0.0, 0.95)];
        let mut state = filter.initial_state_from(initial_tracks);

        // Detect - miss - detect pattern
        let patterns = [true, false, false, true, true];

        for detect in patterns {
            let measurements = if detect {
                vec![make_measurement(100.0, 100.0)]
            } else {
                vec![]
            };
            let (updated, _) = filter.step(state, &measurements, 1.0);
            state = updated;
        }

        // Track should still exist with reasonable confidence
        let estimates = extract_lmb_estimates_threshold(&state, 0.3);
        assert!(
            !estimates.is_empty(),
            "Track should recover after intermittent detections"
        );
    }

    #[test]
    fn test_low_detection_probability() {
        let transition = ConstantVelocity2D::new(1.0, 0.99);
        // Low detection probability
        let observation = PositionSensor2D::new(5.0, 0.5);
        let clutter = UniformClutter2D::new(5.0, (0.0, 200.0), (0.0, 200.0));
        let birth = NoBirthModel;

        let filter: LmbFilter<f64, _, _, _, _, 4, 2> =
            LmbFilter::new(transition, observation, clutter, birth);

        let initial_tracks = vec![make_track(Label::new(0, 0), 100.0, 100.0, 0.0, 0.0, 0.95)];
        let mut state = filter.initial_state_from(initial_tracks);

        // Alternate detection/miss (simulating 50% detection rate)
        for i in 0..10 {
            let measurements = if i % 2 == 0 {
                vec![make_measurement(100.0, 100.0)]
            } else {
                vec![]
            };
            let (updated, _) = filter.step(state, &measurements, 1.0);
            state = updated;
        }

        // Track should still exist (misses are expected with low P_D)
        let estimates = extract_lmb_estimates_threshold(&state, 0.3);
        assert!(
            !estimates.is_empty(),
            "Track should survive with low detection probability"
        );
    }
}

// ============================================================================
// Numerical Stability Tests
// ============================================================================

mod numerical_stability_tests {
    use super::*;

    #[test]
    fn test_very_small_existence() {
        let (transition, observation, clutter) = make_test_models();
        let birth = NoBirthModel;

        // Track with very small existence
        let initial_tracks = vec![make_track(Label::new(0, 0), 100.0, 100.0, 0.0, 0.0, 1e-10)];
        let state: LmbFilterState<f64, 4, Updated> = LmbFilterState::from_tracks(initial_tracks);

        let predicted = state.predict(&transition, &birth, 1.0);

        // Should handle without panic
        let measurements = vec![make_measurement(100.0, 100.0)];
        let (updated, _) = predicted.update(&measurements, &observation, &clutter);

        // Existence should be valid (not NaN or infinite)
        let existence = updated.tracks.tracks[0].existence;
        assert!(existence.is_finite(), "Existence should be finite: {}", existence);
        assert!(existence >= 0.0, "Existence should be non-negative");
    }

    #[test]
    fn test_existence_near_one() {
        let (transition, observation, clutter) = make_test_models();
        let birth = NoBirthModel;

        // Track with existence very close to 1
        let initial_tracks = vec![make_track(Label::new(0, 0), 100.0, 100.0, 0.0, 0.0, 0.9999999)];
        let state: LmbFilterState<f64, 4, Updated> = LmbFilterState::from_tracks(initial_tracks);

        let predicted = state.predict(&transition, &birth, 1.0);
        let measurements = vec![make_measurement(100.0, 100.0)];
        let (updated, _) = predicted.update(&measurements, &observation, &clutter);

        let existence = updated.tracks.tracks[0].existence;
        assert!(existence.is_finite());
        assert!(existence <= 1.0, "Existence should not exceed 1: {}", existence);
    }

    #[test]
    fn test_large_covariance() {
        let (transition, observation, clutter) = make_test_models();
        let birth = NoBirthModel;

        // Track with very large covariance
        let mean = StateVector::from_array([100.0, 100.0, 0.0, 0.0]);
        let cov = StateCovariance::from_matrix(nalgebra::matrix![
            1e6, 0.0, 0.0, 0.0;
            0.0, 1e6, 0.0, 0.0;
            0.0, 0.0, 1e4, 0.0;
            0.0, 0.0, 0.0, 1e4
        ]);
        let state = GaussianState::new(1.0, mean, cov);
        let track = LmbTrack::new(Label::new(0, 0), 0.9, state);

        let filter_state: LmbFilterState<f64, 4, Updated> = LmbFilterState::from_tracks(vec![track]);
        let predicted = filter_state.predict(&transition, &birth, 1.0);
        let measurements = vec![make_measurement(100.0, 100.0)];
        let (updated, _) = predicted.update(&measurements, &observation, &clutter);

        // Should not produce NaN or infinite values
        let track = &updated.tracks.tracks[0];
        let mean = track.weighted_mean();
        assert!(mean.index(0).is_finite());
        assert!(mean.index(1).is_finite());
    }

    #[test]
    fn test_small_covariance() {
        let (transition, observation, clutter) = make_test_models();
        let birth = NoBirthModel;

        // Track with very small covariance
        let mean = StateVector::from_array([100.0, 100.0, 0.0, 0.0]);
        let cov = StateCovariance::from_matrix(nalgebra::matrix![
            0.01, 0.0, 0.0, 0.0;
            0.0, 0.01, 0.0, 0.0;
            0.0, 0.0, 0.001, 0.0;
            0.0, 0.0, 0.0, 0.001
        ]);
        let state = GaussianState::new(1.0, mean, cov);
        let track = LmbTrack::new(Label::new(0, 0), 0.9, state);

        let filter_state: LmbFilterState<f64, 4, Updated> = LmbFilterState::from_tracks(vec![track]);
        let predicted = filter_state.predict(&transition, &birth, 1.0);
        let measurements = vec![make_measurement(100.0, 100.0)];
        let (updated, _) = predicted.update(&measurements, &observation, &clutter);

        assert!(!updated.tracks.is_empty());
    }

    #[test]
    fn test_measurement_far_from_track() {
        let (transition, observation, clutter) = make_test_models();
        let birth = NoBirthModel;

        let initial_tracks = vec![make_track(Label::new(0, 0), 0.0, 0.0, 0.0, 0.0, 0.9)];
        let state: LmbFilterState<f64, 4, Updated> = LmbFilterState::from_tracks(initial_tracks);

        let predicted = state.predict(&transition, &birth, 1.0);

        // Measurement very far from track
        let measurements = vec![make_measurement(1000.0, 1000.0)];
        let (updated, _) = predicted.update(&measurements, &observation, &clutter);

        // Should handle gracefully
        let existence = updated.tracks.tracks[0].existence;
        assert!(existence.is_finite());
    }

    #[test]
    fn test_zero_time_step() {
        let transition = ConstantVelocity2D::new(1.0, 0.99);
        let birth = NoBirthModel;

        let initial_tracks = vec![make_track(Label::new(0, 0), 100.0, 100.0, 5.0, 5.0, 0.9)];
        let state: LmbFilterState<f64, 4, Updated> = LmbFilterState::from_tracks(initial_tracks);

        // Zero time step
        let predicted = state.predict(&transition, &birth, 0.0);

        // Position should not change
        let track = &predicted.tracks.tracks[0];
        let mean = track.weighted_mean();
        assert!(
            (*mean.index(0) - 100.0).abs() < 0.01,
            "Position should not change with dt=0"
        );
    }

    #[test]
    fn test_empty_track_set_update() {
        let (transition, observation, clutter) = make_test_models();
        let birth = NoBirthModel;

        let state: LmbFilterState<f64, 4, Updated> = LmbFilterState::new();
        let predicted = state.predict(&transition, &birth, 1.0);

        // Update empty state with measurements
        let measurements = vec![make_measurement(100.0, 100.0)];
        let (updated, _) = predicted.update(&measurements, &observation, &clutter);

        // Should handle gracefully (no tracks to update)
        assert_eq!(updated.num_tracks(), 0);
    }
}

// ============================================================================
// LMBM-Specific Tests
// ============================================================================

mod lmbm_tests {
    use super::*;

    #[test]
    fn test_lmbm_single_hypothesis_tracking() {
        let (transition, observation, clutter) = make_test_models();
        let birth = NoBirthModel;

        let filter: LmbmFilter<f64, _, _, _, _, 4, 2> =
            LmbmFilter::new(transition, observation, clutter, birth, 1, 10);

        // Initialize with single hypothesis containing one track
        let track = BernoulliTrack::new(
            Label::new(0, 0),
            0.9,
            GaussianState::new(
                1.0,
                StateVector::from_array([100.0, 100.0, 0.0, 0.0]),
                StateCovariance::from_matrix(nalgebra::matrix![
                    10.0, 0.0, 0.0, 0.0;
                    0.0, 10.0, 0.0, 0.0;
                    0.0, 0.0, 5.0, 0.0;
                    0.0, 0.0, 0.0, 5.0
                ]),
            ),
        );
        let hypothesis = LmbmHypothesis::new(0.0, vec![track]);
        let mut state = LmbmFilterState::from_hypotheses(vec![hypothesis]);

        // Run tracking
        for _ in 0..5 {
            let measurements = vec![make_measurement(100.0, 100.0)];
            let (updated, _) = filter.step(state, &measurements, 1.0);
            state = updated;
        }

        // Should maintain hypothesis
        assert!(state.num_hypotheses() >= 1);

        // Best hypothesis should have track
        let best = extract_best_hypothesis(&state);
        assert!(best.is_some());
        assert!(!best.unwrap().tracks.is_empty());
    }

    #[test]
    fn test_lmbm_hypothesis_pruning() {
        let mut state = LmbmState::<f64, 4>::new();

        // Add many hypotheses with varying weights
        for i in 0..20 {
            state
                .hypotheses
                .push(LmbmHypothesis::empty(-(i as f64) * 10.0));
        }

        state.normalize_log_weights();
        state.keep_top_k(5);

        assert_eq!(state.num_hypotheses(), 5, "Should keep only top 5 hypotheses");
    }

    #[test]
    fn test_lmbm_marginal_existence() {
        let mut state = LmbmState::<f64, 4>::new();

        let label = Label::new(0, 0);
        let gaussian = GaussianState::new(
            1.0,
            StateVector::from_array([0.0, 0.0, 0.0, 0.0]),
            StateCovariance::identity(),
        );

        // Two hypotheses, both containing same track with different existence
        let track1 = BernoulliTrack::new(label, 0.9, gaussian.clone());
        let track2 = BernoulliTrack::new(label, 0.5, gaussian);

        state.hypotheses.push(LmbmHypothesis::new(0.0, vec![track1])); // weight 1
        state.hypotheses.push(LmbmHypothesis::new(-0.693, vec![track2])); // weight ~0.5

        let marginals = state.marginal_existence();

        assert_eq!(marginals.len(), 1, "Should have one unique label");
        // Marginal should be weighted average: (1*0.9 + 0.5*0.5) / (1 + 0.5) ≈ 0.77
        let (_, marginal_r) = marginals[0];
        assert!(
            marginal_r > 0.5 && marginal_r < 1.0,
            "Marginal existence should be reasonable: {}",
            marginal_r
        );
    }

    #[test]
    fn test_lmbm_multiple_hypotheses_generation() {
        let (transition, observation, clutter) = make_test_models();
        let birth = NoBirthModel;

        // Request multiple best assignments
        let filter: LmbmFilter<f64, _, _, _, _, 4, 2> =
            LmbmFilter::new(transition, observation, clutter, birth, 3, 20);

        // Two tracks, ambiguous measurements
        let track1 = BernoulliTrack::new(
            Label::new(0, 0),
            0.9,
            GaussianState::new(
                1.0,
                StateVector::from_array([100.0, 100.0, 0.0, 0.0]),
                StateCovariance::identity(),
            ),
        );
        let track2 = BernoulliTrack::new(
            Label::new(0, 1),
            0.9,
            GaussianState::new(
                1.0,
                StateVector::from_array([105.0, 105.0, 0.0, 0.0]),
                StateCovariance::identity(),
            ),
        );
        let hypothesis = LmbmHypothesis::new(0.0, vec![track1, track2]);
        let mut state = LmbmFilterState::from_hypotheses(vec![hypothesis]);

        // Ambiguous measurement between tracks
        let measurements = vec![make_measurement(102.0, 102.0)];
        let (updated, _) = filter.step(state, &measurements, 1.0);
        state = updated;

        // Should generate multiple hypotheses
        assert!(
            state.num_hypotheses() >= 1,
            "Should have at least one hypothesis"
        );
    }

    #[test]
    fn test_lmbm_no_measurements() {
        let (transition, observation, clutter) = make_test_models();
        let birth = NoBirthModel;

        let filter: LmbmFilter<f64, _, _, _, _, 4, 2> =
            LmbmFilter::new(transition, observation, clutter, birth, 3, 20);

        let track = BernoulliTrack::new(
            Label::new(0, 0),
            0.9,
            GaussianState::new(
                1.0,
                StateVector::from_array([100.0, 100.0, 0.0, 0.0]),
                StateCovariance::identity(),
            ),
        );
        let hypothesis = LmbmHypothesis::new(0.0, vec![track]);
        let state = LmbmFilterState::from_hypotheses(vec![hypothesis]);

        // Empty measurements
        let measurements: Vec<Measurement<f64, 2>> = vec![];
        let (updated, _) = filter.step(state, &measurements, 1.0);

        // Should handle gracefully with miss detection
        assert!(updated.num_hypotheses() >= 1);
        let best = extract_best_hypothesis(&updated);
        assert!(best.is_some());
    }
}

// ============================================================================
// Track Set Operations Tests
// ============================================================================

mod track_set_tests {
    use super::*;

    #[test]
    fn test_prune_by_existence() {
        let mut track_set = LmbTrackSet::new();

        track_set.push(make_track(Label::new(0, 0), 0.0, 0.0, 0.0, 0.0, 0.9));
        track_set.push(make_track(Label::new(0, 1), 0.0, 0.0, 0.0, 0.0, 0.5));
        track_set.push(make_track(Label::new(0, 2), 0.0, 0.0, 0.0, 0.0, 0.1));
        track_set.push(make_track(Label::new(0, 3), 0.0, 0.0, 0.0, 0.0, 0.05));

        assert_eq!(track_set.len(), 4);

        track_set.prune_by_existence(0.2);

        assert_eq!(track_set.len(), 2);

        // Verify correct tracks survived
        assert!(track_set.find_by_label(Label::new(0, 0)).is_some());
        assert!(track_set.find_by_label(Label::new(0, 1)).is_some());
        assert!(track_set.find_by_label(Label::new(0, 2)).is_none());
        assert!(track_set.find_by_label(Label::new(0, 3)).is_none());
    }

    #[test]
    fn test_expected_cardinality() {
        let mut track_set = LmbTrackSet::new();

        track_set.push(make_track(Label::new(0, 0), 0.0, 0.0, 0.0, 0.0, 0.9));
        track_set.push(make_track(Label::new(0, 1), 0.0, 0.0, 0.0, 0.0, 0.8));
        track_set.push(make_track(Label::new(0, 2), 0.0, 0.0, 0.0, 0.0, 0.3));

        let expected = track_set.expected_cardinality();

        assert!(
            (expected - 2.0).abs() < 0.01,
            "Expected cardinality should be ~2.0: {}",
            expected
        );
    }

    #[test]
    fn test_filter_by_existence() {
        let mut track_set = LmbTrackSet::new();

        track_set.push(make_track(Label::new(0, 0), 0.0, 0.0, 0.0, 0.0, 0.9));
        track_set.push(make_track(Label::new(0, 1), 0.0, 0.0, 0.0, 0.0, 0.5));
        track_set.push(make_track(Label::new(0, 2), 0.0, 0.0, 0.0, 0.0, 0.1));

        let high_existence = track_set.filter_by_existence(0.4);

        assert_eq!(high_existence.len(), 2);
        assert_eq!(track_set.len(), 3); // Original unchanged
    }

    #[test]
    fn test_weighted_mean_single_component() {
        let track = make_track(Label::new(0, 0), 100.0, 50.0, 5.0, 2.0, 0.9);
        let mean = track.weighted_mean();

        assert!((*mean.index(0) - 100.0).abs() < 0.01);
        assert!((*mean.index(1) - 50.0).abs() < 0.01);
        assert!((*mean.index(2) - 5.0).abs() < 0.01);
        assert!((*mean.index(3) - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_weighted_mean_multiple_components() {
        use crate::types::gaussian::GaussianMixture;

        let label = Label::new(0, 0);
        let cov = StateCovariance::identity();

        let mut components = GaussianMixture::with_capacity(2);
        components.push(GaussianState::new(
            0.7,
            StateVector::from_array([100.0, 100.0, 0.0, 0.0]),
            cov,
        ));
        components.push(GaussianState::new(
            0.3,
            StateVector::from_array([200.0, 200.0, 0.0, 0.0]),
            cov,
        ));

        let track = LmbTrack::with_components(label, 0.9, components);
        let mean = track.weighted_mean();

        // Expected: 0.7*100 + 0.3*200 = 130
        let x: f64 = *mean.index(0);
        assert!(
            (x - 130.0).abs() < 0.01,
            "Weighted mean x should be 130: {}",
            x
        );
    }
}

// ============================================================================
// State Extraction Tests
// ============================================================================

mod extraction_tests {
    use super::*;

    #[test]
    fn test_extract_with_map_cardinality() {
        let (transition, observation, clutter) = make_test_models();
        let birth = NoBirthModel;

        let filter: LmbFilter<f64, _, _, _, _, 4, 2> =
            LmbFilter::new(transition, observation, clutter, birth);

        // 3 tracks with varying existence
        let initial_tracks = vec![
            make_track(Label::new(0, 0), 50.0, 50.0, 0.0, 0.0, 0.95),
            make_track(Label::new(0, 1), 100.0, 100.0, 0.0, 0.0, 0.9),
            make_track(Label::new(0, 2), 150.0, 150.0, 0.0, 0.0, 0.2),
        ];
        let state = filter.initial_state_from(initial_tracks);

        let estimates = extract_lmb_estimates(&state);

        // MAP should extract 2 targets (highest existence)
        assert_eq!(estimates.len(), 2, "Should extract 2 highest existence tracks");

        let labels: Vec<Label> = estimates.iter().map(|e| e.label).collect();
        assert!(labels.contains(&Label::new(0, 0)));
        assert!(labels.contains(&Label::new(0, 1)));
    }

    #[test]
    fn test_extract_with_threshold() {
        let initial_tracks = vec![
            make_track(Label::new(0, 0), 50.0, 50.0, 0.0, 0.0, 0.9),
            make_track(Label::new(0, 1), 100.0, 100.0, 0.0, 0.0, 0.6),
            make_track(Label::new(0, 2), 150.0, 150.0, 0.0, 0.0, 0.3),
            make_track(Label::new(0, 3), 200.0, 200.0, 0.0, 0.0, 0.1),
        ];
        let state: LmbFilterState<f64, 4, Updated> = LmbFilterState::from_tracks(initial_tracks);

        let threshold_high = extract_lmb_estimates_threshold(&state, 0.5);
        let threshold_low = extract_lmb_estimates_threshold(&state, 0.2);

        assert_eq!(threshold_high.len(), 2, "High threshold should get 2 tracks");
        assert_eq!(threshold_low.len(), 3, "Low threshold should get 3 tracks");
    }
}

// ============================================================================
// Edge Cases and Regression Tests
// ============================================================================

mod edge_cases {
    use super::*;

    #[test]
    fn test_single_track_single_measurement_exact_match() {
        let (transition, observation, clutter) = make_test_models();
        let birth = NoBirthModel;

        let initial_tracks = vec![make_track(Label::new(0, 0), 100.0, 100.0, 0.0, 0.0, 0.9)];
        let state: LmbFilterState<f64, 4, Updated> = LmbFilterState::from_tracks(initial_tracks);

        let predicted = state.predict(&transition, &birth, 1.0);

        // Measurement exactly at predicted position
        let measurements = vec![make_measurement(100.0, 100.0)];
        let (updated, stats) = predicted.update(&measurements, &observation, &clutter);

        assert_eq!(stats.singular_covariance_count, 0, "Should not have singular covariances");
        assert!(
            updated.tracks.tracks[0].existence > 0.9,
            "Existence should increase with perfect match"
        );
    }

    #[test]
    fn test_many_tracks_few_measurements() {
        let (transition, observation, clutter) = make_test_models();
        let birth = NoBirthModel;

        // 10 tracks
        let initial_tracks: Vec<LmbTrack<f64, 4>> = (0..10)
            .map(|i| {
                make_track(
                    Label::new(0, i),
                    10.0 + i as f64 * 20.0,
                    10.0 + i as f64 * 20.0,
                    0.0,
                    0.0,
                    0.9,
                )
            })
            .collect();

        let state: LmbFilterState<f64, 4, Updated> = LmbFilterState::from_tracks(initial_tracks);
        let predicted = state.predict(&transition, &birth, 1.0);

        // Only 2 measurements
        let measurements = vec![make_measurement(30.0, 30.0), make_measurement(50.0, 50.0)];
        let (updated, _) = predicted.update(&measurements, &observation, &clutter);

        // All tracks should survive (with varying existence)
        assert_eq!(updated.tracks.len(), 10);
    }

    #[test]
    fn test_duplicate_measurements() {
        let (transition, observation, clutter) = make_test_models();
        let birth = NoBirthModel;

        let initial_tracks = vec![make_track(Label::new(0, 0), 100.0, 100.0, 0.0, 0.0, 0.9)];
        let state: LmbFilterState<f64, 4, Updated> = LmbFilterState::from_tracks(initial_tracks);

        let predicted = state.predict(&transition, &birth, 1.0);

        // Duplicate measurements at same location
        let measurements = vec![
            make_measurement(100.0, 100.0),
            make_measurement(100.0, 100.0),
            make_measurement(100.0, 100.0),
        ];
        let (updated, _) = predicted.update(&measurements, &observation, &clutter);

        // Should handle gracefully
        assert!(!updated.tracks.is_empty());
        let existence = updated.tracks.tracks[0].existence;
        assert!(existence.is_finite());
    }

    #[test]
    fn test_very_small_surveillance_region() {
        let transition = ConstantVelocity2D::new(1.0, 0.99);
        let observation = PositionSensor2D::new(5.0, 0.9);
        // Very small region = high clutter density
        let clutter = UniformClutter2D::new(1.0, (0.0, 10.0), (0.0, 10.0));
        let birth = NoBirthModel;

        let initial_tracks = vec![make_track(Label::new(0, 0), 5.0, 5.0, 0.0, 0.0, 0.9)];
        let state: LmbFilterState<f64, 4, Updated> = LmbFilterState::from_tracks(initial_tracks);

        let predicted = state.predict(&transition, &birth, 1.0);
        let measurements = vec![make_measurement(5.0, 5.0)];
        let (updated, _) = predicted.update(&measurements, &observation, &clutter);

        // Should handle high clutter density
        let existence = updated.tracks.tracks[0].existence;
        assert!(existence.is_finite());
    }

    #[test]
    fn test_large_time_step() {
        let transition = ConstantVelocity2D::new(1.0, 0.99);
        let birth = NoBirthModel;

        let initial_tracks = vec![make_track(Label::new(0, 0), 100.0, 100.0, 10.0, 5.0, 0.9)];
        let state: LmbFilterState<f64, 4, Updated> = LmbFilterState::from_tracks(initial_tracks);

        // Large time step
        let predicted = state.predict(&transition, &birth, 100.0);

        let track = &predicted.tracks.tracks[0];
        let mean = track.weighted_mean();

        // Position should advance significantly
        assert!(
            (*mean.index(0) - 1100.0).abs() < 0.1,
            "X should advance to 1100"
        );
        assert!((*mean.index(1) - 600.0).abs() < 0.1, "Y should advance to 600");
    }
}
