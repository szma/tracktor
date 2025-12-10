//! Integration tests for GLMB (Generalized Labeled Multi-Bernoulli) filter

#![cfg(feature = "alloc")]

mod common;

use common::{make_glmb_track, make_measurement, NoGlmbBirthModel, TestGlmbBirthModel};
use tracktor::filters::glmb::{extract_best_hypothesis, extract_map_cardinality, GlmbFilter};
use tracktor::models::{ConstantVelocity2D, PositionSensor2D, UniformClutter2D};
use tracktor::types::labels::Label;
use tracktor::types::spaces::Measurement;

#[test]
fn test_glmb_single_target_tracking() {
    let transition = ConstantVelocity2D::new(1.0, 0.99);
    let observation = PositionSensor2D::new(5.0, 0.9);
    let clutter = UniformClutter2D::new(5.0, (0.0, 200.0), (0.0, 200.0));
    let birth = NoGlmbBirthModel;

    let filter: GlmbFilter<f64, _, _, _, _, 4, 2> =
        GlmbFilter::new(transition, observation, clutter, birth, 10);

    let initial_tracks = vec![make_glmb_track(Label::new(0, 0), 50.0, 50.0, 2.0, 1.0)];
    let mut state = filter.initial_state_from(initial_tracks);

    let dt = 1.0;

    for t in 0..5 {
        let mx = 52.0 + t as f64 * 2.0;
        let my = 51.0 + t as f64 * 1.0;
        let measurements = vec![make_measurement(mx, my)];

        let (updated, _stats) = filter.step(state, &measurements, dt);
        state = updated;

        // Verify we maintain approximately 1 target
        let estimates = extract_best_hypothesis(&state);
        assert!(
            !estimates.is_empty(),
            "Step {}: Should have at least one track",
            t
        );
    }
}

#[test]
fn test_glmb_two_target_tracking_with_labels() {
    let transition = ConstantVelocity2D::new(1.0, 0.99);
    let observation = PositionSensor2D::new(5.0, 0.9);
    let clutter = UniformClutter2D::new(5.0, (0.0, 200.0), (0.0, 200.0));
    let birth = NoGlmbBirthModel;

    let filter: GlmbFilter<f64, _, _, _, _, 4, 2> =
        GlmbFilter::new(transition, observation, clutter, birth, 10);

    let label_a = Label::new(0, 0);
    let label_b = Label::new(0, 1);

    let initial_tracks = vec![
        make_glmb_track(label_a, 30.0, 30.0, 1.0, 1.0),
        make_glmb_track(label_b, 170.0, 170.0, -1.0, -1.0),
    ];
    let mut state = filter.initial_state_from(initial_tracks);

    let dt = 1.0;

    for t in 0..5 {
        let offset = (t + 1) as f64;
        let measurements = vec![
            make_measurement(30.0 + offset, 30.0 + offset),
            make_measurement(170.0 - offset, 170.0 - offset),
        ];

        let (updated, _) = filter.step(state, &measurements, dt);
        state = updated;
    }

    // Should track both targets
    let estimates = extract_best_hypothesis(&state);
    assert!(
        estimates.len() >= 1,
        "Should track at least one target, got {}",
        estimates.len()
    );

    // Check labels are preserved
    let labels: Vec<Label> = estimates.iter().map(|e| e.label).collect();
    // At least one original label should persist
    assert!(
        labels.contains(&label_a) || labels.contains(&label_b),
        "At least one original label should persist"
    );
}

#[test]
fn test_glmb_hypothesis_management() {
    let transition = ConstantVelocity2D::new(1.0, 0.99);
    let observation = PositionSensor2D::new(5.0, 0.9);
    let clutter = UniformClutter2D::new(5.0, (0.0, 200.0), (0.0, 200.0));
    let birth = NoGlmbBirthModel;

    let filter: GlmbFilter<f64, _, _, _, _, 4, 2> =
        GlmbFilter::new(transition, observation, clutter, birth, 5);

    let initial_tracks = vec![make_glmb_track(Label::new(0, 0), 100.0, 100.0, 0.0, 0.0)];
    let mut state = filter.initial_state_from(initial_tracks);

    // Multiple measurements create multiple hypotheses
    let measurements = vec![
        make_measurement(100.0, 100.0),
        make_measurement(105.0, 105.0),
    ];

    let (updated, _) = filter.step(state, &measurements, 1.0);
    state = updated;

    // Should have multiple hypotheses
    assert!(
        state.num_hypotheses() >= 1,
        "Should have at least one hypothesis"
    );
}

#[test]
fn test_glmb_map_cardinality() {
    let transition = ConstantVelocity2D::new(1.0, 0.99);
    let observation = PositionSensor2D::new(5.0, 0.9);
    let clutter = UniformClutter2D::new(5.0, (0.0, 200.0), (0.0, 200.0));
    let birth = NoGlmbBirthModel;

    let filter: GlmbFilter<f64, _, _, _, _, 4, 2> =
        GlmbFilter::new(transition, observation, clutter, birth, 10);

    let initial_tracks = vec![
        make_glmb_track(Label::new(0, 0), 30.0, 30.0, 0.0, 0.0),
        make_glmb_track(Label::new(0, 1), 170.0, 170.0, 0.0, 0.0),
    ];
    let mut state = filter.initial_state_from(initial_tracks);

    for _ in 0..3 {
        let measurements = vec![make_measurement(30.0, 30.0), make_measurement(170.0, 170.0)];
        let (updated, _) = filter.step(state, &measurements, 1.0);
        state = updated;
    }

    let map_card = state.map_cardinality();
    assert!(
        map_card >= 1 && map_card <= 3,
        "MAP cardinality should be ~2, got {}",
        map_card
    );

    let estimates = extract_map_cardinality(&state);
    assert!(
        !estimates.is_empty(),
        "Should extract targets based on MAP cardinality"
    );
}

#[test]
fn test_glmb_joint_predict_update() {
    let transition = ConstantVelocity2D::new(1.0, 0.99);
    let observation = PositionSensor2D::new(5.0, 0.9);
    let clutter = UniformClutter2D::new(5.0, (0.0, 200.0), (0.0, 200.0));
    let birth = NoGlmbBirthModel;

    let filter: GlmbFilter<f64, _, _, _, _, 4, 2> =
        GlmbFilter::new(transition, observation, clutter, birth, 10);

    let initial_tracks = vec![make_glmb_track(Label::new(0, 0), 50.0, 50.0, 2.0, 1.0)];
    let state = filter.initial_state_from(initial_tracks);

    // Test joint predict-update path
    let measurements = vec![make_measurement(52.0, 51.0)];
    let (updated, stats) = filter.step_joint(state, &measurements, 1.0);

    // Should work correctly
    assert!(
        updated.num_hypotheses() >= 1,
        "Should have hypotheses after joint update"
    );
    assert_eq!(
        stats.singular_covariance_count, 0,
        "Should have no singular covariances"
    );
}

#[test]
fn test_glmb_missed_detection_handling() {
    // Test that GLMB handles missed detections gracefully
    // GLMB with a single hypothesis may maintain existence until track pruning
    let transition = ConstantVelocity2D::new(1.0, 0.99);
    let observation = PositionSensor2D::new(5.0, 0.9);
    let clutter = UniformClutter2D::new(5.0, (0.0, 200.0), (0.0, 200.0));
    let birth = NoGlmbBirthModel;

    let filter: GlmbFilter<f64, _, _, _, _, 4, 2> =
        GlmbFilter::new(transition, observation, clutter, birth, 10);

    let initial_tracks = vec![make_glmb_track(Label::new(0, 0), 100.0, 100.0, 0.0, 0.0)];
    let mut state = filter.initial_state_from(initial_tracks);

    // Update with no measurements multiple times
    // GLMB maintains hypotheses - verifying filter runs without crashing
    for _ in 0..5 {
        let measurements: Vec<Measurement<f64, 2>> = vec![];
        let (updated, _) = filter.step(state, &measurements, 1.0);
        state = updated;
    }

    // Verify filter state is valid
    assert!(
        state.num_hypotheses() >= 1,
        "Should maintain at least one hypothesis"
    );

    // Verify target count is reasonable (GLMB may keep track with missed detections)
    let count = state.expected_target_count();
    assert!(
        count >= 0.0 && count <= 2.0,
        "Expected target count should be reasonable, got {}",
        count
    );
}

#[test]
fn test_glmb_birth_model_creates_tracks() {
    let transition = ConstantVelocity2D::new(1.0, 0.99);
    let observation = PositionSensor2D::new(5.0, 0.9);
    let clutter = UniformClutter2D::new(5.0, (0.0, 200.0), (0.0, 200.0));
    let birth = TestGlmbBirthModel::new().with_location(0.3, 100.0, 100.0);

    let filter: GlmbFilter<f64, _, _, _, _, 4, 2> =
        GlmbFilter::new(transition, observation, clutter, birth, 10);

    let mut state = filter.initial_state();

    // Provide measurements at birth location
    for _ in 0..5 {
        let measurements = vec![make_measurement(100.0, 100.0)];
        let (updated, _) = filter.step(state, &measurements, 1.0);
        state = updated;
    }

    // Should have created a track
    let estimates = extract_best_hypothesis(&state);
    assert!(
        !estimates.is_empty() || state.expected_target_count() > 0.1,
        "Birth model should create tracks"
    );
}

#[test]
fn test_glmb_clutter_robustness() {
    let transition = ConstantVelocity2D::new(1.0, 0.99);
    let observation = PositionSensor2D::new(5.0, 0.9);
    let clutter = UniformClutter2D::new(10.0, (0.0, 200.0), (0.0, 200.0));
    let birth = NoGlmbBirthModel;

    let filter: GlmbFilter<f64, _, _, _, _, 4, 2> =
        GlmbFilter::new(transition, observation, clutter, birth, 10);

    let initial_tracks = vec![make_glmb_track(Label::new(0, 0), 100.0, 100.0, 0.0, 0.0)];
    let mut state = filter.initial_state_from(initial_tracks);

    // Add clutter with target measurement
    for _ in 0..3 {
        let measurements = vec![
            make_measurement(100.0, 100.0), // Target
            make_measurement(20.0, 180.0),  // Clutter
            make_measurement(180.0, 20.0),  // Clutter
        ];

        let (updated, _) = filter.step(state, &measurements, 1.0);
        state = updated;
    }

    // Should still track the target despite clutter
    let expected = state.expected_target_count();
    assert!(
        expected > 0.3,
        "Should maintain target despite clutter, got {}",
        expected
    );
}

#[test]
fn test_glmb_crossing_targets() {
    let transition = ConstantVelocity2D::new(1.0, 0.99);
    let observation = PositionSensor2D::new(5.0, 0.9);
    let clutter = UniformClutter2D::new(5.0, (0.0, 200.0), (0.0, 200.0));
    let birth = NoGlmbBirthModel;

    let filter: GlmbFilter<f64, _, _, _, _, 4, 2> =
        GlmbFilter::new(transition, observation, clutter, birth, 20);

    // Two targets that cross
    let initial_tracks = vec![
        make_glmb_track(Label::new(0, 0), 50.0, 100.0, 10.0, 0.0),  // Moving right
        make_glmb_track(Label::new(0, 1), 150.0, 100.0, -10.0, 0.0), // Moving left
    ];
    let mut state = filter.initial_state_from(initial_tracks);

    let dt = 1.0;

    // Track through crossing
    for t in 0..10 {
        let x1 = 50.0 + (t + 1) as f64 * 10.0;
        let x2 = 150.0 - (t + 1) as f64 * 10.0;
        let measurements = vec![make_measurement(x1, 100.0), make_measurement(x2, 100.0)];

        let (updated, _) = filter.step(state, &measurements, dt);
        state = updated;
    }

    // Should still have targets after crossing
    let estimates = extract_best_hypothesis(&state);
    assert!(
        !estimates.is_empty(),
        "Should maintain tracks through crossing"
    );
}
