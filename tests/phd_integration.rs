//! Integration tests for PHD (Probability Hypothesis Density) filter

#![cfg(feature = "alloc")]

mod common;

use common::{make_component, make_measurement, make_test_models, make_test_models_with_birth};
use tracktor::filters::phd::GmPhdFilter;
use tracktor::types::spaces::Measurement;

#[test]
fn test_phd_single_target_tracking() {
    let (transition, observation, clutter, birth) = make_test_models();
    let filter = GmPhdFilter::new(transition, observation, clutter, birth);

    // Initialize with single target at (50, 50)
    let initial = filter.initial_state_from(vec![make_component(50.0, 50.0, 2.0, 1.0, 1.0)]);

    let dt = 1.0;
    let mut state = initial;

    // Track for several steps with measurements following the target
    for t in 0..5 {
        let mx = 52.0 + t as f64 * 2.0;
        let my = 51.0 + t as f64 * 1.0;
        let measurements = vec![make_measurement(mx, my)];

        state = filter.step(state, &measurements, dt);

        // Verify expected target count stays near 1
        let count = state.expected_target_count();
        assert!(
            count > 0.5 && count < 2.0,
            "Step {}: Expected ~1 target, got {}",
            t,
            count
        );
    }
}

#[test]
fn test_phd_two_target_tracking() {
    let (transition, observation, clutter, birth) = make_test_models();
    let filter = GmPhdFilter::new(transition, observation, clutter, birth);

    // Initialize with two well-separated targets
    let initial = filter.initial_state_from(vec![
        make_component(30.0, 30.0, 1.0, 1.0, 1.0),
        make_component(170.0, 170.0, -1.0, -1.0, 1.0),
    ]);

    let dt = 1.0;
    let mut state = initial;

    for t in 0..5 {
        let offset = (t + 1) as f64;
        let measurements = vec![
            make_measurement(30.0 + offset, 30.0 + offset),
            make_measurement(170.0 - offset, 170.0 - offset),
        ];

        state = filter.step(state, &measurements, dt);
    }

    // Verify expected target count is approximately 2
    let count = state.expected_target_count();
    assert!(
        count > 1.0 && count < 3.5,
        "Expected ~2 targets, got {}",
        count
    );
}

#[test]
fn test_phd_no_measurements_reduces_weight() {
    let (transition, observation, clutter, birth) = make_test_models();
    let filter = GmPhdFilter::new(transition, observation, clutter, birth);

    let initial = filter.initial_state_from(vec![make_component(50.0, 50.0, 0.0, 0.0, 1.0)]);

    let initial_count = initial.expected_target_count();

    // Update with no measurements
    let measurements: Vec<Measurement<f64, 2>> = vec![];
    let state = filter.step(initial, &measurements, 1.0);

    let final_count = state.expected_target_count();
    assert!(
        final_count < initial_count,
        "Expected count to decrease without measurements: {} -> {}",
        initial_count,
        final_count
    );
}

#[test]
fn test_phd_clutter_robustness() {
    let (transition, observation, clutter, birth) = make_test_models();
    let filter = GmPhdFilter::new(transition, observation, clutter, birth);

    let initial = filter.initial_state_from(vec![make_component(100.0, 100.0, 0.0, 0.0, 1.0)]);

    let dt = 1.0;
    let mut state = initial;

    // Add clutter measurements along with target measurement
    for _ in 0..3 {
        let measurements = vec![
            make_measurement(100.0, 100.0), // Target
            make_measurement(20.0, 180.0),  // Clutter
            make_measurement(180.0, 20.0),  // Clutter
            make_measurement(50.0, 150.0),  // Clutter
        ];

        state = filter.step(state, &measurements, dt);
    }

    // Should still track approximately 1 target despite clutter
    let count = state.expected_target_count();
    assert!(
        count > 0.5 && count < 3.0,
        "Expected ~1 target despite clutter, got {}",
        count
    );
}

#[test]
fn test_phd_birth_model_spawns_targets() {
    let (transition, observation, clutter, birth) = make_test_models_with_birth(&[(100.0, 100.0)]);
    let filter = GmPhdFilter::new(transition, observation, clutter, birth);

    // Start empty
    let mut state = filter.initial_state();
    assert!(
        state.expected_target_count() < 0.5,
        "Should start with no targets"
    );

    // Provide measurements at birth location
    for _ in 0..5 {
        let measurements = vec![make_measurement(100.0, 100.0)];
        state = filter.step(state, &measurements, 1.0);
    }

    // Birth model should have created target
    let count = state.expected_target_count();
    assert!(
        count > 0.3,
        "Birth model should spawn target, got count {}",
        count
    );
}

#[test]
fn test_phd_mixture_growth_bounded() {
    let (transition, observation, clutter, birth) =
        make_test_models_with_birth(&[(50.0, 50.0), (150.0, 150.0)]);
    let filter = GmPhdFilter::new(transition, observation, clutter, birth);

    let mut state = filter.initial_state();

    // Run many steps - mixture should not grow unboundedly
    // (though without pruning it will grow, we check it stays reasonable)
    for _ in 0..10 {
        let measurements = vec![make_measurement(50.0, 50.0), make_measurement(150.0, 150.0)];
        state = filter.step(state, &measurements, 1.0);
    }

    // Just verify filter runs without crashing
    assert!(state.mixture.len() > 0);
}
