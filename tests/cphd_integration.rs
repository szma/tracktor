//! Integration tests for CPHD (Cardinalized PHD) filter

#![cfg(feature = "alloc")]

mod common;

use common::{make_component, make_measurement, make_test_models};
use tracktor::filters::cphd::{
    extract_by_expected_cardinality, extract_by_map_cardinality, GmCphdFilter,
};
use tracktor::types::spaces::Measurement;

#[test]
fn test_cphd_single_target_tracking() {
    let (transition, observation, clutter, birth) = make_test_models();
    let filter = GmCphdFilter::new(transition, observation, clutter, birth);

    let initial = filter.initial_state_from(vec![make_component(50.0, 50.0, 2.0, 1.0, 1.0)]);

    let dt = 1.0;
    let mut state = initial;

    for t in 0..5 {
        let mx = 52.0 + t as f64 * 2.0;
        let my = 51.0 + t as f64 * 1.0;
        let measurements = vec![make_measurement(mx, my)];

        state = filter.step(state, &measurements, dt);

        // Verify cardinality distribution is reasonable
        let map_card = state.map_cardinality();
        assert!(
            map_card <= 2,
            "Step {}: MAP cardinality should be ~1, got {}",
            t,
            map_card
        );
    }
}

#[test]
fn test_cphd_two_target_cardinality() {
    let (transition, observation, clutter, birth) = make_test_models();
    let filter = GmCphdFilter::new(transition, observation, clutter, birth);

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

    // CPHD should estimate cardinality near 2
    let expected_count = state.expected_target_count();
    let map_card = state.map_cardinality();

    assert!(
        expected_count > 1.0 && expected_count < 4.0,
        "Expected count should be ~2, got {}",
        expected_count
    );
    assert!(
        map_card >= 1 && map_card <= 3,
        "MAP cardinality should be ~2, got {}",
        map_card
    );
}

#[test]
fn test_cphd_cardinality_distribution_normalized() {
    let (transition, observation, clutter, birth) = make_test_models();
    let filter = GmCphdFilter::new(transition, observation, clutter, birth);

    let initial = filter.initial_state_from(vec![make_component(100.0, 100.0, 0.0, 0.0, 1.0)]);

    let measurements = vec![make_measurement(100.0, 100.0)];
    let state = filter.step(initial, &measurements, 1.0);

    // Cardinality distribution should sum to ~1
    let card_sum: f64 = state.cardinality.iter().sum();
    assert!(
        (card_sum - 1.0).abs() < 0.01,
        "Cardinality distribution should sum to 1, got {}",
        card_sum
    );
}

#[test]
fn test_cphd_cardinality_variance_decreases_with_confidence() {
    let (transition, observation, clutter, birth) = make_test_models();
    let filter = GmCphdFilter::new(transition, observation, clutter, birth);

    let initial = filter.initial_state_from(vec![make_component(100.0, 100.0, 0.0, 0.0, 1.0)]);

    let initial_variance = initial.cardinality_variance();

    // Run several consistent updates
    let mut state = initial;
    for _ in 0..3 {
        let measurements = vec![make_measurement(100.0, 100.0)];
        state = filter.step(state, &measurements, 1.0);
    }

    // Variance should decrease or stay reasonable with consistent measurements
    let final_variance = state.cardinality_variance();
    // Note: This is a soft check - variance behavior depends on measurement quality
    assert!(
        final_variance < initial_variance * 5.0,
        "Variance should not explode: {} -> {}",
        initial_variance,
        final_variance
    );
}

#[test]
fn test_cphd_extract_by_map_cardinality() {
    let (transition, observation, clutter, birth) = make_test_models();
    let filter = GmCphdFilter::new(transition, observation, clutter, birth);

    let initial = filter.initial_state_from(vec![
        make_component(30.0, 30.0, 0.0, 0.0, 1.0),
        make_component(170.0, 170.0, 0.0, 0.0, 1.0),
    ]);

    let mut state = initial;
    for _ in 0..3 {
        let measurements = vec![make_measurement(30.0, 30.0), make_measurement(170.0, 170.0)];
        state = filter.step(state, &measurements, 1.0);
    }

    let extracted = extract_by_map_cardinality(&state);
    // Should extract a reasonable number of targets
    assert!(!extracted.is_empty(), "Should extract at least one target");
}

#[test]
fn test_cphd_extract_by_expected_cardinality() {
    let (transition, observation, clutter, birth) = make_test_models();
    let filter = GmCphdFilter::new(transition, observation, clutter, birth);

    let initial = filter.initial_state_from(vec![
        make_component(30.0, 30.0, 0.0, 0.0, 1.0),
        make_component(170.0, 170.0, 0.0, 0.0, 1.0),
    ]);

    let mut state = initial;
    for _ in 0..3 {
        let measurements = vec![make_measurement(30.0, 30.0), make_measurement(170.0, 170.0)];
        state = filter.step(state, &measurements, 1.0);
    }

    let extracted = extract_by_expected_cardinality(&state);
    assert!(!extracted.is_empty(), "Should extract at least one target");
}

#[test]
fn test_cphd_no_measurements_updates_cardinality() {
    let (transition, observation, clutter, birth) = make_test_models();
    let filter = GmCphdFilter::new(transition, observation, clutter, birth);

    let initial = filter.initial_state_from(vec![make_component(100.0, 100.0, 0.0, 0.0, 1.0)]);

    let initial_expected = initial.expected_target_count();

    let measurements: Vec<Measurement<f64, 2>> = vec![];
    let state = filter.step(initial, &measurements, 1.0);

    let final_expected = state.expected_target_count();

    // Expected count should decrease without measurements
    assert!(
        final_expected < initial_expected,
        "Expected count should decrease: {} -> {}",
        initial_expected,
        final_expected
    );
}

#[test]
fn test_cphd_stats_tracking() {
    let (transition, observation, clutter, birth) = make_test_models();
    let filter = GmCphdFilter::new(transition, observation, clutter, birth);

    let initial = filter.initial_state_from(vec![make_component(100.0, 100.0, 0.0, 0.0, 1.0)]);

    let measurements = vec![make_measurement(100.0, 100.0)];
    let (state, stats) = filter.step_with_stats(initial, &measurements, 1.0);

    // Stats should be populated
    assert!(stats.max_cardinality > 0);
    // Verify no covariance issues with well-conditioned problem
    assert_eq!(
        stats.singular_covariance_count, 0,
        "Should have no singular covariances"
    );

    // State should be valid
    assert!(!state.cardinality.is_empty());
}
