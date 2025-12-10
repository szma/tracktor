//! Multi-target scenario tests comparing behavior across filter types

#![cfg(feature = "alloc")]

mod common;

use common::{make_component, make_measurement, make_test_models, make_test_models_with_birth};
use tracktor::filters::cphd::GmCphdFilter;
use tracktor::filters::phd::GmPhdFilter;
use tracktor::models::ConstantVelocity2D;
use tracktor::types::spaces::Measurement;

#[test]
fn test_five_target_phd() {
    let (transition, observation, clutter, birth) = make_test_models();
    let filter = GmPhdFilter::new(transition, observation, clutter, birth);

    let initial = filter.initial_state_from(vec![
        make_component(20.0, 20.0, 1.0, 1.0, 1.0),
        make_component(180.0, 20.0, -1.0, 1.0, 1.0),
        make_component(20.0, 180.0, 1.0, -1.0, 1.0),
        make_component(180.0, 180.0, -1.0, -1.0, 1.0),
        make_component(100.0, 100.0, 0.0, 0.0, 1.0),
    ]);

    let dt = 1.0;
    let mut state = initial;

    for t in 0..5 {
        let offset = (t + 1) as f64;
        let measurements = vec![
            make_measurement(20.0 + offset, 20.0 + offset),
            make_measurement(180.0 - offset, 20.0 + offset),
            make_measurement(20.0 + offset, 180.0 - offset),
            make_measurement(180.0 - offset, 180.0 - offset),
            make_measurement(100.0, 100.0),
        ];

        state = filter.step(state, &measurements, dt);
    }

    // Should track approximately 5 targets
    let count = state.expected_target_count();
    assert!(
        count > 2.0 && count < 8.0,
        "Expected ~5 targets, got {}",
        count
    );
}

#[test]
fn test_five_target_cphd() {
    let (transition, observation, clutter, birth) = make_test_models();
    let filter = GmCphdFilter::new(transition, observation, clutter, birth);

    let initial = filter.initial_state_from(vec![
        make_component(20.0, 20.0, 1.0, 1.0, 1.0),
        make_component(180.0, 20.0, -1.0, 1.0, 1.0),
        make_component(20.0, 180.0, 1.0, -1.0, 1.0),
        make_component(180.0, 180.0, -1.0, -1.0, 1.0),
        make_component(100.0, 100.0, 0.0, 0.0, 1.0),
    ]);

    let dt = 1.0;
    let mut state = initial;

    for t in 0..5 {
        let offset = (t + 1) as f64;
        let measurements = vec![
            make_measurement(20.0 + offset, 20.0 + offset),
            make_measurement(180.0 - offset, 20.0 + offset),
            make_measurement(20.0 + offset, 180.0 - offset),
            make_measurement(180.0 - offset, 180.0 - offset),
            make_measurement(100.0, 100.0),
        ];

        state = filter.step(state, &measurements, dt);
    }

    // CPHD should have good cardinality estimate
    let map_card = state.map_cardinality();
    assert!(
        map_card >= 2 && map_card <= 8,
        "MAP cardinality should be ~5, got {}",
        map_card
    );
}

#[test]
fn test_target_appearance_disappearance_phd() {
    let (_, observation, clutter, birth) = make_test_models_with_birth(&[(100.0, 100.0)]);

    // Lower survival probability
    let transition = ConstantVelocity2D::new(1.0, 0.95);
    let filter = GmPhdFilter::new(transition, observation, clutter, birth);

    let mut state = filter.initial_state();

    // Target appears (birth)
    for _ in 0..5 {
        let measurements = vec![make_measurement(100.0, 100.0)];
        state = filter.step(state, &measurements, 1.0);
    }

    let count_during = state.expected_target_count();

    // Target disappears (no measurements)
    for _ in 0..10 {
        let measurements: Vec<Measurement<f64, 2>> = vec![];
        state = filter.step(state, &measurements, 1.0);
    }

    let count_after = state.expected_target_count();

    assert!(
        count_after < count_during,
        "Count should decrease when target disappears: {} -> {}",
        count_during,
        count_after
    );
}
