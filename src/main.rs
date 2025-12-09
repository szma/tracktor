//! Example usage of the Tracktor library
//!
//! Demonstrates a simple multi-target tracking scenario using the GM-PHD filter.

use tracktor::filters::phd::GmPhdFilter;
use tracktor::prelude::*;
use tracktor::utils::{extract_targets, prune_and_merge, ExtractionConfig, PruningConfig};

fn main() {
    println!("Tracktor: Multi-Target Tracking Library");
    println!("======================================\n");

    // Create models
    let transition = ConstantVelocity2D::new(
        1.0,  // Process noise (acceleration std)
        0.99, // Survival probability
    );

    let observation = PositionSensor2D::new(
        5.0, // Measurement noise std
        0.9, // Detection probability
    );

    let clutter = UniformClutter2D::new(
        5.0,          // Clutter rate (expected false alarms per scan)
        (0.0, 200.0), // X bounds
        (0.0, 200.0), // Y bounds
    );

    // Fixed birth model with entry points
    let mut birth = FixedBirthModel::<f64, 4>::new();
    birth.add_birth_location(
        0.05, // Birth weight
        StateVector::from_array([10.0, 10.0, 0.0, 0.0]),
        StateCovariance::from_matrix(nalgebra::matrix![
            100.0, 0.0, 0.0, 0.0;
            0.0, 100.0, 0.0, 0.0;
            0.0, 0.0, 25.0, 0.0;
            0.0, 0.0, 0.0, 25.0
        ]),
    );
    birth.add_birth_location(
        0.05,
        StateVector::from_array([190.0, 190.0, 0.0, 0.0]),
        StateCovariance::from_matrix(nalgebra::matrix![
            100.0, 0.0, 0.0, 0.0;
            0.0, 100.0, 0.0, 0.0;
            0.0, 0.0, 25.0, 0.0;
            0.0, 0.0, 0.0, 25.0
        ]),
    );

    // Create filter
    let filter = GmPhdFilter::new(transition, observation, clutter, birth);

    // Initialize with two known targets
    let initial_components = vec![
        GaussianState::new(
            1.0,
            StateVector::from_array([50.0, 50.0, 2.0, 1.0]),
            StateCovariance::from_matrix(nalgebra::matrix![
                10.0, 0.0, 0.0, 0.0;
                0.0, 10.0, 0.0, 0.0;
                0.0, 0.0, 5.0, 0.0;
                0.0, 0.0, 0.0, 5.0
            ]),
        ),
        GaussianState::new(
            1.0,
            StateVector::from_array([100.0, 100.0, -1.0, 2.0]),
            StateCovariance::from_matrix(nalgebra::matrix![
                10.0, 0.0, 0.0, 0.0;
                0.0, 10.0, 0.0, 0.0;
                0.0, 0.0, 5.0, 0.0;
                0.0, 0.0, 0.0, 5.0
            ]),
        ),
    ];

    let mut state = filter.initial_state_from(initial_components);

    println!(
        "Initial state: {} components, {:.2} expected targets\n",
        state.mixture.len(),
        state.expected_target_count()
    );

    // Simulate measurements for 5 time steps
    let measurements_per_step = [
        // Time 0: Detections near both targets + clutter
        vec![[52.0, 51.0], [99.0, 102.0], [150.0, 30.0]],
        // Time 1: One detection, one miss
        vec![[54.0, 52.0], [45.0, 180.0]],
        // Time 2: Both detected
        vec![[56.0, 53.0], [97.0, 106.0]],
        // Time 3: Both detected + clutter
        vec![[58.0, 54.0], [95.0, 110.0], [20.0, 20.0], [180.0, 50.0]],
        // Time 4: Both detected
        vec![[60.0, 55.0], [93.0, 114.0]],
    ];

    let dt = 1.0;
    let pruning_config = PruningConfig::new(1e-4, 4.0, 50);
    let extraction_config = ExtractionConfig::weight_threshold(0.5);

    for (t, meas_data) in measurements_per_step.iter().enumerate() {
        // Convert measurements
        let measurements: Vec<Measurement<f64, 2>> = meas_data
            .iter()
            .map(|m| Measurement::from_array(*m))
            .collect();

        println!("Time step {}: {} measurements", t, measurements.len());

        // Predict
        let predicted = state.predict(&filter.transition, &filter.birth, dt);
        println!(
            "  After predict: {} components, {:.2} expected targets",
            predicted.mixture.len(),
            predicted.expected_target_count()
        );

        // Update
        let updated = predicted.update(&measurements, &filter.observation, &filter.clutter);
        println!(
            "  After update:  {} components, {:.2} expected targets",
            updated.mixture.len(),
            updated.expected_target_count()
        );

        // Prune and merge
        let pruned = prune_and_merge(&updated.mixture, &pruning_config);
        state = tracktor::filters::phd::PhdFilterState::from_mixture(pruned);
        println!(
            "  After prune:   {} components, {:.2} expected targets",
            state.mixture.len(),
            state.expected_target_count()
        );

        // Extract targets
        let targets = extract_targets(&state.mixture, &extraction_config);
        println!("  Extracted targets:");
        for (i, target) in targets.iter().enumerate() {
            println!(
                "    Target {}: pos=({:.1}, {:.1}), vel=({:.1}, {:.1}), conf={:.2}",
                i,
                target.state.index(0),
                target.state.index(1),
                target.state.index(2),
                target.state.index(3),
                target.confidence
            );
        }
        println!();
    }

    println!("Tracking complete!");

    // Demonstrate type safety (these would not compile):
    // let state_vec: StateVector<f64, 4> = StateVector::from_array([1.0, 2.0, 3.0, 4.0]);
    // let meas_vec: Measurement<f64, 2> = Measurement::from_array([5.0, 6.0]);
    // let invalid = state_vec + meas_vec;  // ERROR: cannot add different spaces

    // let updated_state: PhdFilterState<f64, 4, Updated> = ...;
    // let double_update = updated_state.update(&measurements);  // ERROR: update not available on Updated
}
