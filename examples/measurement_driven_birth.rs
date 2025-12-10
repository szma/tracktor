//! Measurement-driven birth model example
//!
//! Demonstrates using the MeasurementDrivenBirthModel for scenarios where
//! new targets can appear anywhere in the surveillance region. Instead of
//! predefined birth locations, this model creates birth components from
//! unassociated measurements.

use tracktor::filters::phd::GmPhdFilter;
use tracktor::prelude::*;
use tracktor::utils::{ExtractionConfig, PruningConfig, extract_targets, prune_and_merge};

fn main() {
    println!("Measurement-Driven Birth Model Example");
    println!("======================================\n");

    // Create models
    let transition = ConstantVelocity2D::new(
        1.0,  // Velocity noise diffusion coefficient
        0.98, // Survival probability
    );

    let observation = PositionSensor2D::new(
        10.0, // Measurement noise std
        0.9,  // Detection probability
    );

    let clutter = UniformClutter2D::new(
        3.0,          // Clutter rate (expected false alarms per scan)
        (0.0, 200.0), // X bounds
        (0.0, 200.0), // Y bounds
    );

    // Create measurement-driven birth model using ConstantVelocity2DExpander
    // This expands 2D position measurements [x, y] into 4D CV state [x, y, vx, vy]
    let expander = ConstantVelocity2DExpander::new(
        100.0, // Position covariance (should be larger than sensor noise)
        100.0, // Velocity covariance (large since we don't know initial velocity)
        0.02,  // Birth weight per measurement
    );

    let mut birth = MeasurementDrivenBirthModel::new(expander);

    // Create filter
    let filter = GmPhdFilter::new(transition, observation, clutter.clone(), birth.clone());

    // Start with empty state - no prior knowledge of targets
    let mut state = filter.initial_state();

    println!(
        "Initial state: {} components, {:.2} expected targets\n",
        state.mixture.len(),
        state.expected_target_count()
    );

    // Simulate scenario: targets appearing at different times
    // - Target 1: appears at time 0 at (30, 30), moves right
    // - Target 2: appears at time 2 at (150, 150), moves down-left
    // - Target 3: appears at time 4 at (100, 50), moves up

    let measurements_per_step = [
        // Time 0: First target appears + clutter
        vec![[32.0, 28.0], [180.0, 90.0], [45.0, 160.0]],
        // Time 1: Target 1 detected + clutter
        vec![[37.0, 33.0], [10.0, 10.0]],
        // Time 2: Target 1 + new Target 2 appears
        vec![[42.0, 38.0], [148.0, 152.0]],
        // Time 3: Both targets detected
        vec![[47.0, 43.0], [141.0, 143.0]],
        // Time 4: Both targets + new Target 3 appears
        vec![[52.0, 48.0], [134.0, 134.0], [102.0, 48.0]],
        // Time 5: All three targets
        vec![[57.0, 53.0], [127.0, 125.0], [100.0, 56.0]],
        // Time 6: All three targets + clutter
        vec![[62.0, 58.0], [120.0, 116.0], [98.0, 64.0], [175.0, 175.0]],
        // Time 7: All three targets
        vec![[67.0, 63.0], [113.0, 107.0], [96.0, 72.0]],
    ];

    let dt = 1.0;
    let pruning_config = PruningConfig::new(1e-4, 4.0, 100);
    let extraction_config = ExtractionConfig::weight_threshold(0.5);

    for (t, meas_data) in measurements_per_step.iter().enumerate() {
        // Convert measurements
        let measurements: Vec<Measurement<f64, 2>> = meas_data
            .iter()
            .map(|m| Measurement::from_array(*m))
            .collect();

        println!("Time step {}: {} measurements", t, measurements.len());

        // For measurement-driven birth, we feed measurements to the birth model
        // In practice, you'd typically use unassociated measurements from the previous step
        // For simplicity here, we use all measurements
        birth.set_measurements(measurements.clone());

        // Predict with measurement-driven birth
        let predicted = state.predict(&filter.transition, &birth, dt);
        println!(
            "  After predict: {} components, {:.2} expected targets (birth added {} components)",
            predicted.mixture.len(),
            predicted.expected_target_count(),
            birth.num_adaptive_measurements()
        );

        // Update
        let updated = predicted.update(&measurements, &filter.observation, &clutter);
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
        println!("  Extracted {} targets:", targets.len());
        for (i, target) in targets.iter().enumerate() {
            println!(
                "    Target {}: pos=({:.1}, {:.1}), vel=({:.1}, {:.1}), weight={:.2}",
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
}
