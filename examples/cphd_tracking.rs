//! CPHD (Cardinalized PHD) Filter Tracking Example
//!
//! This example demonstrates the GM-CPHD filter, which provides improved
//! cardinality (target count) estimation compared to the standard PHD filter.
//!
//! The CPHD filter maintains an explicit cardinality distribution alongside
//! the intensity function, leading to more reliable target count estimates.
//!
//! Reference: Vo, B.-T., Vo, B.-N., & Cantoni, A. (2007). "Analytic Implementations
//! of the Cardinalized Probability Hypothesis Density Filter"

use tracktor::filters::cphd::{CphdFilterState, GmCphdFilter, extract_by_map_cardinality};
use tracktor::prelude::*;
use tracktor::types::phase::Updated;
use tracktor::utils::{ExtractionConfig, PruningConfig, extract_targets, prune_and_merge};

fn main() {
    println!("Tracktor: GM-CPHD Filter Example");
    println!("=================================\n");

    // Create models (same as PHD filter)
    let transition = ConstantVelocity2D::new(
        1.0,  // Velocity noise diffusion coefficient
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

    // Create CPHD filter
    let filter = GmCphdFilter::new(transition, observation, clutter, birth);

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
        "Initial state: {} components, {:.2} expected targets",
        state.mixture.len(),
        state.expected_target_count()
    );
    println!(
        "  MAP cardinality: {}, variance: {:.4}\n",
        state.map_cardinality(),
        state.cardinality_variance()
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
        println!(
            "    Cardinality: MAP={}, variance={:.4}",
            predicted.map_cardinality(),
            predicted.cardinality_variance()
        );

        // Update with stats
        let (updated, stats) =
            predicted.update_with_stats(&measurements, &filter.observation, &filter.clutter);
        println!(
            "  After update:  {} components, {:.2} expected targets",
            updated.mixture.len(),
            updated.expected_target_count()
        );
        println!(
            "    Cardinality: MAP={}, variance={:.4}",
            stats.map_cardinality,
            updated.cardinality_variance()
        );

        if stats.has_issues() {
            println!(
                "    Warnings: {} singular, {} zero likelihood",
                stats.singular_covariance_count, stats.zero_likelihood_count
            );
        }

        // Prune and merge
        let pruned = prune_and_merge(&updated.mixture, &pruning_config);
        state = CphdFilterState::<f64, 4, Updated>::from_mixture_and_cardinality(
            pruned,
            updated.cardinality.clone(),
        );
        println!(
            "  After prune:   {} components, {:.2} expected targets",
            state.mixture.len(),
            state.expected_target_count()
        );

        // Extract targets using standard method
        let targets_standard = extract_targets(&state.mixture, &extraction_config);
        println!("  Standard extraction (weight threshold):");
        for (i, target) in targets_standard.iter().enumerate() {
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

        // Extract targets using CPHD-specific MAP cardinality method
        let targets_map = extract_by_map_cardinality(&state);
        println!(
            "  CPHD extraction (MAP cardinality={}):",
            state.map_cardinality()
        );
        for (i, (target_state, weight)) in targets_map.iter().enumerate() {
            println!(
                "    Target {}: pos=({:.1}, {:.1}), vel=({:.1}, {:.1}), weight={:.2}",
                i,
                target_state.index(0),
                target_state.index(1),
                target_state.index(2),
                target_state.index(3),
                weight
            );
        }

        // Show cardinality distribution
        print!("  Cardinality distribution: [");
        for (n, &rho) in state.cardinality.iter().enumerate().take(6) {
            if n > 0 {
                print!(", ");
            }
            print!("P({})={:.3}", n, rho);
        }
        if state.cardinality.len() > 6 {
            print!(", ...");
        }
        println!("]");

        println!();
    }
}
