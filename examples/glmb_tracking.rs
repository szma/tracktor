//! GLMB (Generalized Labeled Multi-Bernoulli) Tracking Example
//!
//! Demonstrates multi-target tracking using the GLMB filter with both
//! the standard `step()` method and the fast `step_joint()` method.
//!
//! The GLMB filter maintains a mixture of hypotheses, each representing
//! a specific combination of track existence and measurement association.
//! This enables representing correlations between track existences that
//! cannot be captured by simpler filters like LMB.

use tracktor::filters::glmb::{
    GlmbFilter, GlmbTruncationConfig, extract_best_hypothesis, extract_marginal_states,
};
use tracktor::filters::lmb::{LabeledBirthModel, NoBirthModel};
use tracktor::prelude::*;

/// Birth model that generates potential new tracks at specified locations
struct GlmbBirthModel {
    /// Birth locations with (existence_probability, x, y)
    locations: Vec<(f64, f64, f64)>,
}

impl GlmbBirthModel {
    fn new() -> Self {
        Self {
            // Birth locations covering the surveillance region
            locations: vec![
                (0.02, 20.0, 20.0), // Lower-left region
                (0.02, 80.0, 80.0), // Upper-right region
                (0.02, 50.0, 50.0), // Center
            ],
        }
    }
}

impl LabeledBirthModel<f64, 4> for GlmbBirthModel {
    fn birth_tracks(&self, label_gen: &mut LabelGenerator) -> Vec<BernoulliTrack<f64, 4>> {
        let cov = StateCovariance::from_matrix(nalgebra::matrix![
            100.0, 0.0, 0.0, 0.0;
            0.0, 100.0, 0.0, 0.0;
            0.0, 0.0, 25.0, 0.0;
            0.0, 0.0, 0.0, 25.0
        ]);

        self.locations
            .iter()
            .map(|(existence, x, y)| {
                BernoulliTrack::new(
                    label_gen.next_label(),
                    *existence,
                    GaussianState::new(1.0, StateVector::from_array([*x, *y, 0.0, 0.0]), cov),
                )
            })
            .collect()
    }

    fn expected_birth_count(&self) -> f64 {
        self.locations.iter().map(|(e, _, _)| e).sum()
    }
}

fn main() {
    println!("GLMB Filter: Multi-Target Tracking");
    println!("===================================\n");

    // Example 1: Basic GLMB tracking with step_joint()
    example_basic_tracking();

    println!("\n");

    // Example 2: Comparing step() vs step_joint()
    example_compare_methods();
}

/// Basic GLMB tracking example using the fast joint predict-update
fn example_basic_tracking() {
    println!("Example 1: Basic GLMB Tracking with step_joint()");
    println!("-------------------------------------------------\n");

    // Create models
    let transition = ConstantVelocity2D::new(1.0, 0.99); // process_noise=1.0, survival=0.99
    let observation = PositionSensor2D::new(5.0, 0.95); // meas_noise=5.0, detection=0.95
    let clutter = UniformClutter2D::new(2.0, (0.0, 100.0), (0.0, 100.0)); // 2 false alarms avg
    let birth = GlmbBirthModel::new();

    // Custom truncation configuration
    let truncation = GlmbTruncationConfig {
        log_weight_threshold: -15.0,   // Prune hypotheses with log_weight < -15
        max_hypotheses: 100,           // Keep at most 100 hypotheses
        max_per_cardinality: Some(20), // Keep at most 20 per cardinality
    };

    // Create filter with k_best=10 assignments per hypothesis
    let filter = GlmbFilter::with_truncation(
        transition,
        observation,
        clutter,
        birth,
        truncation,
        10, // k_best
    );

    let mut state = filter.initial_state();
    let dt = 1.0;

    // Simulated scenario: Two targets appearing and moving
    // Target 1: starts at (20, 20), moves with velocity (2, 1)
    // Target 2: starts at (70, 30), moves with velocity (-1, 2)
    let measurements_per_step = [
        // Time 0: Both targets detected + one clutter
        vec![[22.0, 21.0], [68.0, 32.0], [50.0, 90.0]],
        // Time 1: Both detected
        vec![[24.0, 23.0], [67.0, 35.0]],
        // Time 2: Target 1 missed, Target 2 detected + clutter
        vec![[65.0, 38.0], [10.0, 10.0]],
        // Time 3: Both detected
        vec![[28.0, 25.0], [64.0, 41.0]],
        // Time 4: Both detected + clutter
        vec![[30.0, 26.0], [62.0, 44.0], [80.0, 80.0]],
    ];

    println!("Initial state: {} hypotheses\n", state.num_hypotheses());

    for (t, meas_data) in measurements_per_step.iter().enumerate() {
        let measurements: Vec<Measurement<f64, 2>> = meas_data
            .iter()
            .map(|m| Measurement::from_array(*m))
            .collect();

        println!("Time {}: {} measurements", t, measurements.len());

        // Use the fast joint predict-update method
        let (new_state, stats) = filter.step_joint(state, &measurements, dt);
        state = new_state;

        println!(
            "  Hypotheses: {}, Issues: {}",
            state.num_hypotheses(),
            if stats.has_issues() { "yes" } else { "none" }
        );

        // Extract estimates from best hypothesis
        let estimates = extract_best_hypothesis(&state);
        println!("  Best hypothesis tracks: {}", estimates.len());
        for est in &estimates {
            println!(
                "    Label {:?}: position=({:.1}, {:.1})",
                est.label,
                est.state.as_slice()[0],
                est.state.as_slice()[1]
            );
        }

        // Show MAP cardinality
        println!("  MAP cardinality: {}", state.map_cardinality());

        // Show marginal estimates (tracks with existence > 0.5)
        let marginals = extract_marginal_states(&state, 0.5);
        if !marginals.is_empty() {
            println!("  Marginal estimates (r > 0.5): {}", marginals.len());
        }

        println!();
    }
}

/// Compare step() vs step_joint() to show they produce equivalent results
fn example_compare_methods() {
    println!("Example 2: Comparing step() vs step_joint()");
    println!("--------------------------------------------\n");

    // Use simpler setup without birth for clearer comparison
    let transition = ConstantVelocity2D::new(1.0, 0.99);
    let observation = PositionSensor2D::new(5.0, 0.95);
    let clutter = UniformClutter2D::new(1.0, (0.0, 100.0), (0.0, 100.0));
    let birth = NoBirthModel;

    let filter = GlmbFilter::new(transition, observation, clutter, birth, 5);

    // Create initial state with two known tracks
    let label1 = Label::new(0, 0);
    let label2 = Label::new(0, 1);
    let cov: StateCovariance<f64, 4> = StateCovariance::from_matrix(nalgebra::matrix![
        10.0, 0.0, 0.0, 0.0;
        0.0, 10.0, 0.0, 0.0;
        0.0, 0.0, 1.0, 0.0;
        0.0, 0.0, 0.0, 1.0
    ]);

    let track1 = tracktor::filters::glmb::GlmbTrack::new(
        label1,
        GaussianState::new(1.0, StateVector::from_array([20.0, 20.0, 1.0, 0.5]), cov),
    );
    let track2 = tracktor::filters::glmb::GlmbTrack::new(
        label2,
        GaussianState::new(1.0, StateVector::from_array([60.0, 40.0, -0.5, 1.0]), cov),
    );

    // Create two identical initial states (with two tracks each)
    let state1 =
        tracktor::filters::glmb::GlmbFilterState::from_tracks(vec![track1.clone(), track2.clone()]);
    let state2 = tracktor::filters::glmb::GlmbFilterState::from_tracks(vec![track1, track2]);

    // Measurements near expected positions
    let measurements = vec![
        Measurement::from_array([21.5, 21.0]), // Near track 1's predicted position
    ];

    let dt = 1.0;

    // Run both methods
    println!("Running step() (separate predict + update)...");
    let start1 = std::time::Instant::now();
    let (result1, _) = filter.step(state1, &measurements, dt);
    let time1 = start1.elapsed();

    println!("Running step_joint() (fast joint predict-update)...");
    let start2 = std::time::Instant::now();
    let (result2, _) = filter.step_joint(state2, &measurements, dt);
    let time2 = start2.elapsed();

    println!("\nResults:");
    println!(
        "  step():       {} hypotheses, took {:?}",
        result1.num_hypotheses(),
        time1
    );
    println!(
        "  step_joint(): {} hypotheses, took {:?}",
        result2.num_hypotheses(),
        time2
    );

    // Compare best hypothesis estimates
    let est1 = extract_best_hypothesis(&result1);
    let est2 = extract_best_hypothesis(&result2);

    println!("\n  Best hypothesis comparison:");
    println!("    step() tracks:       {}", est1.len());
    println!("    step_joint() tracks: {}", est2.len());

    if !est1.is_empty() && !est2.is_empty() {
        let x1 = est1[0].state.as_slice()[0];
        let x2 = est2[0].state.as_slice()[0];
        let y1 = est1[0].state.as_slice()[1];
        let y2 = est2[0].state.as_slice()[1];

        println!("\n    Position from step():       ({:.2}, {:.2})", x1, y1);
        println!("    Position from step_joint(): ({:.2}, {:.2})", x2, y2);
        println!(
            "    Difference: ({:.4}, {:.4})",
            (x1 - x2).abs(),
            (y1 - y2).abs()
        );
    }

    println!("\nNote: step_joint() is more efficient when there are many hypotheses");
    println!("sharing the same track sets, as it shares computation across them.");
}
