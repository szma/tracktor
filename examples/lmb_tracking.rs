//! Basic LMB (Labeled Multi-Bernoulli) Tracking Example
//!
//! Demonstrates single-sensor multi-target tracking with track identity
//! preservation using the LMB filter.

use tracktor::filters::lmb::{extract_lmb_estimates, LabeledBirthModel, LmbFilter, LmbTrack};
use tracktor::prelude::*;

// Labeled birth model implementation
struct SimpleBirthModel {
    birth_locations: Vec<(f64, StateVector<f64, 4>, StateCovariance<f64, 4>)>,
}

impl SimpleBirthModel {
    fn new() -> Self {
        let cov = StateCovariance::from_matrix(nalgebra::matrix![
            100.0, 0.0, 0.0, 0.0;
            0.0, 100.0, 0.0, 0.0;
            0.0, 0.0, 25.0, 0.0;
            0.0, 0.0, 0.0, 25.0
        ]);

        Self {
            birth_locations: vec![
                (0.03, StateVector::from_array([10.0, 10.0, 0.0, 0.0]), cov),
                (0.03, StateVector::from_array([190.0, 190.0, 0.0, 0.0]), cov),
            ],
        }
    }
}

impl LabeledBirthModel<f64, 4> for SimpleBirthModel {
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

fn main() {
    println!("LMB Filter: Labeled Multi-Target Tracking");
    println!("==========================================\n");

    // Create models
    let transition = ConstantVelocity2D::new(1.0, 0.99);
    let observation = PositionSensor2D::new(5.0, 0.9);
    let clutter = UniformClutter2D::new(5.0, (0.0, 200.0), (0.0, 200.0));
    let birth = SimpleBirthModel::new();

    // Create LMB filter
    let filter: LmbFilter<f64, _, _, _, _, 4, 2> =
        LmbFilter::new(transition, observation, clutter, birth);

    // Initialize with known targets
    let initial_tracks = vec![
        LmbTrack::new(
            Label::new(0, 0),
            0.95,
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
        ),
        LmbTrack::new(
            Label::new(0, 1),
            0.95,
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
        ),
    ];

    let mut state = filter.initial_state_from(initial_tracks);

    println!(
        "Initial: {} tracks, {:.2} expected targets\n",
        state.num_tracks(),
        state.expected_target_count()
    );

    // Simulated measurements over time
    let measurements_per_step = [
        vec![[52.0, 51.0], [99.0, 102.0], [150.0, 30.0]], // Both + clutter
        vec![[54.0, 52.0], [45.0, 180.0]],                // One detected + clutter
        vec![[56.0, 53.0], [97.0, 106.0]],                // Both detected
        vec![[58.0, 54.0], [95.0, 110.0], [20.0, 20.0]],  // Both + clutter
        vec![[60.0, 55.0], [93.0, 114.0]],                // Both detected
    ];

    let dt = 1.0;

    for (t, meas_data) in measurements_per_step.iter().enumerate() {
        let measurements: Vec<Measurement<f64, 2>> = meas_data
            .iter()
            .map(|m| Measurement::from_array(*m))
            .collect();

        println!("Time {}: {} measurements", t, measurements.len());

        // Run filter step
        let (updated, stats) = filter.step(state, &measurements, dt);
        state = updated;

        // Prune low-existence tracks
        state.tracks.prune_by_existence(0.01);

        println!(
            "  Tracks: {}, Expected: {:.2}",
            state.num_tracks(),
            state.expected_target_count()
        );

        if stats.singular_covariance_count > 0 {
            println!(
                "  Warning: {} singular covariances",
                stats.singular_covariance_count
            );
        }

        // Extract and display estimates
        let estimates = extract_lmb_estimates(&state);
        for est in &estimates {
            println!(
                "  Track {:?}: pos=({:.1}, {:.1}), r={:.3}",
                est.label,
                est.state.index(0),
                est.state.index(1),
                est.existence
            );
        }
        println!();
    }

    println!("LMB tracking complete!");
}
