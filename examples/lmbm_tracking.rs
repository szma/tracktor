//! LMBM (Labeled Multi-Bernoulli Multi-hypothesis) Tracking Example
//!
//! Demonstrates multi-hypothesis tracking where the filter maintains
//! multiple data association hypotheses weighted by probability.

use tracktor::filters::lmb::{
    LabeledBirthModel, LmbmFilter, extract_best_hypothesis, extract_lmbm_estimates,
};
use tracktor::prelude::*;

// Birth model for LMBM
struct LmbmBirthModel;

impl LabeledBirthModel<f64, 4> for LmbmBirthModel {
    fn birth_tracks(&self, label_gen: &mut LabelGenerator) -> Vec<BernoulliTrack<f64, 4>> {
        let cov = StateCovariance::from_matrix(nalgebra::matrix![
            150.0, 0.0, 0.0, 0.0;
            0.0, 150.0, 0.0, 0.0;
            0.0, 0.0, 30.0, 0.0;
            0.0, 0.0, 0.0, 30.0
        ]);

        vec![BernoulliTrack::new(
            label_gen.next_label(),
            0.01,
            GaussianState::new(1.0, StateVector::from_array([50.0, 50.0, 0.0, 0.0]), cov),
        )]
    }

    fn expected_birth_count(&self) -> f64 {
        0.01
    }
}

fn main() {
    println!("LMBM Filter: Multi-Hypothesis Tracking");
    println!("======================================\n");

    // Create models
    let transition = ConstantVelocity2D::new(1.5, 0.98);
    let observation = PositionSensor2D::new(6.0, 0.88);
    let clutter = UniformClutter2D::new(4.0, (0.0, 150.0), (0.0, 150.0));
    let birth = LmbmBirthModel;

    // Create LMBM filter with:
    // - k_best = 5: Consider top 5 assignments per hypothesis
    // - max_hypotheses = 20: Keep at most 20 hypotheses
    let filter: LmbmFilter<f64, _, _, _, _, 4, 2> =
        LmbmFilter::new(transition, observation, clutter, birth, 5, 20);

    let mut state = filter.initial_state();

    println!("Initial: {} hypotheses\n", state.num_hypotheses());

    // Measurements with ambiguous association
    // Two targets close together create association uncertainty
    let measurements_per_step = [
        vec![[30.0, 30.0], [35.0, 32.0], [100.0, 80.0]], // Two close + clutter
        vec![[32.0, 32.0], [37.0, 34.0]],                // Two close targets
        vec![[34.0, 33.0], [40.0, 37.0], [20.0, 100.0]], // Two close + clutter
        vec![[35.0, 35.0], [43.0, 39.0]],                // Separating
        vec![[37.0, 36.0], [47.0, 42.0]],                // More separated
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

        println!(
            "  Hypotheses: {}, Singular: {}",
            state.num_hypotheses(),
            stats.singular_covariance_count
        );

        // Show best hypothesis
        if let Some(best) = extract_best_hypothesis(&state) {
            println!(
                "  Best hypothesis: {} tracks, log_w={:.2}",
                best.num_tracks(),
                best.log_weight
            );
            for track in &best.tracks {
                println!(
                    "    {:?}: ({:.1}, {:.1}), r={:.3}",
                    track.label,
                    track.state.mean.index(0),
                    track.state.mean.index(1),
                    track.existence
                );
            }
        }

        // Show marginal estimates
        let marginals = extract_lmbm_estimates(&state, 0.3);
        if !marginals.is_empty() {
            println!("  Marginal estimates (r > 0.3):");
            for (label, mean, existence) in &marginals {
                println!(
                    "    {:?}: ({:.1}, {:.1}), r={:.3}",
                    label,
                    mean.index(0),
                    mean.index(1),
                    existence
                );
            }
        }
        println!();
    }
}
