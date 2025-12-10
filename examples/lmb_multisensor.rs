//! Multi-Sensor LMB Tracking Example
//!
//! Demonstrates multi-sensor fusion using different strategies:
//! - AA-LMB: Arithmetic Average
//! - GA-LMB: Geometric Average (Covariance Intersection)
//! - PU-LMB: Parallel Update (Information Filter)
//! - IC-LMB: Iterated Corrector

use tracktor::filters::lmb::{
    LabeledBirthModel, LmbTrack, MultisensorLmbFilterBuilder, SensorConfig, extract_lmb_estimates,
};
use tracktor::prelude::*;

// Birth model for multi-sensor scenario
#[derive(Clone)]
struct MultisensorBirthModel;

impl LabeledBirthModel<f64, 4> for MultisensorBirthModel {
    fn birth_tracks(&self, label_gen: &mut LabelGenerator) -> Vec<BernoulliTrack<f64, 4>> {
        let cov = StateCovariance::from_matrix(nalgebra::matrix![
            200.0, 0.0, 0.0, 0.0;
            0.0, 200.0, 0.0, 0.0;
            0.0, 0.0, 50.0, 0.0;
            0.0, 0.0, 0.0, 50.0
        ]);

        vec![
            BernoulliTrack::new(
                label_gen.next_label(),
                0.02,
                GaussianState::new(1.0, StateVector::from_array([0.0, 0.0, 0.0, 0.0]), cov),
            ),
            BernoulliTrack::new(
                label_gen.next_label(),
                0.02,
                GaussianState::new(1.0, StateVector::from_array([100.0, 100.0, 0.0, 0.0]), cov),
            ),
        ]
    }

    fn expected_birth_count(&self) -> f64 {
        0.04
    }
}

fn main() {
    println!("Multi-Sensor LMB Filter Comparison");
    println!("===================================\n");

    // Shared models
    let transition = ConstantVelocity2D::new(2.0, 0.98);
    // Sensor configurations (different noise levels)
    let sensor1_obs = PositionSensor2D::new(8.0, 0.85); // Noisier sensor
    let sensor1_clutter = UniformClutter2D::new(3.0, (-50.0, 150.0), (-50.0, 150.0));

    let sensor2_obs = PositionSensor2D::new(5.0, 0.92); // Better sensor
    let sensor2_clutter = UniformClutter2D::new(2.0, (-50.0, 150.0), (-50.0, 150.0));

    let sensor_configs = vec![
        SensorConfig::new(sensor1_obs.clone(), sensor1_clutter.clone(), 0.4),
        SensorConfig::new(sensor2_obs.clone(), sensor2_clutter.clone(), 0.6),
    ];

    // Build AA-LMB filter
    let aa_filter = MultisensorLmbFilterBuilder::new(transition.clone(), MultisensorBirthModel)
        .with_sensor_weights(vec![0.4, 0.6])
        .build_aa(10);

    // Build GA-LMB filter
    let ga_filter = MultisensorLmbFilterBuilder::new(transition.clone(), MultisensorBirthModel)
        .with_sensor_weights(vec![0.4, 0.6])
        .build_ga();

    // Build IC-LMB filter
    let ic_filter = MultisensorLmbFilterBuilder::new(transition.clone(), MultisensorBirthModel)
        .with_sensor_weights(vec![0.4, 0.6])
        .build_ic();

    // Initial track
    let make_initial_tracks = || {
        vec![LmbTrack::new(
            Label::new(0, 0),
            0.9,
            GaussianState::new(
                1.0,
                StateVector::from_array([50.0, 50.0, 3.0, 2.0]),
                StateCovariance::from_matrix(nalgebra::matrix![
                    20.0, 0.0, 0.0, 0.0;
                    0.0, 20.0, 0.0, 0.0;
                    0.0, 0.0, 10.0, 0.0;
                    0.0, 0.0, 0.0, 10.0
                ]),
            ),
        )]
    };

    let mut aa_state = aa_filter.initial_state();
    aa_state.tracks.tracks = make_initial_tracks();
    let mut ga_state = ga_filter.initial_state();
    ga_state.tracks.tracks = make_initial_tracks();
    let mut ic_state = ic_filter.initial_state();
    ic_state.tracks.tracks = make_initial_tracks();

    // Simulated measurements from both sensors
    let sensor1_measurements = [
        vec![[53.0, 52.0], [120.0, 80.0]], // Detection + clutter
        vec![[57.0, 55.0]],                // Detection only
        vec![[60.0, 57.0], [10.0, 90.0]],  // Detection + clutter
    ];

    let sensor2_measurements = [
        vec![[52.0, 51.0]],               // Detection only
        vec![[55.0, 53.0], [80.0, 30.0]], // Detection + clutter
        vec![[59.0, 56.0]],               // Detection only
    ];

    let dt = 1.0;

    println!("Running 3 time steps with 2 sensors...\n");

    for t in 0..3 {
        let s1_meas: Vec<Measurement<f64, 2>> = sensor1_measurements[t]
            .iter()
            .map(|m| Measurement::from_array(*m))
            .collect();
        let s2_meas: Vec<Measurement<f64, 2>> = sensor2_measurements[t]
            .iter()
            .map(|m| Measurement::from_array(*m))
            .collect();

        let all_measurements = vec![s1_meas, s2_meas];

        println!("Time {}", t);
        println!(
            "  Sensor 1: {} meas, Sensor 2: {} meas",
            all_measurements[0].len(),
            all_measurements[1].len()
        );

        // Run all filters
        let (aa_updated, _) = aa_filter.step(aa_state, &all_measurements, &sensor_configs, dt);
        let (ga_updated, _) = ga_filter.step(ga_state, &all_measurements, &sensor_configs, dt);
        let (ic_updated, _) = ic_filter.step(ic_state, &all_measurements, &sensor_configs, dt);

        aa_state = aa_updated;
        ga_state = ga_updated;
        ic_state = ic_updated;

        // Prune
        aa_state.tracks.prune_by_existence(0.01);
        ga_state.tracks.prune_by_existence(0.01);
        ic_state.tracks.prune_by_existence(0.01);

        // Extract estimates
        let aa_est = extract_lmb_estimates(&aa_state);
        let ga_est = extract_lmb_estimates(&ga_state);
        let ic_est = extract_lmb_estimates(&ic_state);

        println!(
            "  AA-LMB: {} tracks, exp={:.2}",
            aa_state.num_tracks(),
            aa_state.expected_target_count()
        );
        if let Some(e) = aa_est.first() {
            println!(
                "    Best: ({:.1}, {:.1}), r={:.3}",
                e.state.index(0),
                e.state.index(1),
                e.existence
            );
        }

        println!(
            "  GA-LMB: {} tracks, exp={:.2}",
            ga_state.num_tracks(),
            ga_state.expected_target_count()
        );
        if let Some(e) = ga_est.first() {
            println!(
                "    Best: ({:.1}, {:.1}), r={:.3}",
                e.state.index(0),
                e.state.index(1),
                e.existence
            );
        }

        println!(
            "  IC-LMB: {} tracks, exp={:.2}",
            ic_state.num_tracks(),
            ic_state.expected_target_count()
        );
        if let Some(e) = ic_est.first() {
            println!(
                "    Best: ({:.1}, {:.1}), r={:.3}",
                e.state.index(0),
                e.state.index(1),
                e.existence
            );
        }
        println!();
    }
}
