//! Common test helpers for multi-target filter integration tests

#![cfg(feature = "alloc")]
#![allow(dead_code)]

use tracktor::filters::lmb::filter::LabeledBirthModel;
use tracktor::models::{ConstantVelocity2D, FixedBirthModel, PositionSensor2D, UniformClutter2D};
use tracktor::types::gaussian::GaussianState;
use tracktor::types::labels::{BernoulliTrack, Label, LabelGenerator};
use tracktor::types::spaces::{Measurement, StateCovariance, StateVector};

#[cfg(feature = "alloc")]
use tracktor::filters::glmb::GlmbTrack;

/// Creates a measurement at given position
pub fn make_measurement(x: f64, y: f64) -> Measurement<f64, 2> {
    Measurement::from_array([x, y])
}

/// Creates standard test models for PHD/CPHD
pub fn make_test_models() -> (
    ConstantVelocity2D<f64>,
    PositionSensor2D<f64>,
    UniformClutter2D<f64>,
    FixedBirthModel<f64, 4>,
) {
    let transition = ConstantVelocity2D::new(1.0, 0.99);
    let observation = PositionSensor2D::new(5.0, 0.9);
    let clutter = UniformClutter2D::new(5.0, (0.0, 200.0), (0.0, 200.0));
    let birth = FixedBirthModel::<f64, 4>::new();
    (transition, observation, clutter, birth)
}

/// Creates models with birth locations for PHD/CPHD
pub fn make_test_models_with_birth(
    birth_locations: &[(f64, f64)],
) -> (
    ConstantVelocity2D<f64>,
    PositionSensor2D<f64>,
    UniformClutter2D<f64>,
    FixedBirthModel<f64, 4>,
) {
    let transition = ConstantVelocity2D::new(1.0, 0.99);
    let observation = PositionSensor2D::new(5.0, 0.9);
    let clutter = UniformClutter2D::new(5.0, (0.0, 200.0), (0.0, 200.0));
    let mut birth = FixedBirthModel::<f64, 4>::new();

    for &(x, y) in birth_locations {
        birth.add_birth_location(
            0.1,
            StateVector::from_array([x, y, 0.0, 0.0]),
            StateCovariance::from_matrix(nalgebra::matrix![
                100.0, 0.0, 0.0, 0.0;
                0.0, 100.0, 0.0, 0.0;
                0.0, 0.0, 25.0, 0.0;
                0.0, 0.0, 0.0, 25.0
            ]),
        );
    }

    (transition, observation, clutter, birth)
}

/// Creates a Gaussian component at given position
pub fn make_component(x: f64, y: f64, vx: f64, vy: f64, weight: f64) -> GaussianState<f64, 4> {
    let mean = StateVector::from_array([x, y, vx, vy]);
    let cov = StateCovariance::from_matrix(nalgebra::matrix![
        10.0, 0.0, 0.0, 0.0;
        0.0, 10.0, 0.0, 0.0;
        0.0, 0.0, 5.0, 0.0;
        0.0, 0.0, 0.0, 5.0
    ]);
    GaussianState::new(weight, mean, cov)
}

/// Simple birth model for GLMB testing
pub struct TestGlmbBirthModel {
    birth_locations: Vec<(f64, StateVector<f64, 4>, StateCovariance<f64, 4>)>,
}

impl TestGlmbBirthModel {
    pub fn new() -> Self {
        Self {
            birth_locations: Vec::new(),
        }
    }

    pub fn with_location(mut self, existence: f64, x: f64, y: f64) -> Self {
        let cov = StateCovariance::from_matrix(nalgebra::matrix![
            100.0, 0.0, 0.0, 0.0;
            0.0, 100.0, 0.0, 0.0;
            0.0, 0.0, 25.0, 0.0;
            0.0, 0.0, 0.0, 25.0
        ]);
        self.birth_locations
            .push((existence, StateVector::from_array([x, y, 0.0, 0.0]), cov));
        self
    }
}

impl LabeledBirthModel<f64, 4> for TestGlmbBirthModel {
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

/// No-op birth model for deterministic GLMB testing
#[derive(Clone)]
pub struct NoGlmbBirthModel;

impl LabeledBirthModel<f64, 4> for NoGlmbBirthModel {
    fn birth_tracks(&self, _label_gen: &mut LabelGenerator) -> Vec<BernoulliTrack<f64, 4>> {
        Vec::new()
    }

    fn expected_birth_count(&self) -> f64 {
        0.0
    }
}

/// Creates a GLMB track at given position
#[cfg(feature = "alloc")]
pub fn make_glmb_track(label: Label, x: f64, y: f64, vx: f64, vy: f64) -> GlmbTrack<f64, 4> {
    let mean = StateVector::from_array([x, y, vx, vy]);
    let cov = StateCovariance::from_matrix(nalgebra::matrix![
        10.0, 0.0, 0.0, 0.0;
        0.0, 10.0, 0.0, 0.0;
        0.0, 0.0, 5.0, 0.0;
        0.0, 0.0, 0.0, 5.0
    ]);
    let state = GaussianState::new(1.0, mean, cov);
    GlmbTrack::new(label, state)
}
