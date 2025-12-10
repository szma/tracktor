//! Extended Kalman Filter (EKF) for single-target tracking with nonlinear models
//!
//! The EKF extends the standard Kalman filter to handle nonlinear transition
//! and/or observation models by linearizing them around the current state estimate.
//!
//! # Reference
//!
//! Smith, G. L., Schmidt, S. F., & McGee, L. A. (1962). "Application of
//! Statistical Filter Theory to the Optimal Estimation of Position and Velocity
//! on Board a Circumlunar Vehicle." NASA Technical Report TR R-135.
//!
//! # Type Safety
//!
//! The filter uses phantom types and const generics to ensure:
//! - State vectors cannot be mixed with measurements
//! - Dimension mismatches are caught at compile time
//! - Transitions preserve state space typing
//!
//! # Usage
//!
//! The EKF can work with:
//! - Nonlinear transition models (via [`NonlinearTransitionModel`])
//! - Nonlinear observation models (via [`NonlinearObservationModel`])
//! - Any combination of linear and nonlinear models
//!
//! # Example
//!
//! ```
//! use tracktor::filters::ekf::{ExtendedKalmanFilter, EkfState};
//! use tracktor::models::{CoordinatedTurn2D, RangeBearingSensor5D};
//! use tracktor::types::spaces::{StateVector, StateCovariance, Measurement};
//!
//! // Create nonlinear models
//! // CoordinatedTurn2D uses 5D state: [x, y, vx, vy, omega]
//! let transition = CoordinatedTurn2D::new(1.0, 0.1, 0.99);
//! // RangeBearingSensor5D is compatible with 5D state
//! let sensor = RangeBearingSensor5D::new(10.0, 0.01, 0.95);
//!
//! // Create filter
//! let filter = ExtendedKalmanFilter::new(transition, sensor);
//!
//! // Initial state: [x, y, vx, vy, omega]
//! let initial_state = StateVector::from_array([100.0, 100.0, 10.0, 5.0, 0.1]);
//! let initial_cov = StateCovariance::from_diagonal(
//!     &nalgebra::vector![100.0, 100.0, 10.0, 10.0, 0.01]
//! );
//! let mut state = EkfState::new(initial_state, initial_cov);
//!
//! // Predict step (dt = 0.1 second)
//! state = filter.predict(&state, 0.1);
//!
//! // Update with measurement [range, bearing]
//! let measurement = Measurement::from_array([150.0, 0.5]);
//! if let Some(updated) = filter.update(&state, &measurement) {
//!     state = updated;
//! }
//! ```

use core::marker::PhantomData;

use nalgebra::RealField;
use num_traits::Float;

use crate::models::{NonlinearObservationModel, NonlinearTransitionModel, TransitionModel};
use crate::types::spaces::{
    ComputeInnovation, Measurement, MeasurementCovariance, StateCovariance, StateVector,
};
use crate::types::transforms::{compute_innovation_covariance, compute_kalman_gain, joseph_update};

// Re-export KalmanState as EkfState for consistency
pub use super::kalman::KalmanState as EkfState;

// ============================================================================
// Extended Kalman Filter
// ============================================================================

/// An Extended Kalman Filter for nonlinear systems.
///
/// This filter handles nonlinear transition and observation models by
/// linearizing them around the current state estimate using Jacobians.
///
/// # Type Parameters
///
/// - `T`: Scalar type
/// - `Trans`: Transition model type (must implement [`NonlinearTransitionModel`])
/// - `Obs`: Observation model type (must implement [`NonlinearObservationModel`])
/// - `N`: State dimension
/// - `M`: Measurement dimension
///
/// # Linearization
///
/// The EKF performs first-order Taylor series expansion:
/// - Prediction: Uses Jacobian F = ∂f/∂x at the current state
/// - Update: Uses Jacobian H = ∂h/∂x at the predicted state
#[derive(Debug, Clone)]
pub struct ExtendedKalmanFilter<T, Trans, Obs, const N: usize, const M: usize>
where
    T: RealField,
    Trans: NonlinearTransitionModel<T, N>,
    Obs: NonlinearObservationModel<T, N, M>,
{
    /// Nonlinear transition (motion) model
    pub transition: Trans,
    /// Nonlinear observation (sensor) model
    pub observation: Obs,
    /// Phantom marker for scalar type
    _marker: PhantomData<T>,
}

impl<T, Trans, Obs, const N: usize, const M: usize> ExtendedKalmanFilter<T, Trans, Obs, N, M>
where
    T: RealField + Float + Copy,
    Trans: NonlinearTransitionModel<T, N>,
    Obs: NonlinearObservationModel<T, N, M>,
{
    /// Creates a new Extended Kalman Filter with the given nonlinear models.
    #[inline]
    pub fn new(transition: Trans, observation: Obs) -> Self {
        Self {
            transition,
            observation,
            _marker: PhantomData,
        }
    }

    /// Performs the EKF prediction step.
    ///
    /// Uses the nonlinear transition function for mean propagation and
    /// the Jacobian for covariance propagation:
    /// - x_pred = f(x, dt)
    /// - P_pred = F * P * F^T + Q  where F = ∂f/∂x
    ///
    /// # Arguments
    /// - `state`: Current state estimate
    /// - `dt`: Time step (must be non-negative)
    ///
    /// # Returns
    /// Predicted state estimate
    pub fn predict(&self, state: &EkfState<T, N>, dt: T) -> EkfState<T, N> {
        // Nonlinear state prediction
        let predicted_mean = self.transition.predict_nonlinear(&state.mean, dt);

        // Jacobian at current state
        let f = self.transition.jacobian_at(&state.mean, dt);

        // Process noise
        let q = self.transition.process_noise(dt);

        // Linearized covariance propagation
        let predicted_cov = f.propagate_covariance(&state.covariance).add(&q);

        EkfState {
            mean: predicted_mean,
            covariance: predicted_cov,
        }
    }

    /// Performs the EKF update step with a measurement.
    ///
    /// Uses the nonlinear observation function for predicted measurement and
    /// the Jacobian for Kalman gain computation:
    /// - z_pred = h(x)
    /// - H = ∂h/∂x at x_pred
    /// - y = z - z_pred (innovation)
    /// - S = H * P * H^T + R (innovation covariance)
    /// - K = P * H^T * S^{-1} (Kalman gain)
    /// - x_upd = x + K * y
    /// - P_upd = (I - K*H) * P * (I - K*H)^T + K * R * K^T (Joseph form)
    ///
    /// # Arguments
    /// - `state`: Predicted state estimate
    /// - `measurement`: Sensor measurement
    ///
    /// # Returns
    /// Updated state estimate, or `None` if the Jacobian is undefined or
    /// the innovation covariance is singular
    pub fn update(
        &self,
        state: &EkfState<T, N>,
        measurement: &Measurement<T, M>,
    ) -> Option<EkfState<T, N>> {
        // Jacobian at predicted state
        let h = self.observation.jacobian_at(&state.mean)?;

        // Nonlinear predicted measurement
        let predicted_meas = self.observation.observe(&state.mean);

        // Measurement noise
        let r = self.observation.measurement_noise();

        // Innovation
        let innovation = measurement.innovation(predicted_meas);

        // Innovation covariance: S = H * P * H^T + R
        let innovation_cov = compute_innovation_covariance(&state.covariance, &h, &r);

        // Kalman gain: K = P * H^T * S^{-1}
        let kalman_gain = compute_kalman_gain(&state.covariance, &h, &innovation_cov)?;

        // Updated mean: x = x + K * y
        let correction = kalman_gain.correct(&innovation);
        let updated_mean =
            StateVector::from_svector(state.mean.as_svector() + correction.as_svector());

        // Updated covariance (Joseph form for numerical stability)
        let updated_cov = joseph_update(&state.covariance, &kalman_gain, &h, &r);

        Some(EkfState {
            mean: updated_mean,
            covariance: updated_cov,
        })
    }

    /// Performs a single predict-update cycle.
    ///
    /// Convenience method that combines prediction and update.
    ///
    /// # Arguments
    /// - `state`: Current state estimate
    /// - `dt`: Time step for prediction
    /// - `measurement`: Sensor measurement
    ///
    /// # Returns
    /// Updated state after prediction and measurement incorporation
    pub fn step(
        &self,
        state: &EkfState<T, N>,
        dt: T,
        measurement: &Measurement<T, M>,
    ) -> Option<EkfState<T, N>> {
        let predicted = self.predict(state, dt);
        self.update(&predicted, measurement)
    }

    /// Computes the likelihood of a measurement given the current state.
    ///
    /// Uses the nonlinear observation model and linearized innovation covariance.
    ///
    /// # Returns
    /// Measurement likelihood, or `None` if computation fails
    pub fn measurement_likelihood(
        &self,
        state: &EkfState<T, N>,
        measurement: &Measurement<T, M>,
    ) -> Option<T> {
        let h = self.observation.jacobian_at(&state.mean)?;
        let r = self.observation.measurement_noise();

        // Nonlinear predicted measurement
        let predicted_meas = self.observation.observe(&state.mean);

        // Innovation
        let innovation = measurement.innovation(predicted_meas);

        // Innovation covariance
        let innovation_cov = compute_innovation_covariance(&state.covariance, &h, &r);

        // Convert to innovation space covariance for likelihood computation
        let innovation_cov_typed =
            crate::types::spaces::Covariance::from_matrix(*innovation_cov.as_matrix());
        crate::types::gaussian::gaussian_likelihood(&innovation, &innovation_cov_typed)
    }

    /// Computes the Mahalanobis distance between a measurement and the predicted measurement.
    ///
    /// Uses the nonlinear observation model for computing the innovation.
    ///
    /// # Returns
    /// Squared Mahalanobis distance, or `None` if the innovation covariance is singular
    pub fn mahalanobis_distance_squared(
        &self,
        state: &EkfState<T, N>,
        measurement: &Measurement<T, M>,
    ) -> Option<T> {
        let h = self.observation.jacobian_at(&state.mean)?;
        let r = self.observation.measurement_noise();

        // Nonlinear predicted measurement
        let predicted_meas = self.observation.observe(&state.mean);

        // Innovation
        let innovation = measurement.innovation(predicted_meas);

        // Innovation covariance
        let innovation_cov = compute_innovation_covariance(&state.covariance, &h, &r);

        // Inverse of innovation covariance
        let s_inv = innovation_cov.as_matrix().try_inverse()?;

        // Mahalanobis distance: d² = y^T * S^{-1} * y
        let y = innovation.as_svector();
        let d_sq = (y.transpose() * s_inv * y)[(0, 0)];

        Some(d_sq)
    }
}

// ============================================================================
// EKF with Linear Transition
// ============================================================================

/// An Extended Kalman Filter with a linear transition model but nonlinear observation.
///
/// This is a common configuration where the dynamics are linear (e.g., constant velocity)
/// but the sensor is nonlinear (e.g., range-bearing radar).
#[derive(Debug, Clone)]
pub struct EkfLinearTransition<T, Trans, Obs, const N: usize, const M: usize>
where
    T: RealField,
    Trans: TransitionModel<T, N>,
    Obs: NonlinearObservationModel<T, N, M>,
{
    /// Linear transition (motion) model
    pub transition: Trans,
    /// Nonlinear observation (sensor) model
    pub observation: Obs,
    /// Phantom marker for scalar type
    _marker: PhantomData<T>,
}

impl<T, Trans, Obs, const N: usize, const M: usize> EkfLinearTransition<T, Trans, Obs, N, M>
where
    T: RealField + Float + Copy,
    Trans: TransitionModel<T, N>,
    Obs: NonlinearObservationModel<T, N, M>,
{
    /// Creates a new EKF with linear transition and nonlinear observation.
    #[inline]
    pub fn new(transition: Trans, observation: Obs) -> Self {
        Self {
            transition,
            observation,
            _marker: PhantomData,
        }
    }

    /// Performs the standard Kalman prediction step (linear).
    pub fn predict(&self, state: &EkfState<T, N>, dt: T) -> EkfState<T, N> {
        let f = self.transition.transition_matrix(dt);
        let q = self.transition.process_noise(dt);

        let predicted_mean = f.apply_state(&state.mean);
        let predicted_cov = f.propagate_covariance(&state.covariance).add(&q);

        EkfState {
            mean: predicted_mean,
            covariance: predicted_cov,
        }
    }

    /// Performs the EKF update step with nonlinear observation.
    pub fn update(
        &self,
        state: &EkfState<T, N>,
        measurement: &Measurement<T, M>,
    ) -> Option<EkfState<T, N>> {
        let h = self.observation.jacobian_at(&state.mean)?;
        let predicted_meas = self.observation.observe(&state.mean);
        let r = self.observation.measurement_noise();

        let innovation = measurement.innovation(predicted_meas);
        let innovation_cov = compute_innovation_covariance(&state.covariance, &h, &r);
        let kalman_gain = compute_kalman_gain(&state.covariance, &h, &innovation_cov)?;

        let correction = kalman_gain.correct(&innovation);
        let updated_mean =
            StateVector::from_svector(state.mean.as_svector() + correction.as_svector());
        let updated_cov = joseph_update(&state.covariance, &kalman_gain, &h, &r);

        Some(EkfState {
            mean: updated_mean,
            covariance: updated_cov,
        })
    }

    /// Performs a single predict-update cycle.
    pub fn step(
        &self,
        state: &EkfState<T, N>,
        dt: T,
        measurement: &Measurement<T, M>,
    ) -> Option<EkfState<T, N>> {
        let predicted = self.predict(state, dt);
        self.update(&predicted, measurement)
    }
}

// ============================================================================
// Standalone EKF Functions
// ============================================================================

/// Performs an EKF prediction step with explicit nonlinear function and Jacobian.
///
/// This is useful for one-off predictions without creating a filter struct.
///
/// # Arguments
/// - `state`: Current state estimate
/// - `predicted_mean`: The result of applying the nonlinear transition f(x, dt)
/// - `jacobian`: The transition Jacobian F = ∂f/∂x
/// - `process_noise`: The process noise covariance Q
pub fn predict_ekf<T: RealField + Copy, const N: usize>(
    state: &EkfState<T, N>,
    predicted_mean: StateVector<T, N>,
    jacobian: &crate::types::transforms::TransitionMatrix<T, N>,
    process_noise: &StateCovariance<T, N>,
) -> EkfState<T, N> {
    let predicted_cov = jacobian
        .propagate_covariance(&state.covariance)
        .add(process_noise);

    EkfState {
        mean: predicted_mean,
        covariance: predicted_cov,
    }
}

/// Performs an EKF update step with explicit nonlinear observation and Jacobian.
///
/// # Arguments
/// - `state`: Predicted state estimate
/// - `measurement`: Actual measurement
/// - `predicted_measurement`: The result of applying the nonlinear observation h(x)
/// - `jacobian`: The observation Jacobian H = ∂h/∂x
/// - `meas_noise`: The measurement noise covariance R
pub fn update_ekf<T: RealField + Copy, const N: usize, const M: usize>(
    state: &EkfState<T, N>,
    measurement: &Measurement<T, M>,
    predicted_measurement: Measurement<T, M>,
    jacobian: &crate::types::transforms::ObservationMatrix<T, M, N>,
    meas_noise: &MeasurementCovariance<T, M>,
) -> Option<EkfState<T, N>> {
    let innovation = measurement.innovation(predicted_measurement);
    let innovation_cov = compute_innovation_covariance(&state.covariance, jacobian, meas_noise);
    let kalman_gain = compute_kalman_gain(&state.covariance, jacobian, &innovation_cov)?;

    let correction = kalman_gain.correct(&innovation);
    let updated_mean = StateVector::from_svector(state.mean.as_svector() + correction.as_svector());
    let updated_cov = joseph_update(&state.covariance, &kalman_gain, jacobian, meas_noise);

    Some(EkfState {
        mean: updated_mean,
        covariance: updated_cov,
    })
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{
        ConstantVelocity2D, CoordinatedTurn2D, RangeBearingSensor, RangeBearingSensor5D,
    };

    #[test]
    fn test_ekf_state_creation() {
        let mean: StateVector<f64, 5> = StateVector::from_array([0.0, 0.0, 1.0, 0.0, 0.1]);
        let cov: StateCovariance<f64, 5> = StateCovariance::identity();

        let state = EkfState::new(mean, cov);
        assert!((state.mean.index(2) - 1.0).abs() < 1e-10);
        assert!((state.uncertainty() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_ekf_predict_coordinated_turn() {
        let transition = CoordinatedTurn2D::new(0.1_f64, 0.01, 0.99);
        let sensor = RangeBearingSensor5D::new(10.0, 0.01, 0.95);
        let filter = ExtendedKalmanFilter::new(transition, sensor);

        // State at (100, 0) moving north at 10 m/s with no turn
        let state = EkfState::new(
            StateVector::from_array([100.0, 0.0, 0.0, 10.0, 0.0]),
            StateCovariance::identity(),
        );

        // Predict 1 second ahead
        let predicted = filter.predict(&state, 1.0);

        // Should move north by ~10m
        assert!((predicted.mean.index(0) - 100.0).abs() < 1e-6);
        assert!((predicted.mean.index(1) - 10.0).abs() < 1e-6);

        // Covariance should increase
        assert!(predicted.uncertainty() > state.uncertainty());
    }

    #[test]
    fn test_ekf_predict_with_turn() {
        use std::f64::consts::FRAC_PI_2;

        let transition = CoordinatedTurn2D::new(0.1_f64, 0.01, 0.99);
        let sensor = RangeBearingSensor5D::new(10.0, 0.01, 0.95);
        let filter = ExtendedKalmanFilter::new(transition, sensor);

        // Moving east at 10 m/s, turning left at pi/2 rad/s
        let state = EkfState::new(
            StateVector::from_array([0.0, 0.0, 10.0, 0.0, FRAC_PI_2]),
            StateCovariance::identity(),
        );

        // After 1 second, should have turned 90 degrees
        let predicted = filter.predict(&state, 1.0);

        // Turn radius r = v/omega = 10/(pi/2) ≈ 6.37
        let r = 10.0 / FRAC_PI_2;

        // Position should be at approximately (r, r)
        assert!(
            (predicted.mean.index(0) - r).abs() < 0.1,
            "x: {} vs {}",
            predicted.mean.index(0),
            r
        );
        assert!(
            (predicted.mean.index(1) - r).abs() < 0.1,
            "y: {} vs {}",
            predicted.mean.index(1),
            r
        );

        // Velocity should now be pointing north (0, 10)
        assert!(
            (predicted.mean.index(2) - 0.0).abs() < 0.1,
            "vx: {}",
            predicted.mean.index(2)
        );
        assert!(
            (predicted.mean.index(3) - 10.0).abs() < 0.1,
            "vy: {}",
            predicted.mean.index(3)
        );
    }

    #[test]
    fn test_ekf_update_range_bearing() {
        let transition = CoordinatedTurn2D::new(0.1_f64, 0.01, 0.99);
        let sensor = RangeBearingSensor5D::new(5.0, 0.01, 0.95);
        let filter = ExtendedKalmanFilter::new(transition, sensor);

        // High uncertainty state at (100, 0)
        let state = EkfState::new(
            StateVector::from_array([100.0, 0.0, 0.0, 0.0, 0.0]),
            StateCovariance::from_matrix(nalgebra::SMatrix::<f64, 5, 5>::identity().scale(1000.0)),
        );

        // Measurement: range=100, bearing=0 (confirms position)
        let measurement = Measurement::from_array([100.0, 0.0]);
        let updated = filter.update(&state, &measurement).unwrap();

        // Should stay near (100, 0)
        assert!((updated.mean.index(0) - 100.0).abs() < 10.0);
        assert!(updated.mean.index(1).abs() < 10.0);

        // Uncertainty should decrease
        assert!(updated.uncertainty() < state.uncertainty());
    }

    #[test]
    fn test_ekf_linear_transition() {
        let transition = ConstantVelocity2D::new(0.1_f64, 0.99);
        let sensor = RangeBearingSensor::new(10.0, 0.01, 0.95);
        let filter = EkfLinearTransition::new(transition, sensor);

        // State at (100, 0) moving east at 10 m/s
        let state = EkfState::new(
            StateVector::from_array([100.0, 0.0, 10.0, 0.0]),
            StateCovariance::identity(),
        );

        // Predict 1 second
        let predicted = filter.predict(&state, 1.0);

        // Should move to (110, 0)
        assert!((predicted.mean.index(0) - 110.0).abs() < 1e-6);
        assert!((predicted.mean.index(1) - 0.0).abs() < 1e-6);

        // Update with range-bearing measurement
        // At (110, 0), range = 110, bearing = 0
        let measurement = Measurement::from_array([110.0, 0.0]);
        let updated = filter.update(&predicted, &measurement).unwrap();

        // Should still be near (110, 0)
        assert!((updated.mean.index(0) - 110.0).abs() < 5.0);
    }

    #[test]
    fn test_ekf_step() {
        let transition = CoordinatedTurn2D::new(0.1_f64, 0.01, 0.99);
        let sensor = RangeBearingSensor5D::new(10.0, 0.01, 0.95);
        let filter = ExtendedKalmanFilter::new(transition, sensor);

        let state = EkfState::new(
            StateVector::from_array([100.0, 0.0, 10.0, 0.0, 0.0]),
            StateCovariance::identity(),
        );

        // After 1 second moving east, should be at ~(110, 0)
        let measurement = Measurement::from_array([110.0, 0.0]);
        let updated = filter.step(&state, 1.0, &measurement).unwrap();

        assert!((updated.mean.index(0) - 110.0).abs() < 5.0);
    }

    #[test]
    fn test_ekf_mahalanobis_distance() {
        let transition = CoordinatedTurn2D::new(0.1_f64, 0.01, 0.99);
        let sensor = RangeBearingSensor5D::new(10.0, 0.01, 0.95);
        let filter = ExtendedKalmanFilter::new(transition, sensor);

        let state = EkfState::new(
            StateVector::from_array([100.0, 0.0, 0.0, 0.0, 0.0]),
            StateCovariance::identity(),
        );

        // Close measurement (expected: range=100, bearing=0)
        let close_meas = Measurement::from_array([100.0, 0.0]);
        let close_dist = filter
            .mahalanobis_distance_squared(&state, &close_meas)
            .unwrap();

        // Far measurement
        let far_meas = Measurement::from_array([200.0, 1.0]);
        let far_dist = filter
            .mahalanobis_distance_squared(&state, &far_meas)
            .unwrap();

        assert!(close_dist < far_dist);
    }

    #[test]
    fn test_standalone_ekf_functions() {
        let mean = StateVector::from_array([100.0_f64, 0.0, 10.0, 0.0]);
        let cov = StateCovariance::identity();
        let state = EkfState::new(mean, cov);

        // Manual nonlinear prediction (constant velocity for simplicity)
        let dt = 1.0;
        let predicted_mean = StateVector::from_array([110.0, 0.0, 10.0, 0.0]);
        let jacobian = crate::types::transforms::TransitionMatrix::from_matrix(nalgebra::matrix![
            1.0, 0.0, dt, 0.0_f64;
            0.0, 1.0, 0.0, dt;
            0.0, 0.0, 1.0, 0.0;
            0.0, 0.0, 0.0, 1.0
        ]);
        let process_noise =
            StateCovariance::from_matrix(nalgebra::SMatrix::<f64, 4, 4>::identity().scale(0.1));

        let predicted = predict_ekf(&state, predicted_mean, &jacobian, &process_noise);
        assert!((predicted.mean.index(0) - 110.0).abs() < 1e-10);

        // Range-bearing update
        let sensor = RangeBearingSensor::new(10.0, 0.01, 0.95);
        let h = sensor.jacobian_at(&predicted.mean).unwrap();
        let z_pred = sensor.observe(&predicted.mean);
        let r = sensor.measurement_noise();

        let measurement = Measurement::from_array([110.0, 0.0]);
        let updated = update_ekf(&predicted, &measurement, z_pred, &h, &r).unwrap();

        assert!((updated.mean.index(0) - 110.0).abs() < 5.0);
    }
}
