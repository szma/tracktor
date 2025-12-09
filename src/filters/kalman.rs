//! Standard Kalman Filter for single-target tracking
//!
//! A type-safe implementation of the discrete-time Kalman filter that leverages
//! the library's compile-time dimension and space checking.
//!
//! # Type Safety
//!
//! The filter uses phantom types to ensure:
//! - State vectors cannot be mixed with measurements
//! - Dimension mismatches are caught at compile time
//! - Transitions preserve state space typing
//!
//! # Example
//!
//! ```
//! use tracktor::filters::kalman::{KalmanFilter, KalmanState};
//! use tracktor::models::{ConstantVelocity2D, PositionSensor2D};
//! use tracktor::types::spaces::{StateVector, StateCovariance, Measurement};
//!
//! // Create models
//! let transition = ConstantVelocity2D::new(1.0, 0.99);
//! let sensor = PositionSensor2D::new(5.0, 0.95);
//!
//! // Create filter
//! let filter = KalmanFilter::new(transition, sensor);
//!
//! // Initial state: [x, y, vx, vy]
//! let initial_state = StateVector::from_array([0.0, 0.0, 1.0, 0.0]);
//! let initial_cov = StateCovariance::from_diagonal(
//!     &nalgebra::vector![10.0, 10.0, 1.0, 1.0]
//! );
//! let mut state = KalmanState::new(initial_state, initial_cov);
//!
//! // Predict step (dt = 1.0 second)
//! state = filter.predict(&state, 1.0);
//!
//! // Update with measurement
//! let measurement = Measurement::from_array([1.5, 0.2]);
//! if let Some(updated) = filter.update(&state, &measurement) {
//!     state = updated;
//! }
//! ```

use core::marker::PhantomData;

use nalgebra::RealField;
use num_traits::Float;

use crate::models::{ObservationModel, TransitionModel};
use crate::types::spaces::{
    ComputeInnovation, Measurement, MeasurementCovariance, StateCovariance, StateVector,
};
use crate::types::transforms::{
    compute_innovation_covariance, compute_kalman_gain, joseph_update, ObservationMatrix,
    TransitionMatrix,
};

// ============================================================================
// Kalman Filter State
// ============================================================================

/// State estimate for the Kalman filter.
///
/// Contains the mean and covariance of the state estimate. Unlike `GaussianState`,
/// this does not carry a weight since single-target tracking always has exactly
/// one target.
///
/// # Type Parameters
///
/// - `T`: Scalar type (typically `f32` or `f64`)
/// - `N`: State dimension (compile-time constant)
#[derive(Debug, Clone, PartialEq)]
pub struct KalmanState<T: RealField, const N: usize> {
    /// State estimate mean
    pub mean: StateVector<T, N>,
    /// State estimate covariance
    pub covariance: StateCovariance<T, N>,
}

impl<T: RealField + Copy, const N: usize> KalmanState<T, N> {
    /// Creates a new Kalman filter state.
    #[inline]
    pub fn new(mean: StateVector<T, N>, covariance: StateCovariance<T, N>) -> Self {
        Self { mean, covariance }
    }

    /// Creates a state with identity covariance.
    #[inline]
    pub fn with_identity_covariance(mean: StateVector<T, N>) -> Self {
        Self {
            mean,
            covariance: StateCovariance::identity(),
        }
    }

    /// Creates a state with diagonal covariance.
    #[inline]
    pub fn with_diagonal_covariance(
        mean: StateVector<T, N>,
        diagonal: &nalgebra::SVector<T, N>,
    ) -> Self {
        Self {
            mean,
            covariance: StateCovariance::from_diagonal(diagonal),
        }
    }

    /// Returns the trace of the covariance matrix (sum of variances).
    #[inline]
    pub fn uncertainty(&self) -> T {
        self.covariance.trace()
    }

    /// Extracts a position from the state (assumes position is in first components).
    ///
    /// Returns the first `P` components of the state vector.
    #[inline]
    pub fn position<const P: usize>(&self) -> [T; P] {
        let mut pos = [T::zero(); P];
        for (i, p) in pos.iter_mut().enumerate() {
            *p = *self.mean.index(i);
        }
        pos
    }
}

// ============================================================================
// Kalman Filter
// ============================================================================

/// A standard discrete-time Kalman filter.
///
/// This filter combines a transition model (dynamics) with an observation model
/// (sensor) to perform optimal linear state estimation.
///
/// # Type Parameters
///
/// - `T`: Scalar type
/// - `Trans`: Transition model type
/// - `Obs`: Observation model type
/// - `N`: State dimension
/// - `M`: Measurement dimension
///
/// # Type Safety
///
/// The const generics ensure at compile time that:
/// - The transition model operates on `N`-dimensional states
/// - The observation model maps `N`-dimensional states to `M`-dimensional measurements
/// - All matrix operations have compatible dimensions
#[derive(Debug, Clone)]
pub struct KalmanFilter<T, Trans, Obs, const N: usize, const M: usize>
where
    T: RealField,
    Trans: TransitionModel<T, N>,
    Obs: ObservationModel<T, N, M>,
{
    /// Transition (motion) model
    pub transition: Trans,
    /// Observation (sensor) model
    pub observation: Obs,
    /// Phantom marker for scalar type
    _marker: PhantomData<T>,
}

impl<T, Trans, Obs, const N: usize, const M: usize> KalmanFilter<T, Trans, Obs, N, M>
where
    T: RealField + Float + Copy,
    Trans: TransitionModel<T, N>,
    Obs: ObservationModel<T, N, M>,
{
    /// Creates a new Kalman filter with the given models.
    #[inline]
    pub fn new(transition: Trans, observation: Obs) -> Self {
        Self {
            transition,
            observation,
            _marker: PhantomData,
        }
    }

    /// Performs the prediction step.
    ///
    /// Propagates the state estimate forward in time using the transition model:
    /// - x_pred = F * x
    /// - P_pred = F * P * F^T + Q
    ///
    /// # Arguments
    /// - `state`: Current state estimate
    /// - `dt`: Time step (must be non-negative)
    ///
    /// # Returns
    /// Predicted state estimate
    pub fn predict(&self, state: &KalmanState<T, N>, dt: T) -> KalmanState<T, N> {
        let f = self.transition.transition_matrix(dt);
        let q = self.transition.process_noise(dt);

        let predicted_mean = f.apply_state(&state.mean);
        let predicted_cov = f.propagate_covariance(&state.covariance).add(&q);

        KalmanState {
            mean: predicted_mean,
            covariance: predicted_cov,
        }
    }

    /// Performs the update step with a measurement.
    ///
    /// Incorporates a measurement to refine the state estimate:
    /// - y = z - H * x (innovation)
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
    /// Updated state estimate, or `None` if the innovation covariance is singular
    pub fn update(
        &self,
        state: &KalmanState<T, N>,
        measurement: &Measurement<T, M>,
    ) -> Option<KalmanState<T, N>> {
        let h = self.observation.observation_matrix();
        let r = self.observation.measurement_noise();

        self.update_with_matrices(state, measurement, &h, &r)
    }

    /// Performs the update step with explicit matrices.
    ///
    /// This is useful when you have pre-computed observation matrices (e.g., for EKF).
    pub fn update_with_matrices(
        &self,
        state: &KalmanState<T, N>,
        measurement: &Measurement<T, M>,
        obs_matrix: &ObservationMatrix<T, M, N>,
        meas_noise: &MeasurementCovariance<T, M>,
    ) -> Option<KalmanState<T, N>> {
        // Predicted measurement
        let predicted_meas = obs_matrix.observe(&state.mean);

        // Innovation
        let innovation = measurement.innovation(predicted_meas);

        // Innovation covariance: S = H * P * H^T + R
        let innovation_cov =
            compute_innovation_covariance(&state.covariance, obs_matrix, meas_noise);

        // Kalman gain: K = P * H^T * S^{-1}
        let kalman_gain = compute_kalman_gain(&state.covariance, obs_matrix, &innovation_cov)?;

        // Updated mean: x = x + K * y
        let correction = kalman_gain.correct(&innovation);
        let updated_mean =
            StateVector::from_svector(state.mean.as_svector() + correction.as_svector());

        // Updated covariance (Joseph form for numerical stability)
        let updated_cov = joseph_update(&state.covariance, &kalman_gain, obs_matrix, meas_noise);

        Some(KalmanState {
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
        state: &KalmanState<T, N>,
        dt: T,
        measurement: &Measurement<T, M>,
    ) -> Option<KalmanState<T, N>> {
        let predicted = self.predict(state, dt);
        self.update(&predicted, measurement)
    }

    /// Computes the likelihood of a measurement given the current state.
    ///
    /// This is useful for gating (rejecting unlikely measurements) or for
    /// data association in multi-target scenarios.
    ///
    /// # Returns
    /// Measurement likelihood, or `None` if computation fails
    pub fn measurement_likelihood(
        &self,
        state: &KalmanState<T, N>,
        measurement: &Measurement<T, M>,
    ) -> Option<T> {
        let h = self.observation.observation_matrix();
        let r = self.observation.measurement_noise();

        // Predicted measurement
        let predicted_meas = h.observe(&state.mean);

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
    /// This is useful for gating: measurements with Mahalanobis distance above a threshold
    /// (e.g., chi-squared with M degrees of freedom) are unlikely to originate from this target.
    ///
    /// # Returns
    /// Squared Mahalanobis distance, or `None` if the innovation covariance is singular
    pub fn mahalanobis_distance_squared(
        &self,
        state: &KalmanState<T, N>,
        measurement: &Measurement<T, M>,
    ) -> Option<T> {
        let h = self.observation.observation_matrix();
        let r = self.observation.measurement_noise();

        // Predicted measurement
        let predicted_meas = h.observe(&state.mean);

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
// Standalone Functions (for use without a filter struct)
// ============================================================================

/// Performs a single Kalman filter prediction step.
///
/// This is a standalone function for cases where you don't want to create
/// a full `KalmanFilter` struct.
pub fn predict<T: RealField + Copy, const N: usize>(
    state: &KalmanState<T, N>,
    transition: &TransitionMatrix<T, N>,
    process_noise: &StateCovariance<T, N>,
) -> KalmanState<T, N> {
    let predicted_mean = transition.apply_state(&state.mean);
    let predicted_cov = transition
        .propagate_covariance(&state.covariance)
        .add(process_noise);

    KalmanState {
        mean: predicted_mean,
        covariance: predicted_cov,
    }
}

/// Performs a single Kalman filter update step.
///
/// This is a standalone function for cases where you don't want to create
/// a full `KalmanFilter` struct.
///
/// # Returns
/// Updated state, or `None` if the innovation covariance is singular
pub fn update<T: RealField + Copy, const N: usize, const M: usize>(
    state: &KalmanState<T, N>,
    measurement: &Measurement<T, M>,
    obs_matrix: &ObservationMatrix<T, M, N>,
    meas_noise: &MeasurementCovariance<T, M>,
) -> Option<KalmanState<T, N>> {
    // Predicted measurement
    let predicted_meas = obs_matrix.observe(&state.mean);

    // Innovation
    let innovation = measurement.innovation(predicted_meas);

    // Innovation covariance
    let innovation_cov = compute_innovation_covariance(&state.covariance, obs_matrix, meas_noise);

    // Kalman gain
    let kalman_gain = compute_kalman_gain(&state.covariance, obs_matrix, &innovation_cov)?;

    // Updated mean
    let correction = kalman_gain.correct(&innovation);
    let updated_mean = StateVector::from_svector(state.mean.as_svector() + correction.as_svector());

    // Updated covariance (Joseph form)
    let updated_cov = joseph_update(&state.covariance, &kalman_gain, obs_matrix, meas_noise);

    Some(KalmanState {
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
    use crate::models::{ConstantVelocity2D, PositionSensor2D};

    #[test]
    fn test_kalman_state_creation() {
        let mean: StateVector<f64, 4> = StateVector::from_array([0.0, 0.0, 1.0, 0.0]);
        let cov: StateCovariance<f64, 4> = StateCovariance::identity();

        let state = KalmanState::new(mean, cov);
        assert!((state.mean.index(2) - 1.0).abs() < 1e-10);
        assert!((state.uncertainty() - 4.0).abs() < 1e-10); // trace of 4x4 identity
    }

    #[test]
    fn test_kalman_predict() {
        let transition = ConstantVelocity2D::new(0.1, 0.99);
        let sensor = PositionSensor2D::new(1.0, 0.95);
        let filter = KalmanFilter::new(transition, sensor);

        // Initial state: at origin, moving right at 10 m/s
        let state = KalmanState::new(
            StateVector::from_array([0.0, 0.0, 10.0, 0.0]),
            StateCovariance::identity(),
        );

        // Predict 1 second ahead
        let predicted = filter.predict(&state, 1.0);

        // Position should advance by velocity
        assert!((predicted.mean.index(0) - 10.0).abs() < 1e-10);
        assert!((predicted.mean.index(1) - 0.0).abs() < 1e-10);

        // Velocity should remain the same
        assert!((predicted.mean.index(2) - 10.0).abs() < 1e-10);
        assert!((predicted.mean.index(3) - 0.0).abs() < 1e-10);

        // Covariance should increase (uncertainty grows during prediction)
        assert!(predicted.uncertainty() > state.uncertainty());
    }

    #[test]
    fn test_kalman_update() {
        let transition = ConstantVelocity2D::new(0.1, 0.99);
        let sensor = PositionSensor2D::new(1.0, 0.95);
        let filter = KalmanFilter::new(transition, sensor);

        // High uncertainty state
        let state = KalmanState::new(
            StateVector::from_array([0.0, 0.0, 0.0, 0.0]),
            StateCovariance::from_matrix(nalgebra::SMatrix::<f64, 4, 4>::identity().scale(100.0)),
        );

        // Measurement at (10, 5)
        let measurement = Measurement::from_array([10.0, 5.0]);

        let updated = filter.update(&state, &measurement).unwrap();

        // State should move toward measurement
        assert!(updated.mean.index(0) > &5.0);
        assert!(updated.mean.index(1) > &2.0);

        // Uncertainty should decrease
        assert!(updated.uncertainty() < state.uncertainty());
    }

    #[test]
    fn test_kalman_step() {
        let transition = ConstantVelocity2D::new(0.1, 0.99);
        let sensor = PositionSensor2D::new(1.0, 0.95);
        let filter = KalmanFilter::new(transition, sensor);

        let state = KalmanState::new(
            StateVector::from_array([0.0, 0.0, 10.0, 5.0]),
            StateCovariance::identity(),
        );

        // After 1 second, expect to be at (10, 5)
        // Measurement confirms this
        let measurement = Measurement::from_array([10.0, 5.0]);

        let updated = filter.step(&state, 1.0, &measurement).unwrap();

        // Should be very close to (10, 5)
        assert!((updated.mean.index(0) - 10.0).abs() < 1.0);
        assert!((updated.mean.index(1) - 5.0).abs() < 1.0);
    }

    #[test]
    fn test_measurement_likelihood() {
        let transition = ConstantVelocity2D::new(0.1, 0.99);
        let sensor = PositionSensor2D::new(1.0, 0.95);
        let filter = KalmanFilter::new(transition, sensor);

        let state = KalmanState::new(
            StateVector::from_array([10.0, 5.0, 0.0, 0.0]),
            StateCovariance::identity(),
        );

        // Measurement exactly at expected position
        let close_meas = Measurement::from_array([10.0, 5.0]);
        let close_likelihood = filter.measurement_likelihood(&state, &close_meas).unwrap();

        // Measurement far from expected position
        let far_meas = Measurement::from_array([100.0, 100.0]);
        let far_likelihood = filter.measurement_likelihood(&state, &far_meas).unwrap();

        // Close measurement should have higher likelihood
        assert!(close_likelihood > far_likelihood);
    }

    #[test]
    fn test_mahalanobis_distance() {
        let transition = ConstantVelocity2D::new(0.1, 0.99);
        let sensor = PositionSensor2D::new(1.0, 0.95);
        let filter = KalmanFilter::new(transition, sensor);

        let state = KalmanState::new(
            StateVector::from_array([0.0, 0.0, 0.0, 0.0]),
            StateCovariance::identity(),
        );

        // Measurement at expected position
        let close_meas = Measurement::from_array([0.0, 0.0]);
        let close_dist = filter
            .mahalanobis_distance_squared(&state, &close_meas)
            .unwrap();

        // Measurement far away
        let far_meas = Measurement::from_array([10.0, 10.0]);
        let far_dist = filter
            .mahalanobis_distance_squared(&state, &far_meas)
            .unwrap();

        // Close measurement should have smaller Mahalanobis distance
        assert!(close_dist < far_dist);
        assert!(close_dist < 0.1); // Should be essentially zero
    }

    #[test]
    fn test_standalone_functions() {
        let f = TransitionMatrix::from_matrix(nalgebra::matrix![
            1.0, 0.0, 1.0, 0.0_f64;
            0.0, 1.0, 0.0, 1.0;
            0.0, 0.0, 1.0, 0.0;
            0.0, 0.0, 0.0, 1.0
        ]);
        let q = StateCovariance::from_matrix(nalgebra::SMatrix::<f64, 4, 4>::identity().scale(0.1));

        let state = KalmanState::new(
            StateVector::from_array([0.0, 0.0, 5.0, 3.0]),
            StateCovariance::identity(),
        );

        let predicted = predict(&state, &f, &q);
        assert!((predicted.mean.index(0) - 5.0).abs() < 1e-10);
        assert!((predicted.mean.index(1) - 3.0).abs() < 1e-10);

        let h = ObservationMatrix::from_matrix(nalgebra::matrix![
            1.0, 0.0, 0.0, 0.0_f64;
            0.0, 1.0, 0.0, 0.0
        ]);
        let r = MeasurementCovariance::from_matrix(nalgebra::SMatrix::<f64, 2, 2>::identity());

        let measurement = Measurement::from_array([5.0, 3.0]);
        let updated = update(&predicted, &measurement, &h, &r).unwrap();

        assert!((updated.mean.index(0) - 5.0).abs() < 1.0);
        assert!((updated.mean.index(1) - 3.0).abs() < 1.0);
    }

    #[test]
    fn test_position_extraction() {
        let state = KalmanState::new(
            StateVector::from_array([10.0_f64, 20.0, 1.0, 2.0]),
            StateCovariance::identity(),
        );

        let pos: [f64; 2] = state.position();
        assert!((pos[0] - 10.0).abs() < 1e-10);
        assert!((pos[1] - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_type_safety_compiles() {
        // This test verifies that the type system prevents mixing spaces.
        // The following would NOT compile (uncomment to verify):
        //
        // let state_vec: StateVector<f64, 4> = StateVector::from_array([0.0; 4]);
        // let meas_vec: Measurement<f64, 2> = Measurement::from_array([0.0; 2]);
        // let _ = state_vec + meas_vec; // ERROR: mismatched types
        //
        // The type system also ensures dimension compatibility:
        // let filter: KalmanFilter<f64, ConstantVelocity2D<f64>, PositionSensor3D<f64>, 4, 3>
        //     = KalmanFilter::new(...); // ERROR: PositionSensor3D expects 6-dim state

        // This compiles because dimensions match
        let _filter: KalmanFilter<f64, ConstantVelocity2D<f64>, PositionSensor2D<f64>, 4, 2> =
            KalmanFilter::new(
                ConstantVelocity2D::new(1.0, 0.99),
                PositionSensor2D::new(1.0, 0.95),
            );
    }
}
