//! Observation (sensor) models
//!
//! Describes how sensor measurements relate to target states.

use nalgebra::RealField;
use num_traits::Float;

use crate::types::spaces::{StateVector, MeasurementCovariance};
use crate::types::transforms::ObservationMatrix;

/// Trait for linear observation models.
///
/// Describes the measurement process:
/// z = H * x + v
///
/// where:
/// - H is the observation matrix
/// - v is zero-mean Gaussian measurement noise with covariance R
pub trait ObservationModel<T: RealField, const N: usize, const M: usize> {
    /// Returns the observation matrix.
    fn observation_matrix(&self) -> ObservationMatrix<T, M, N>;

    /// Returns the measurement noise covariance.
    fn measurement_noise(&self) -> MeasurementCovariance<T, M>;

    /// Returns the probability of detection for a target at the given state.
    ///
    /// This may depend on the target state (e.g., reduced detection at range).
    fn detection_probability(&self, state: &StateVector<T, N>) -> T;
}

// ============================================================================
// Common Observation Models
// ============================================================================

/// Position-only sensor in 2D.
///
/// Observes [x, y] from state [x, y, vx, vy]
#[derive(Debug, Clone)]
pub struct PositionSensor2D<T: RealField> {
    /// Position measurement noise standard deviation
    pub sigma_pos: T,
    /// Detection probability
    pub p_detection: T,
}

impl<T: RealField + Float + Copy> PositionSensor2D<T> {
    /// Creates a new position sensor.
    ///
    /// # Arguments
    /// - `sigma_pos`: Position measurement noise standard deviation (must be > 0)
    /// - `p_detection`: Probability of detecting a target (must be in [0, 1])
    ///
    /// # Panics
    /// Panics if `sigma_pos <= 0` or `p_detection` is not in [0, 1].
    pub fn new(sigma_pos: T, p_detection: T) -> Self {
        assert!(sigma_pos > T::zero(), "Measurement noise sigma_pos must be positive");
        assert!(p_detection >= T::zero() && p_detection <= T::one(),
            "Detection probability must be in [0, 1]");
        Self { sigma_pos, p_detection }
    }

    /// Creates a sensor with different noise in x and y.
    ///
    /// # Panics
    /// Panics if `sigma_x <= 0`, `sigma_y <= 0`, or `p_detection` is not in [0, 1].
    pub fn with_noise(sigma_x: T, sigma_y: T, p_detection: T) -> PositionSensor2DAsym<T> {
        assert!(sigma_x > T::zero(), "Measurement noise sigma_x must be positive");
        assert!(sigma_y > T::zero(), "Measurement noise sigma_y must be positive");
        assert!(p_detection >= T::zero() && p_detection <= T::one(),
            "Detection probability must be in [0, 1]");
        PositionSensor2DAsym { sigma_x, sigma_y, p_detection }
    }
}

impl<T: RealField + Float + Copy> ObservationModel<T, 4, 2> for PositionSensor2D<T> {
    fn observation_matrix(&self) -> ObservationMatrix<T, 2, 4> {
        let one = T::one();
        let zero = T::zero();

        ObservationMatrix::from_matrix(nalgebra::matrix![
            one, zero, zero, zero;
            zero, one, zero, zero
        ])
    }

    fn measurement_noise(&self) -> MeasurementCovariance<T, 2> {
        let sigma_sq = self.sigma_pos * self.sigma_pos;
        let zero = T::zero();

        MeasurementCovariance::from_matrix(nalgebra::matrix![
            sigma_sq, zero;
            zero, sigma_sq
        ])
    }

    fn detection_probability(&self, _state: &StateVector<T, 4>) -> T {
        self.p_detection
    }
}

/// Position sensor with asymmetric noise in 2D.
#[derive(Debug, Clone)]
pub struct PositionSensor2DAsym<T: RealField> {
    /// X position noise standard deviation
    pub sigma_x: T,
    /// Y position noise standard deviation
    pub sigma_y: T,
    /// Detection probability
    pub p_detection: T,
}

impl<T: RealField + Float + Copy> ObservationModel<T, 4, 2> for PositionSensor2DAsym<T> {
    fn observation_matrix(&self) -> ObservationMatrix<T, 2, 4> {
        let one = T::one();
        let zero = T::zero();

        ObservationMatrix::from_matrix(nalgebra::matrix![
            one, zero, zero, zero;
            zero, one, zero, zero
        ])
    }

    fn measurement_noise(&self) -> MeasurementCovariance<T, 2> {
        let zero = T::zero();
        let sigma_x_sq = self.sigma_x * self.sigma_x;
        let sigma_y_sq = self.sigma_y * self.sigma_y;

        MeasurementCovariance::from_matrix(nalgebra::matrix![
            sigma_x_sq, zero;
            zero, sigma_y_sq
        ])
    }

    fn detection_probability(&self, _state: &StateVector<T, 4>) -> T {
        self.p_detection
    }
}

/// Position-only sensor in 3D.
///
/// Observes [x, y, z] from state [x, y, z, vx, vy, vz]
#[derive(Debug, Clone)]
pub struct PositionSensor3D<T: RealField> {
    /// Position measurement noise standard deviation
    pub sigma_pos: T,
    /// Detection probability
    pub p_detection: T,
}

impl<T: RealField + Float + Copy> PositionSensor3D<T> {
    /// Creates a new position sensor.
    ///
    /// # Arguments
    /// - `sigma_pos`: Position measurement noise standard deviation (must be > 0)
    /// - `p_detection`: Probability of detecting a target (must be in [0, 1])
    ///
    /// # Panics
    /// Panics if `sigma_pos <= 0` or `p_detection` is not in [0, 1].
    pub fn new(sigma_pos: T, p_detection: T) -> Self {
        assert!(sigma_pos > T::zero(), "Measurement noise sigma_pos must be positive");
        assert!(p_detection >= T::zero() && p_detection <= T::one(),
            "Detection probability must be in [0, 1]");
        Self { sigma_pos, p_detection }
    }
}

impl<T: RealField + Float + Copy> ObservationModel<T, 6, 3> for PositionSensor3D<T> {
    fn observation_matrix(&self) -> ObservationMatrix<T, 3, 6> {
        let one = T::one();
        let zero = T::zero();

        ObservationMatrix::from_matrix(nalgebra::matrix![
            one, zero, zero, zero, zero, zero;
            zero, one, zero, zero, zero, zero;
            zero, zero, one, zero, zero, zero
        ])
    }

    fn measurement_noise(&self) -> MeasurementCovariance<T, 3> {
        let sigma_sq = self.sigma_pos * self.sigma_pos;
        let zero = T::zero();

        MeasurementCovariance::from_matrix(nalgebra::matrix![
            sigma_sq, zero, zero;
            zero, sigma_sq, zero;
            zero, zero, sigma_sq
        ])
    }

    fn detection_probability(&self, _state: &StateVector<T, 6>) -> T {
        self.p_detection
    }
}

/// Range-bearing sensor (radar-like).
///
/// This is a non-linear sensor model requiring EKF-style linearization.
/// Use `jacobian_at()` to get the observation matrix linearized at a specific state,
/// or `observe_nonlinear()` for the actual nonlinear measurement.
///
/// **Important**: The `ObservationModel` trait implementation is intentionally
/// not provided because this sensor requires state-dependent linearization.
/// Use the dedicated methods instead.
#[derive(Debug, Clone)]
pub struct RangeBearingSensor<T: RealField> {
    /// Range measurement noise standard deviation
    pub sigma_range: T,
    /// Bearing measurement noise standard deviation (radians)
    pub sigma_bearing: T,
    /// Detection probability
    pub p_detection: T,
    /// Sensor position x
    pub sensor_x: T,
    /// Sensor position y
    pub sensor_y: T,
}

impl<T: RealField + Float + Copy> RangeBearingSensor<T> {
    /// Creates a new range-bearing sensor at the origin.
    ///
    /// # Arguments
    /// - `sigma_range`: Range measurement noise standard deviation (must be > 0)
    /// - `sigma_bearing`: Bearing measurement noise standard deviation in radians (must be > 0)
    /// - `p_detection`: Probability of detecting a target (must be in [0, 1])
    ///
    /// # Panics
    /// Panics if noise parameters are non-positive or `p_detection` is not in [0, 1].
    pub fn new(sigma_range: T, sigma_bearing: T, p_detection: T) -> Self {
        assert!(sigma_range > T::zero(), "Range noise sigma_range must be positive");
        assert!(sigma_bearing > T::zero(), "Bearing noise sigma_bearing must be positive");
        assert!(p_detection >= T::zero() && p_detection <= T::one(),
            "Detection probability must be in [0, 1]");
        Self {
            sigma_range,
            sigma_bearing,
            p_detection,
            sensor_x: T::zero(),
            sensor_y: T::zero(),
        }
    }

    /// Creates a sensor at a specific position.
    ///
    /// # Arguments
    /// - `sigma_range`: Range measurement noise standard deviation (must be > 0)
    /// - `sigma_bearing`: Bearing measurement noise standard deviation in radians (must be > 0)
    /// - `p_detection`: Probability of detecting a target (must be in [0, 1])
    /// - `sensor_x`: X coordinate of sensor position
    /// - `sensor_y`: Y coordinate of sensor position
    ///
    /// # Panics
    /// Panics if noise parameters are non-positive or `p_detection` is not in [0, 1].
    pub fn at_position(sigma_range: T, sigma_bearing: T, p_detection: T, sensor_x: T, sensor_y: T) -> Self {
        assert!(sigma_range > T::zero(), "Range noise sigma_range must be positive");
        assert!(sigma_bearing > T::zero(), "Bearing noise sigma_bearing must be positive");
        assert!(p_detection >= T::zero() && p_detection <= T::one(),
            "Detection probability must be in [0, 1]");
        Self {
            sigma_range,
            sigma_bearing,
            p_detection,
            sensor_x,
            sensor_y,
        }
    }

    /// Returns the detection probability for a target at the given state.
    pub fn detection_probability(&self, _state: &StateVector<T, 4>) -> T {
        self.p_detection
    }

    /// Returns the measurement noise covariance in [range, bearing] space.
    pub fn measurement_noise(&self) -> MeasurementCovariance<T, 2> {
        let zero = T::zero();
        let sigma_r_sq = self.sigma_range * self.sigma_range;
        let sigma_b_sq = self.sigma_bearing * self.sigma_bearing;

        MeasurementCovariance::from_matrix(nalgebra::matrix![
            sigma_r_sq, zero;
            zero, sigma_b_sq
        ])
    }

    /// Computes the nonlinear measurement [range, bearing] for a given state.
    pub fn observe_nonlinear(&self, state: &StateVector<T, 4>) -> (T, T) {
        let dx = *state.index(0) - self.sensor_x;
        let dy = *state.index(1) - self.sensor_y;

        let range = num_traits::Float::sqrt(dx * dx + dy * dy);
        let bearing = num_traits::Float::atan2(dy, dx);

        (range, bearing)
    }

    /// Computes the Jacobian of the measurement function at a given state.
    ///
    /// This is the observation matrix H linearized around the given state,
    /// suitable for use in an Extended Kalman Filter (EKF).
    ///
    /// Returns `None` if the state is too close to the sensor position.
    pub fn jacobian_at(&self, state: &StateVector<T, 4>) -> Option<ObservationMatrix<T, 2, 4>> {
        let dx = *state.index(0) - self.sensor_x;
        let dy = *state.index(1) - self.sensor_y;

        let r_sq = dx * dx + dy * dy;
        let r = num_traits::Float::sqrt(r_sq);

        let zero = T::zero();

        if r < T::from_f64(1e-10).unwrap() {
            // Target at sensor position - Jacobian undefined
            return None;
        }

        // Jacobian: d[range, bearing]/d[x, y, vx, vy]
        // d_range/dx = dx/r, d_range/dy = dy/r
        // d_bearing/dx = -dy/r^2, d_bearing/dy = dx/r^2
        Some(ObservationMatrix::from_matrix(nalgebra::matrix![
            dx / r, dy / r, zero, zero;
            -dy / r_sq, dx / r_sq, zero, zero
        ]))
    }

    /// Legacy alias for `jacobian_at`. Prefer `jacobian_at` for clarity.
    #[deprecated(since = "0.2.0", note = "Use jacobian_at() instead which returns Option")]
    pub fn jacobian(&self, state: &StateVector<T, 4>) -> ObservationMatrix<T, 2, 4> {
        self.jacobian_at(state).unwrap_or_else(ObservationMatrix::zeros)
    }

    /// Legacy alias for `observe_nonlinear`. Prefer `observe_nonlinear` for clarity.
    #[deprecated(since = "0.2.0", note = "Use observe_nonlinear() instead")]
    pub fn measure(&self, state: &StateVector<T, 4>) -> (T, T) {
        self.observe_nonlinear(state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_sensor_2d() {
        let sensor = PositionSensor2D::new(1.0_f64, 0.9);
        let state = StateVector::from_array([10.0, 20.0, 1.0, 2.0]);

        let h = sensor.observation_matrix();
        let z = h.observe(&state);

        assert!((z.index(0) - 10.0).abs() < 1e-10);
        assert!((z.index(1) - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_detection_probability() {
        let sensor = PositionSensor2D::new(1.0_f64, 0.95);
        let state = StateVector::from_array([0.0, 0.0, 0.0, 0.0]);

        assert!((sensor.detection_probability(&state) - 0.95).abs() < 1e-10);
    }

    #[test]
    fn test_range_bearing_sensor() {
        let sensor = RangeBearingSensor::new(1.0_f64, 0.01, 0.9);
        let state = StateVector::from_array([10.0, 0.0, 0.0, 0.0]);

        let (range, bearing) = sensor.observe_nonlinear(&state);

        assert!((range - 10.0).abs() < 1e-10);
        assert!(bearing.abs() < 1e-10); // Should be 0 for positive x-axis
    }

    #[test]
    fn test_range_bearing_jacobian() {
        let sensor = RangeBearingSensor::new(1.0_f64, 0.01, 0.9);
        let state = StateVector::from_array([10.0, 0.0, 0.0, 0.0]);

        let jacobian = sensor.jacobian_at(&state).unwrap();

        // For target at (10, 0), range=10, bearing=0
        // d_range/dx = 1.0, d_range/dy = 0.0
        // d_bearing/dx = 0.0, d_bearing/dy = 0.1 (1/r)
        assert!((jacobian.as_matrix()[(0, 0)] - 1.0).abs() < 1e-10);
        assert!((jacobian.as_matrix()[(0, 1)] - 0.0).abs() < 1e-10);
        assert!((jacobian.as_matrix()[(1, 0)] - 0.0).abs() < 1e-10);
        assert!((jacobian.as_matrix()[(1, 1)] - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_range_bearing_at_sensor_position() {
        let sensor = RangeBearingSensor::new(1.0_f64, 0.01, 0.9);
        let state = StateVector::from_array([0.0, 0.0, 0.0, 0.0]);

        // Jacobian should be None when target is at sensor position
        assert!(sensor.jacobian_at(&state).is_none());
    }
}
