//! Transition (motion) models for target dynamics
//!
//! Describes how targets evolve over time.

use nalgebra::RealField;
use num_traits::Float;

use crate::types::spaces::{StateCovariance, StateVector};
use crate::types::transforms::TransitionMatrix;

/// Trait for linear transition (motion) models.
///
/// Describes target dynamics in the form:
/// x_{k+1} = F * x_k + w
///
/// where:
/// - F is the state transition matrix
/// - w is zero-mean Gaussian process noise with covariance Q
pub trait TransitionModel<T: RealField, const N: usize> {
    /// Returns the state transition matrix for time step dt.
    fn transition_matrix(&self, dt: T) -> TransitionMatrix<T, N>;

    /// Returns the process noise covariance for time step dt.
    fn process_noise(&self, dt: T) -> StateCovariance<T, N>;

    /// Returns the probability that a target survives from one time step to the next.
    ///
    /// This may depend on the target state (e.g., targets leaving a surveillance region).
    fn survival_probability(&self, state: &StateVector<T, N>) -> T;
}

// ============================================================================
// Common Transition Models
// ============================================================================

/// Constant velocity model in 2D.
///
/// State: [x, y, vx, vy]
#[derive(Debug, Clone)]
pub struct ConstantVelocity2D<T: RealField> {
    /// Process noise intensity (acceleration variance)
    pub sigma_a: T,
    /// Survival probability
    pub p_survival: T,
}

impl<T: RealField + Float + Copy> ConstantVelocity2D<T> {
    /// Creates a new constant velocity model.
    ///
    /// # Arguments
    /// - `sigma_a`: Process noise intensity / acceleration standard deviation (must be >= 0)
    /// - `p_survival`: Probability that a target survives to the next time step (must be in [0, 1])
    ///
    /// # Panics
    /// Panics if `sigma_a < 0` or `p_survival` is not in [0, 1].
    pub fn new(sigma_a: T, p_survival: T) -> Self {
        assert!(
            sigma_a >= T::zero(),
            "Process noise sigma_a must be non-negative"
        );
        assert!(
            p_survival >= T::zero() && p_survival <= T::one(),
            "Survival probability must be in [0, 1]"
        );
        Self {
            sigma_a,
            p_survival,
        }
    }
}

impl<T: RealField + Float + Copy> TransitionModel<T, 4> for ConstantVelocity2D<T> {
    fn transition_matrix(&self, dt: T) -> TransitionMatrix<T, 4> {
        assert!(dt >= T::zero(), "Time step dt must be non-negative");
        let one = T::one();
        let zero = T::zero();

        TransitionMatrix::from_matrix(nalgebra::matrix![
            one, zero, dt, zero;
            zero, one, zero, dt;
            zero, zero, one, zero;
            zero, zero, zero, one
        ])
    }

    fn process_noise(&self, dt: T) -> StateCovariance<T, 4> {
        assert!(dt >= T::zero(), "Time step dt must be non-negative");
        let dt2 = dt * dt;
        let dt3 = dt2 * dt;
        let dt4 = dt3 * dt;

        let two = T::from_f64(2.0).unwrap();
        let four = T::from_f64(4.0).unwrap();

        let sigma_sq = self.sigma_a * self.sigma_a;

        // Discrete white noise acceleration model
        let q11 = dt4 / four * sigma_sq;
        let q13 = dt3 / two * sigma_sq;
        let q33 = dt2 * sigma_sq;

        let zero = T::zero();

        StateCovariance::from_matrix(nalgebra::matrix![
            q11, zero, q13, zero;
            zero, q11, zero, q13;
            q13, zero, q33, zero;
            zero, q13, zero, q33
        ])
    }

    fn survival_probability(&self, _state: &StateVector<T, 4>) -> T {
        self.p_survival
    }
}

/// Constant velocity model in 3D.
///
/// State: [x, y, z, vx, vy, vz]
#[derive(Debug, Clone)]
pub struct ConstantVelocity3D<T: RealField> {
    /// Process noise intensity (acceleration variance)
    pub sigma_a: T,
    /// Survival probability
    pub p_survival: T,
}

impl<T: RealField + Float + Copy> ConstantVelocity3D<T> {
    /// Creates a new constant velocity model.
    ///
    /// # Arguments
    /// - `sigma_a`: Process noise intensity / acceleration standard deviation (must be >= 0)
    /// - `p_survival`: Probability that a target survives to the next time step (must be in [0, 1])
    ///
    /// # Panics
    /// Panics if `sigma_a < 0` or `p_survival` is not in [0, 1].
    pub fn new(sigma_a: T, p_survival: T) -> Self {
        assert!(
            sigma_a >= T::zero(),
            "Process noise sigma_a must be non-negative"
        );
        assert!(
            p_survival >= T::zero() && p_survival <= T::one(),
            "Survival probability must be in [0, 1]"
        );
        Self {
            sigma_a,
            p_survival,
        }
    }
}

impl<T: RealField + Float + Copy> TransitionModel<T, 6> for ConstantVelocity3D<T> {
    fn transition_matrix(&self, dt: T) -> TransitionMatrix<T, 6> {
        assert!(dt >= T::zero(), "Time step dt must be non-negative");
        let one = T::one();
        let zero = T::zero();

        TransitionMatrix::from_matrix(nalgebra::matrix![
            one, zero, zero, dt, zero, zero;
            zero, one, zero, zero, dt, zero;
            zero, zero, one, zero, zero, dt;
            zero, zero, zero, one, zero, zero;
            zero, zero, zero, zero, one, zero;
            zero, zero, zero, zero, zero, one
        ])
    }

    fn process_noise(&self, dt: T) -> StateCovariance<T, 6> {
        assert!(dt >= T::zero(), "Time step dt must be non-negative");
        let dt2 = dt * dt;
        let dt3 = dt2 * dt;
        let dt4 = dt3 * dt;

        let two = T::from_f64(2.0).unwrap();
        let four = T::from_f64(4.0).unwrap();

        let sigma_sq = self.sigma_a * self.sigma_a;

        let q11 = dt4 / four * sigma_sq;
        let q14 = dt3 / two * sigma_sq;
        let q44 = dt2 * sigma_sq;

        let zero = T::zero();

        StateCovariance::from_matrix(nalgebra::matrix![
            q11, zero, zero, q14, zero, zero;
            zero, q11, zero, zero, q14, zero;
            zero, zero, q11, zero, zero, q14;
            q14, zero, zero, q44, zero, zero;
            zero, q14, zero, zero, q44, zero;
            zero, zero, q14, zero, zero, q44
        ])
    }

    fn survival_probability(&self, _state: &StateVector<T, 6>) -> T {
        self.p_survival
    }
}

/// Nearly constant turn rate model in 2D.
///
/// State: [x, y, vx, vy, omega] where omega is the turn rate (rad/s).
///
/// This model assumes the target moves with constant speed and turn rate.
/// The nonlinear dynamics are:
/// - x' = x + (v/ω)[sin(θ + ωΔt) - sin(θ)]
/// - y' = y + (v/ω)[cos(θ) - cos(θ + ωΔt)]
/// - vx' = v·cos(θ + ωΔt)
/// - vy' = v·sin(θ + ωΔt)
/// - ω' = ω
///
/// where θ = atan2(vy, vx) and v = sqrt(vx² + vy²).
///
/// **Important**: This is a nonlinear model. The `transition_matrix()` method
/// returns a linearization around zero turn rate for use with the standard
/// Kalman filter. For proper tracking with significant turn rates, use an
/// Extended Kalman Filter (EKF) with `transition_matrix_at()`.
#[derive(Debug, Clone)]
pub struct CoordinatedTurn2D<T: RealField> {
    /// Process noise intensity for linear acceleration
    pub sigma_a: T,
    /// Process noise intensity for turn rate acceleration
    pub sigma_omega: T,
    /// Survival probability
    pub p_survival: T,
}

impl<T: RealField + Float + Copy> CoordinatedTurn2D<T> {
    /// Creates a new coordinated turn model.
    ///
    /// # Arguments
    /// - `sigma_a`: Process noise intensity for linear acceleration (must be >= 0)
    /// - `sigma_omega`: Process noise intensity for turn rate (must be >= 0)
    /// - `p_survival`: Probability that a target survives to the next time step (must be in [0, 1])
    ///
    /// # Panics
    /// Panics if noise parameters are negative or `p_survival` is not in [0, 1].
    pub fn new(sigma_a: T, sigma_omega: T, p_survival: T) -> Self {
        assert!(
            sigma_a >= T::zero(),
            "Process noise sigma_a must be non-negative"
        );
        assert!(
            sigma_omega >= T::zero(),
            "Process noise sigma_omega must be non-negative"
        );
        assert!(
            p_survival >= T::zero() && p_survival <= T::one(),
            "Survival probability must be in [0, 1]"
        );
        Self {
            sigma_a,
            sigma_omega,
            p_survival,
        }
    }

    /// Applies the nonlinear coordinated turn dynamics.
    ///
    /// Returns the predicted state after time dt.
    ///
    /// # Panics
    /// Panics if `dt < 0`.
    pub fn predict_nonlinear(&self, state: &StateVector<T, 5>, dt: T) -> StateVector<T, 5> {
        assert!(dt >= T::zero(), "Time step dt must be non-negative");
        let x = *state.index(0);
        let y = *state.index(1);
        let vx = *state.index(2);
        let vy = *state.index(3);
        let omega = *state.index(4);

        let omega_dt = omega * dt;
        let eps = T::from_f64(1e-10).unwrap();

        if num_traits::Float::abs(omega) < eps {
            // Nearly zero turn rate - use constant velocity
            StateVector::from_array([x + vx * dt, y + vy * dt, vx, vy, omega])
        } else {
            // Apply coordinated turn dynamics
            // The rotation matrix for velocity is:
            // [cos(ωΔt)  -sin(ωΔt)]
            // [sin(ωΔt)   cos(ωΔt)]
            //
            // Position change is the integral of velocity over the turn:
            // Δx = (1/ω)[vx·sin(ωΔt) - vy·(cos(ωΔt) - 1)]
            //    = (1/ω)[vx·sin(ωΔt) + vy·(1 - cos(ωΔt))]
            // Δy = (1/ω)[vx·(1 - cos(ωΔt)) + vy·sin(ωΔt)]
            let sin_omega_dt = num_traits::Float::sin(omega_dt);
            let cos_omega_dt = num_traits::Float::cos(omega_dt);
            let one_minus_cos = T::one() - cos_omega_dt;

            // New position
            let x_new = x + (vx * sin_omega_dt + vy * one_minus_cos) / omega;
            let y_new = y + (vx * one_minus_cos + vy * sin_omega_dt) / omega;

            // New velocity (rotation of velocity vector)
            let vx_new = vx * cos_omega_dt - vy * sin_omega_dt;
            let vy_new = vx * sin_omega_dt + vy * cos_omega_dt;

            StateVector::from_array([x_new, y_new, vx_new, vy_new, omega])
        }
    }

    /// Computes the Jacobian of the transition function at a given state.
    ///
    /// This is the transition matrix linearized around the given state,
    /// suitable for use in an Extended Kalman Filter (EKF).
    ///
    /// # Panics
    /// Panics if `dt < 0`.
    pub fn transition_matrix_at(&self, state: &StateVector<T, 5>, dt: T) -> TransitionMatrix<T, 5> {
        assert!(dt >= T::zero(), "Time step dt must be non-negative");
        let vx = *state.index(2);
        let vy = *state.index(3);
        let omega = *state.index(4);

        let omega_dt = omega * dt;
        let eps = T::from_f64(1e-10).unwrap();
        let one = T::one();
        let zero = T::zero();

        if num_traits::Float::abs(omega) < eps {
            // Jacobian for constant velocity (ω ≈ 0)
            TransitionMatrix::from_matrix(nalgebra::matrix![
                one, zero, dt, zero, zero;
                zero, one, zero, dt, zero;
                zero, zero, one, zero, zero;
                zero, zero, zero, one, zero;
                zero, zero, zero, zero, one
            ])
        } else {
            let sin_omega_dt = num_traits::Float::sin(omega_dt);
            let cos_omega_dt = num_traits::Float::cos(omega_dt);
            let one_minus_cos = one - cos_omega_dt;
            let omega_sq = omega * omega;

            // Partial derivatives for position w.r.t. velocity
            // x_new = x + (vx * sin + vy * (1-cos)) / omega
            // y_new = y + (vx * (1-cos) + vy * sin) / omega
            let dx_dvx = sin_omega_dt / omega;
            let dx_dvy = one_minus_cos / omega;
            let dy_dvx = one_minus_cos / omega;
            let dy_dvy = sin_omega_dt / omega;

            // Partial derivatives w.r.t. omega (using quotient rule)
            // For x: d/dω [(vx*sin(ωt) + vy*(1-cos(ωt)))/ω]
            let dx_domega = (vx * (omega_dt * cos_omega_dt - sin_omega_dt)
                + vy * (omega_dt * sin_omega_dt - one_minus_cos))
                / omega_sq;
            // For y: d/dω [(vx*(1-cos(ωt)) + vy*sin(ωt))/ω]
            let dy_domega = (vx * (omega_dt * sin_omega_dt - one_minus_cos)
                + vy * (omega_dt * cos_omega_dt - sin_omega_dt))
                / omega_sq;

            // Partial derivatives for velocity w.r.t. omega
            // vx_new = vx*cos(ωt) - vy*sin(ωt)
            // vy_new = vx*sin(ωt) + vy*cos(ωt)
            let dvx_domega = -vx * dt * sin_omega_dt - vy * dt * cos_omega_dt;
            let dvy_domega = vx * dt * cos_omega_dt - vy * dt * sin_omega_dt;

            TransitionMatrix::from_matrix(nalgebra::matrix![
                one, zero, dx_dvx, dx_dvy, dx_domega;
                zero, one, dy_dvx, dy_dvy, dy_domega;
                zero, zero, cos_omega_dt, -sin_omega_dt, dvx_domega;
                zero, zero, sin_omega_dt, cos_omega_dt, dvy_domega;
                zero, zero, zero, zero, one
            ])
        }
    }
}

impl<T: RealField + Float + Copy> TransitionModel<T, 5> for CoordinatedTurn2D<T> {
    fn transition_matrix(&self, dt: T) -> TransitionMatrix<T, 5> {
        assert!(dt >= T::zero(), "Time step dt must be non-negative");
        // Default linearization around zero turn rate
        // For proper EKF usage, call transition_matrix_at() with the current state
        let one = T::one();
        let zero = T::zero();

        TransitionMatrix::from_matrix(nalgebra::matrix![
            one, zero, dt, zero, zero;
            zero, one, zero, dt, zero;
            zero, zero, one, zero, zero;
            zero, zero, zero, one, zero;
            zero, zero, zero, zero, one
        ])
    }

    fn process_noise(&self, dt: T) -> StateCovariance<T, 5> {
        assert!(dt >= T::zero(), "Time step dt must be non-negative");
        let dt2 = dt * dt;
        let dt3 = dt2 * dt;
        let dt4 = dt3 * dt;

        let two = T::from_f64(2.0).unwrap();
        let four = T::from_f64(4.0).unwrap();

        let sigma_a_sq = self.sigma_a * self.sigma_a;
        let sigma_omega_sq = self.sigma_omega * self.sigma_omega;
        let zero = T::zero();

        // Discrete white noise acceleration model for position/velocity
        let q11 = dt4 / four * sigma_a_sq;
        let q13 = dt3 / two * sigma_a_sq;
        let q33 = dt2 * sigma_a_sq;

        // Turn rate noise
        let q55 = dt2 * sigma_omega_sq;

        StateCovariance::from_matrix(nalgebra::matrix![
            q11, zero, q13, zero, zero;
            zero, q11, zero, q13, zero;
            q13, zero, q33, zero, zero;
            zero, q13, zero, q33, zero;
            zero, zero, zero, zero, q55
        ])
    }

    fn survival_probability(&self, _state: &StateVector<T, 5>) -> T {
        self.p_survival
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_velocity_2d() {
        let model = ConstantVelocity2D::new(1.0_f64, 0.99);
        let dt = 1.0;

        let f = model.transition_matrix(dt);
        let state = StateVector::from_array([0.0, 0.0, 1.0, 2.0]);
        let next_state = f.apply_state(&state);

        assert!((next_state.index(0) - 1.0).abs() < 1e-10);
        assert!((next_state.index(1) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_survival_probability() {
        let model = ConstantVelocity2D::new(1.0_f64, 0.95);
        let state = StateVector::from_array([0.0, 0.0, 1.0, 2.0]);

        assert!((model.survival_probability(&state) - 0.95).abs() < 1e-10);
    }

    #[test]
    fn test_coordinated_turn_straight() {
        // With zero turn rate, should behave like constant velocity
        let model = CoordinatedTurn2D::new(1.0_f64, 0.1, 0.99);
        let state = StateVector::from_array([0.0, 0.0, 10.0, 0.0, 0.0]);
        let dt = 1.0;

        let predicted = model.predict_nonlinear(&state, dt);

        assert!((predicted.index(0) - 10.0).abs() < 1e-10);
        assert!((predicted.index(1) - 0.0).abs() < 1e-10);
        assert!((predicted.index(2) - 10.0).abs() < 1e-10);
        assert!((predicted.index(3) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_coordinated_turn_90_degrees() {
        // 90 degree turn: omega = pi/2, dt = 1
        use std::f64::consts::FRAC_PI_2;

        let model = CoordinatedTurn2D::new(1.0_f64, 0.1, 0.99);
        // Moving east at 10 m/s, turning left (counter-clockwise) at pi/2 rad/s
        let state = StateVector::from_array([0.0, 0.0, 10.0, 0.0, FRAC_PI_2]);
        let dt = 1.0;

        let predicted = model.predict_nonlinear(&state, dt);

        // After 90 degree CCW turn starting with velocity (+vx, 0):
        // The turn radius is r = v/ω = 10/(π/2) ≈ 6.37
        // Center of turn is at (0, r) since we're turning left
        // After 90 degrees, we're at position (r, r) relative to start
        // x_new = (vx*sin(ωt) + vy*(1-cos(ωt)))/ω = 10*1/(π/2) ≈ 6.37
        // y_new = (vx*(1-cos(ωt)) + vy*sin(ωt))/ω = 10*1/(π/2) ≈ 6.37
        // Velocity is now pointing north: (0, 10)
        let r = 10.0 / FRAC_PI_2; // turn radius ≈ 6.37

        assert!(
            (predicted.index(0) - r).abs() < 1e-6,
            "x: {} vs {}",
            predicted.index(0),
            r
        );
        assert!(
            (predicted.index(1) - r).abs() < 1e-6,
            "y: {} vs {}",
            predicted.index(1),
            r
        );
        assert!(
            (predicted.index(2) - 0.0).abs() < 1e-6,
            "vx: {} should be ~0",
            predicted.index(2)
        );
        assert!(
            (predicted.index(3) - 10.0).abs() < 1e-6,
            "vy: {} should be ~10",
            predicted.index(3)
        );
    }

    #[test]
    fn test_coordinated_turn_jacobian_vs_numerical() {
        use std::f64::consts::FRAC_PI_4;

        let model = CoordinatedTurn2D::new(1.0_f64, 0.1, 0.99);
        let state = StateVector::from_array([5.0, 3.0, 8.0, 4.0, FRAC_PI_4]);
        let dt = 0.5;

        let jacobian = model.transition_matrix_at(&state, dt);

        // Numerical differentiation check for a few entries
        let eps = 1e-6;

        // Check df/dvx (column 2)
        let state_plus = StateVector::from_array([5.0, 3.0, 8.0 + eps, 4.0, FRAC_PI_4]);
        let state_minus = StateVector::from_array([5.0, 3.0, 8.0 - eps, 4.0, FRAC_PI_4]);
        let f_plus = model.predict_nonlinear(&state_plus, dt);
        let f_minus = model.predict_nonlinear(&state_minus, dt);

        let numerical_dx_dvx = (f_plus.index(0) - f_minus.index(0)) / (2.0 * eps);
        let analytical_dx_dvx = jacobian.as_matrix()[(0, 2)];

        assert!(
            (numerical_dx_dvx - analytical_dx_dvx).abs() < 1e-4,
            "dx/dvx: numerical {} vs analytical {}",
            numerical_dx_dvx,
            analytical_dx_dvx
        );
    }
}
