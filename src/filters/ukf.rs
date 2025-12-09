//! Unscented Kalman Filter (UKF) for single-target tracking with nonlinear models
//!
//! The UKF uses the unscented transform to propagate mean and covariance through
//! nonlinear functions without requiring Jacobian computation. This makes it more
//! accurate than the EKF for highly nonlinear systems.
//!
//! # Algorithm
//!
//! The UKF uses a set of carefully chosen sample points (sigma points) that capture
//! the mean and covariance of the state distribution. These points are propagated
//! through the nonlinear functions, and the output statistics are recovered.
//!
//! # Sigma Point Selection
//!
//! This implementation uses the symmetric sigma point selection:
//! - χ₀ = μ (mean)
//! - χᵢ = μ + √((n+λ)P)ᵢ for i = 1...n
//! - χᵢ₊ₙ = μ - √((n+λ)P)ᵢ for i = 1...n
//!
//! where λ = α²(n+κ) - n is the scaling parameter.
//!
//! # Type Safety
//!
//! The filter uses phantom types and const generics to ensure:
//! - State vectors cannot be mixed with measurements
//! - Dimension mismatches are caught at compile time
//!
//! # Example
//!
//! ```
//! use tracktor::filters::ukf::{UnscentedKalmanFilter, UkfState, UkfParams};
//! use tracktor::models::{CoordinatedTurn2D, RangeBearingSensor};
//! use tracktor::types::spaces::{StateVector, StateCovariance, Measurement};
//!
//! // Create nonlinear models
//! let transition = CoordinatedTurn2D::new(1.0, 0.1, 0.99);
//! // Note: RangeBearingSensor is for 4D state [x,y,vx,vy], use a custom sensor for 5D
//! let sensor = RangeBearingSensor::new(10.0, 0.01, 0.95);
//!
//! // For 5D coordinated turn, you need a 5D observation model
//! // This example shows the pattern for 4D state with constant velocity
//! ```

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use core::marker::PhantomData;

use nalgebra::{RealField, SMatrix, SVector};
use num_traits::Float;

use crate::models::{NonlinearObservationModel, NonlinearTransitionModel, TransitionModel};
use crate::types::spaces::{Measurement, StateCovariance, StateVector};

// Re-export KalmanState as UkfState for consistency
pub use super::kalman::KalmanState as UkfState;

// ============================================================================
// UKF Parameters
// ============================================================================

/// Parameters for the Unscented Kalman Filter.
///
/// These parameters control the sigma point spread and weighting.
///
/// # Common Parameter Choices
///
/// - **Standard UKF**: α=1e-3, β=2, κ=0
/// - **Scaled UKF**: α=1, β=2, κ=3-n (for Gaussian, gives same performance as cubature)
/// - **Van der Merwe**: α=1e-3, β=2, κ=0 (good default)
#[derive(Debug, Clone, Copy)]
pub struct UkfParams<T: RealField> {
    /// Primary scaling parameter (controls sigma point spread)
    ///
    /// Typical values: 1e-4 ≤ α ≤ 1
    /// Smaller α puts sigma points closer to the mean.
    pub alpha: T,

    /// Secondary scaling parameter (incorporates prior knowledge of distribution)
    ///
    /// For Gaussian distributions, β=2 is optimal.
    pub beta: T,

    /// Tertiary scaling parameter
    ///
    /// κ ≥ 0 ensures positive semi-definiteness.
    /// Common choices: κ=0 or κ=3-n
    pub kappa: T,
}

impl<T: RealField + Float> Default for UkfParams<T> {
    fn default() -> Self {
        Self {
            alpha: T::from_f64(1e-3).unwrap(),
            beta: T::from_f64(2.0).unwrap(),
            kappa: T::zero(),
        }
    }
}

impl<T: RealField + Float + Copy> UkfParams<T> {
    /// Creates new UKF parameters.
    ///
    /// # Panics
    /// Panics if α ≤ 0.
    pub fn new(alpha: T, beta: T, kappa: T) -> Self {
        assert!(alpha > T::zero(), "Alpha must be positive");
        Self { alpha, beta, kappa }
    }

    /// Computes the scaling parameter λ = α²(n + κ) - n
    #[inline]
    fn lambda(&self, n: usize) -> T {
        let n_t = T::from_usize(n).unwrap();
        self.alpha * self.alpha * (n_t + self.kappa) - n_t
    }

    /// Computes γ = √(n + λ) used for sigma point generation
    #[inline]
    fn gamma(&self, n: usize) -> T {
        let n_t = T::from_usize(n).unwrap();
        Float::sqrt(n_t + self.lambda(n))
    }

    /// Computes the weight for the mean of the central sigma point
    #[inline]
    fn weight_mean_0(&self, n: usize) -> T {
        let n_t = T::from_usize(n).unwrap();
        self.lambda(n) / (n_t + self.lambda(n))
    }

    /// Computes the weight for the covariance of the central sigma point
    #[inline]
    fn weight_cov_0(&self, n: usize) -> T {
        let n_t = T::from_usize(n).unwrap();
        self.lambda(n) / (n_t + self.lambda(n)) + (T::one() - self.alpha * self.alpha + self.beta)
    }

    /// Computes the weight for non-central sigma points (same for mean and covariance)
    #[inline]
    fn weight_i(&self, n: usize) -> T {
        let n_t = T::from_usize(n).unwrap();
        T::one() / (T::from_f64(2.0).unwrap() * (n_t + self.lambda(n)))
    }
}

// ============================================================================
// Sigma Points
// ============================================================================

/// Collection of sigma points with their weights.
///
/// For an n-dimensional state, there are 2n+1 sigma points.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct SigmaPoints<T: RealField, const N: usize> {
    /// The sigma points: [χ₀, χ₁, ..., χ₂ₙ]
    pub points: Vec<StateVector<T, N>>,
    /// Weight for mean calculation of central point
    pub weight_mean_0: T,
    /// Weight for covariance calculation of central point
    pub weight_cov_0: T,
    /// Weight for mean and covariance of other points
    pub weight_i: T,
}

#[cfg(feature = "alloc")]
impl<T: RealField + Float + Copy, const N: usize> SigmaPoints<T, N> {
    /// Generates sigma points from a state estimate.
    ///
    /// Uses the symmetric sigma point selection with Cholesky decomposition.
    ///
    /// # Returns
    /// `None` if the covariance matrix is not positive definite.
    pub fn generate(state: &UkfState<T, N>, params: &UkfParams<T>) -> Option<Self> {
        let gamma = params.gamma(N);

        // Cholesky decomposition: P = L * L^T
        let sqrt_p = state.covariance.cholesky()?;

        // Scale by gamma: √((n+λ)P)
        let scaled_sqrt_p = sqrt_p.scale(gamma);

        // Generate sigma points: 2*N + 1 total
        let num_points = 2 * N + 1;
        let mut points = Vec::with_capacity(num_points);

        // χ₀ = μ
        points.push(state.mean);

        // χᵢ = μ + column_i(scaled_sqrt_p) for i = 1...n
        // χᵢ₊ₙ = μ - column_i(scaled_sqrt_p) for i = 1...n
        for i in 0..N {
            let col = scaled_sqrt_p.column(i);
            let offset = StateVector::from_svector(col.into_owned());

            points.push(StateVector::from_svector(
                state.mean.as_svector() + offset.as_svector(),
            ));
            points.push(StateVector::from_svector(
                state.mean.as_svector() - offset.as_svector(),
            ));
        }

        Some(Self {
            points,
            weight_mean_0: params.weight_mean_0(N),
            weight_cov_0: params.weight_cov_0(N),
            weight_i: params.weight_i(N),
        })
    }

    /// Recovers the mean from transformed sigma points.
    pub fn recover_mean<const D: usize, F>(&self, transform: F) -> SVector<T, D>
    where
        F: Fn(&StateVector<T, N>) -> SVector<T, D>,
    {
        let mut mean = transform(&self.points[0]).scale(self.weight_mean_0);

        for point in self.points.iter().skip(1) {
            mean += transform(point).scale(self.weight_i);
        }

        mean
    }

    /// Recovers the mean and covariance from transformed sigma points.
    pub fn recover_mean_cov<const D: usize, F>(
        &self,
        transform: F,
        additive_noise: Option<&SMatrix<T, D, D>>,
    ) -> (SVector<T, D>, SMatrix<T, D, D>)
    where
        F: Fn(&StateVector<T, N>) -> SVector<T, D>,
    {
        // Transform all sigma points
        let transformed: Vec<SVector<T, D>> = self.points.iter().map(transform).collect();

        // Compute mean
        let mut mean = transformed[0].scale(self.weight_mean_0);
        for t in transformed.iter().skip(1) {
            mean += t.scale(self.weight_i);
        }

        // Compute covariance
        let diff0 = transformed[0] - mean;
        let mut cov = (diff0 * diff0.transpose()).scale(self.weight_cov_0);

        for t in transformed.iter().skip(1) {
            let diff = t - mean;
            cov += (diff * diff.transpose()).scale(self.weight_i);
        }

        // Add noise if provided
        if let Some(noise) = additive_noise {
            cov += noise;
        }

        (mean, cov)
    }

    /// Recovers the cross-covariance between state and transformed sigma points.
    pub fn cross_covariance<const D: usize, F>(
        &self,
        state_mean: &SVector<T, N>,
        transform: F,
        transformed_mean: &SVector<T, D>,
    ) -> SMatrix<T, N, D>
    where
        F: Fn(&StateVector<T, N>) -> SVector<T, D>,
    {
        let state_diff0 = self.points[0].as_svector() - state_mean;
        let trans_diff0 = transform(&self.points[0]) - transformed_mean;
        let mut cross_cov = (state_diff0 * trans_diff0.transpose()).scale(self.weight_cov_0);

        for point in self.points.iter().skip(1) {
            let state_diff = point.as_svector() - state_mean;
            let trans_diff = transform(point) - transformed_mean;
            cross_cov += (state_diff * trans_diff.transpose()).scale(self.weight_i);
        }

        cross_cov
    }
}

// ============================================================================
// Unscented Kalman Filter
// ============================================================================

/// An Unscented Kalman Filter for nonlinear systems.
///
/// The UKF propagates sigma points through nonlinear functions and recovers
/// statistics without requiring Jacobian computation.
///
/// # Type Parameters
///
/// - `T`: Scalar type
/// - `Trans`: Transition model type (must implement [`NonlinearTransitionModel`])
/// - `Obs`: Observation model type (must implement [`NonlinearObservationModel`])
/// - `N`: State dimension
/// - `M`: Measurement dimension
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct UnscentedKalmanFilter<T, Trans, Obs, const N: usize, const M: usize>
where
    T: RealField,
    Trans: NonlinearTransitionModel<T, N>,
    Obs: NonlinearObservationModel<T, N, M>,
{
    /// Nonlinear transition (motion) model
    pub transition: Trans,
    /// Nonlinear observation (sensor) model
    pub observation: Obs,
    /// UKF parameters
    pub params: UkfParams<T>,
    /// Phantom marker for scalar type
    _marker: PhantomData<T>,
}

#[cfg(feature = "alloc")]
impl<T, Trans, Obs, const N: usize, const M: usize> UnscentedKalmanFilter<T, Trans, Obs, N, M>
where
    T: RealField + Float + Copy,
    Trans: NonlinearTransitionModel<T, N>,
    Obs: NonlinearObservationModel<T, N, M>,
{
    /// Creates a new Unscented Kalman Filter.
    #[inline]
    pub fn new(transition: Trans, observation: Obs, params: UkfParams<T>) -> Self {
        Self {
            transition,
            observation,
            params,
            _marker: PhantomData,
        }
    }

    /// Creates a new UKF with default parameters.
    #[inline]
    pub fn with_default_params(transition: Trans, observation: Obs) -> Self {
        Self::new(transition, observation, UkfParams::default())
    }

    /// Performs the UKF prediction step.
    ///
    /// 1. Generate sigma points from current state
    /// 2. Propagate sigma points through transition function
    /// 3. Recover predicted mean and covariance
    ///
    /// # Returns
    /// Predicted state, or `None` if sigma point generation fails.
    pub fn predict(&self, state: &UkfState<T, N>, dt: T) -> Option<UkfState<T, N>> {
        // Generate sigma points
        let sigma_points = SigmaPoints::generate(state, &self.params)?;

        // Process noise
        let q = self.transition.process_noise(dt);

        // Propagate and recover statistics
        let (mean, cov) = sigma_points.recover_mean_cov(
            |x| self.transition.predict_nonlinear(x, dt).into_svector(),
            Some(q.as_matrix()),
        );

        Some(UkfState {
            mean: StateVector::from_svector(mean),
            covariance: StateCovariance::from_matrix(cov),
        })
    }

    /// Performs the UKF prediction step, panicking if it fails.
    ///
    /// Use this when you're confident the covariance is positive definite.
    pub fn predict_unchecked(&self, state: &UkfState<T, N>, dt: T) -> UkfState<T, N> {
        self.predict(state, dt)
            .expect("UKF prediction failed: covariance not positive definite")
    }

    /// Performs the UKF update step with a measurement.
    ///
    /// 1. Generate sigma points from predicted state
    /// 2. Transform sigma points through observation function
    /// 3. Compute predicted measurement mean and covariance
    /// 4. Compute cross-covariance
    /// 5. Compute Kalman gain and update state
    ///
    /// # Returns
    /// Updated state, or `None` if computation fails.
    pub fn update(
        &self,
        state: &UkfState<T, N>,
        measurement: &Measurement<T, M>,
    ) -> Option<UkfState<T, N>> {
        // Generate sigma points
        let sigma_points = SigmaPoints::generate(state, &self.params)?;

        // Measurement noise
        let r = self.observation.measurement_noise();

        // Transform sigma points through observation and recover statistics
        let (z_mean, z_cov) = sigma_points.recover_mean_cov(
            |x| self.observation.observe(x).into_svector(),
            Some(r.as_matrix()),
        );

        // Cross-covariance between state and measurement
        let cross_cov = sigma_points.cross_covariance(
            state.mean.as_svector(),
            |x| self.observation.observe(x).into_svector(),
            &z_mean,
        );

        // Kalman gain: K = P_xz * P_zz^{-1}
        let z_cov_inv = z_cov.try_inverse()?;
        let kalman_gain = cross_cov * z_cov_inv;

        // Innovation
        let innovation = measurement.as_svector() - z_mean;

        // Update mean
        let updated_mean = state.mean.as_svector() + kalman_gain * innovation;

        // Update covariance: P = P - K * P_zz * K^T
        let updated_cov =
            state.covariance.as_matrix() - kalman_gain * z_cov * kalman_gain.transpose();

        Some(UkfState {
            mean: StateVector::from_svector(updated_mean),
            covariance: StateCovariance::from_matrix(updated_cov),
        })
    }

    /// Performs a single predict-update cycle.
    pub fn step(
        &self,
        state: &UkfState<T, N>,
        dt: T,
        measurement: &Measurement<T, M>,
    ) -> Option<UkfState<T, N>> {
        let predicted = self.predict(state, dt)?;
        self.update(&predicted, measurement)
    }

    /// Computes the likelihood of a measurement given the current state.
    pub fn measurement_likelihood(
        &self,
        state: &UkfState<T, N>,
        measurement: &Measurement<T, M>,
    ) -> Option<T> {
        let sigma_points = SigmaPoints::generate(state, &self.params)?;
        let r = self.observation.measurement_noise();

        let (z_mean, z_cov) = sigma_points.recover_mean_cov(
            |x| self.observation.observe(x).into_svector(),
            Some(r.as_matrix()),
        );

        let innovation = measurement.as_svector() - z_mean;
        let innovation_typed =
            crate::types::spaces::Innovation::from_svector(innovation.clone_owned());
        let cov_typed = crate::types::spaces::Covariance::from_matrix(z_cov);

        crate::types::gaussian::gaussian_likelihood(&innovation_typed, &cov_typed)
    }

    /// Computes the Mahalanobis distance between a measurement and the predicted measurement.
    pub fn mahalanobis_distance_squared(
        &self,
        state: &UkfState<T, N>,
        measurement: &Measurement<T, M>,
    ) -> Option<T> {
        let sigma_points = SigmaPoints::generate(state, &self.params)?;
        let r = self.observation.measurement_noise();

        let (z_mean, z_cov) = sigma_points.recover_mean_cov(
            |x| self.observation.observe(x).into_svector(),
            Some(r.as_matrix()),
        );

        let z_cov_inv = z_cov.try_inverse()?;
        let innovation = measurement.as_svector() - z_mean;

        let d_sq = (innovation.transpose() * z_cov_inv * innovation)[(0, 0)];
        Some(d_sq)
    }
}

// ============================================================================
// UKF with Linear Transition
// ============================================================================

/// An Unscented Kalman Filter with a linear transition model but nonlinear observation.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct UkfLinearTransition<T, Trans, Obs, const N: usize, const M: usize>
where
    T: RealField,
    Trans: TransitionModel<T, N>,
    Obs: NonlinearObservationModel<T, N, M>,
{
    /// Linear transition (motion) model
    pub transition: Trans,
    /// Nonlinear observation (sensor) model
    pub observation: Obs,
    /// UKF parameters
    pub params: UkfParams<T>,
    /// Phantom marker
    _marker: PhantomData<T>,
}

#[cfg(feature = "alloc")]
impl<T, Trans, Obs, const N: usize, const M: usize> UkfLinearTransition<T, Trans, Obs, N, M>
where
    T: RealField + Float + Copy,
    Trans: TransitionModel<T, N>,
    Obs: NonlinearObservationModel<T, N, M>,
{
    /// Creates a new UKF with linear transition and nonlinear observation.
    #[inline]
    pub fn new(transition: Trans, observation: Obs, params: UkfParams<T>) -> Self {
        Self {
            transition,
            observation,
            params,
            _marker: PhantomData,
        }
    }

    /// Performs the standard Kalman prediction step (linear).
    pub fn predict(&self, state: &UkfState<T, N>, dt: T) -> UkfState<T, N> {
        let f = self.transition.transition_matrix(dt);
        let q = self.transition.process_noise(dt);

        let predicted_mean = f.apply_state(&state.mean);
        let predicted_cov = f.propagate_covariance(&state.covariance).add(&q);

        UkfState {
            mean: predicted_mean,
            covariance: predicted_cov,
        }
    }

    /// Performs the UKF update step with nonlinear observation.
    pub fn update(
        &self,
        state: &UkfState<T, N>,
        measurement: &Measurement<T, M>,
    ) -> Option<UkfState<T, N>> {
        let sigma_points = SigmaPoints::generate(state, &self.params)?;
        let r = self.observation.measurement_noise();

        let (z_mean, z_cov) = sigma_points.recover_mean_cov(
            |x| self.observation.observe(x).into_svector(),
            Some(r.as_matrix()),
        );

        let cross_cov = sigma_points.cross_covariance(
            state.mean.as_svector(),
            |x| self.observation.observe(x).into_svector(),
            &z_mean,
        );

        let z_cov_inv = z_cov.try_inverse()?;
        let kalman_gain = cross_cov * z_cov_inv;

        let innovation = measurement.as_svector() - z_mean;
        let updated_mean = state.mean.as_svector() + kalman_gain * innovation;
        let updated_cov =
            state.covariance.as_matrix() - kalman_gain * z_cov * kalman_gain.transpose();

        Some(UkfState {
            mean: StateVector::from_svector(updated_mean),
            covariance: StateCovariance::from_matrix(updated_cov),
        })
    }

    /// Performs a single predict-update cycle.
    pub fn step(
        &self,
        state: &UkfState<T, N>,
        dt: T,
        measurement: &Measurement<T, M>,
    ) -> Option<UkfState<T, N>> {
        let predicted = self.predict(state, dt);
        self.update(&predicted, measurement)
    }
}

// ============================================================================
// Standalone UKF Functions
// ============================================================================

/// Performs unscented transform of a state distribution through a nonlinear function.
///
/// This is the core algorithm of the UKF, useful for custom implementations.
///
/// # Type Parameters
/// - `T`: Scalar type
/// - `N`: Input state dimension
/// - `D`: Output dimension
///
/// # Arguments
/// - `mean`: Input mean
/// - `cov`: Input covariance
/// - `params`: UKF parameters
/// - `transform`: Nonlinear transformation function
/// - `additive_noise`: Optional additive noise covariance
///
/// # Returns
/// Transformed mean and covariance, or `None` if Cholesky fails.
#[cfg(feature = "alloc")]
pub fn unscented_transform<T, const N: usize, const D: usize, F>(
    mean: &StateVector<T, N>,
    cov: &StateCovariance<T, N>,
    params: &UkfParams<T>,
    transform: F,
    additive_noise: Option<&SMatrix<T, D, D>>,
) -> Option<(SVector<T, D>, SMatrix<T, D, D>)>
where
    T: RealField + Float + Copy,
    F: Fn(&StateVector<T, N>) -> SVector<T, D>,
{
    let state = UkfState::new(*mean, *cov);
    let sigma_points = SigmaPoints::generate(&state, params)?;
    Some(sigma_points.recover_mean_cov(transform, additive_noise))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use super::*;
    use crate::models::{ConstantVelocity2D, CoordinatedTurn2D, RangeBearingSensor};

    #[test]
    fn test_ukf_params_default() {
        let params: UkfParams<f64> = UkfParams::default();
        assert!((params.alpha - 1e-3).abs() < 1e-10);
        assert!((params.beta - 2.0).abs() < 1e-10);
        assert!(params.kappa.abs() < 1e-10);
    }

    #[test]
    fn test_ukf_params_weights() {
        let params: UkfParams<f64> = UkfParams::default();
        let n = 5;

        let w0_mean = params.weight_mean_0(n);
        let wi = params.weight_i(n);

        // Weights should sum to 1 for mean
        let sum_mean = w0_mean + 2.0 * n as f64 * wi;
        assert!(
            (sum_mean - 1.0).abs() < 1e-6,
            "Mean weights sum: {}",
            sum_mean
        );
    }

    #[test]
    fn test_sigma_point_generation() {
        let mean: StateVector<f64, 4> = StateVector::from_array([1.0, 2.0, 3.0, 4.0]);
        let cov: StateCovariance<f64, 4> = StateCovariance::identity();
        let state = UkfState::new(mean, cov);

        let params = UkfParams::default();
        let sigma = SigmaPoints::generate(&state, &params).unwrap();

        // Central point should be the mean
        for i in 0..4 {
            assert!(
                (sigma.points[0].index(i) - state.mean.index(i)).abs() < 1e-10,
                "Central point mismatch at {}",
                i
            );
        }

        // Should have 2*4+1 = 9 sigma points
        assert_eq!(sigma.points.len(), 9);
    }

    #[test]
    fn test_sigma_points_recover_identity() {
        // If we recover without transformation, should get original mean/cov
        let mean: StateVector<f64, 4> = StateVector::from_array([1.0, 2.0, 3.0, 4.0]);
        let cov: StateCovariance<f64, 4> = StateCovariance::identity();
        let state = UkfState::new(mean, cov);

        let params = UkfParams::default();
        let sigma = SigmaPoints::generate(&state, &params).unwrap();

        // Identity transform
        let (recovered_mean, recovered_cov) =
            sigma.recover_mean_cov(|x| x.as_svector().clone_owned(), None);

        for i in 0..4 {
            assert!(
                (recovered_mean[i] - *state.mean.index(i)).abs() < 1e-6,
                "Mean mismatch at {}: {} vs {}",
                i,
                recovered_mean[i],
                state.mean.index(i)
            );
        }

        // Covariance should be close to original (with some numerical error)
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (recovered_cov[(i, j)] - expected).abs() < 1e-4,
                    "Cov mismatch at ({}, {}): {} vs {}",
                    i,
                    j,
                    recovered_cov[(i, j)],
                    expected
                );
            }
        }
    }

    #[test]
    fn test_ukf_linear_transition_predict() {
        // Test UKF with linear transition (should behave like standard Kalman)
        let transition = ConstantVelocity2D::new(0.1_f64, 0.99);
        let sensor = RangeBearingSensor::new(10.0, 0.01, 0.95);
        let filter = UkfLinearTransition::new(transition, sensor, UkfParams::default());

        // State at (100, 0) moving east at 10 m/s
        let state = UkfState::new(
            StateVector::from_array([100.0, 0.0, 10.0, 0.0]),
            StateCovariance::identity(),
        );

        let predicted = filter.predict(&state, 1.0);

        // Should move to (110, 0)
        assert!(
            (predicted.mean.index(0) - 110.0).abs() < 1e-6,
            "x: {}",
            predicted.mean.index(0)
        );
        assert!(
            (predicted.mean.index(1) - 0.0).abs() < 1e-6,
            "y: {}",
            predicted.mean.index(1)
        );
    }

    #[test]
    fn test_ukf_linear_transition_update() {
        let transition = ConstantVelocity2D::new(0.1_f64, 0.99);
        let sensor = RangeBearingSensor::new(10.0, 0.01, 0.95);
        let filter = UkfLinearTransition::new(transition, sensor, UkfParams::default());

        let state = UkfState::new(
            StateVector::from_array([100.0, 0.0, 10.0, 0.0]),
            StateCovariance::identity(),
        );

        // Predict 1 second
        let predicted = filter.predict(&state, 1.0);
        assert!((predicted.mean.index(0) - 110.0).abs() < 1e-6);

        // Update with range-bearing
        let measurement = Measurement::from_array([110.0, 0.0]);
        let updated = filter.update(&predicted, &measurement).unwrap();
        assert!((updated.mean.index(0) - 110.0).abs() < 5.0);
    }

    #[test]
    fn test_ukf_step_linear_transition() {
        let transition = ConstantVelocity2D::new(0.1_f64, 0.99);
        let sensor = RangeBearingSensor::new(10.0, 0.01, 0.95);
        let filter = UkfLinearTransition::new(transition, sensor, UkfParams::default());

        let state = UkfState::new(
            StateVector::from_array([100.0, 0.0, 10.0, 0.0]),
            StateCovariance::identity(),
        );

        let measurement = Measurement::from_array([110.0, 0.0]);
        let updated = filter.step(&state, 1.0, &measurement).unwrap();

        assert!((updated.mean.index(0) - 110.0).abs() < 5.0);
    }

    #[test]
    fn test_unscented_transform() {
        let mean: StateVector<f64, 4> = StateVector::from_array([1.0, 2.0, 3.0, 4.0]);
        let cov: StateCovariance<f64, 4> = StateCovariance::identity();
        let params = UkfParams::default();

        // Linear transform: should recover exact statistics
        let (transformed_mean, _) = unscented_transform(
            &mean,
            &cov,
            &params,
            |x| x.as_svector().clone_owned(),
            None,
        )
        .unwrap();

        for i in 0..4 {
            assert!(
                (transformed_mean[i] - *mean.index(i)).abs() < 1e-6,
                "Mean mismatch at {}: {} vs {}",
                i,
                transformed_mean[i],
                mean.index(i)
            );
        }
    }

    // Test with the full nonlinear UKF using CoordinatedTurn2D
    // Note: We need a 5D observation model for this. Since RangeBearingSensor is 4D,
    // we'll create a simple wrapper or test the prediction-only path.

    #[test]
    fn test_coordinated_turn_prediction_via_unscented_transform() {
        // Test the nonlinear prediction using unscented_transform directly
        let transition = CoordinatedTurn2D::new(0.1_f64, 0.01, 0.99);

        let mean: StateVector<f64, 5> = StateVector::from_array([100.0, 0.0, 10.0, 0.0, 0.0]);
        let cov: StateCovariance<f64, 5> = StateCovariance::identity();
        let params = UkfParams::default();

        let dt = 1.0;
        let q = transition.process_noise(dt);

        let (pred_mean, _pred_cov) = unscented_transform(
            &mean,
            &cov,
            &params,
            |x| transition.predict_nonlinear(x, dt).into_svector(),
            Some(q.as_matrix()),
        )
        .unwrap();

        // With zero turn rate, should move east by 10m
        // Note: UKF with sigma points spread around the mean will average slightly differently
        // due to the nonlinear nature of the coordinated turn model at omega=0 boundary.
        // The sigma points with small positive/negative omega values average out.
        assert!(
            (pred_mean[0] - 110.0).abs() < 2.0,
            "x: {} vs 110.0",
            pred_mean[0]
        );
        assert!((pred_mean[1] - 0.0).abs() < 2.0, "y: {} vs 0.0", pred_mean[1]);
    }

    #[test]
    fn test_coordinated_turn_with_turn() {
        use std::f64::consts::FRAC_PI_2;

        let transition = CoordinatedTurn2D::new(0.1_f64, 0.01, 0.99);

        // Moving east at 10 m/s, turning left at pi/2 rad/s
        let mean: StateVector<f64, 5> = StateVector::from_array([0.0, 0.0, 10.0, 0.0, FRAC_PI_2]);
        let cov: StateCovariance<f64, 5> = StateCovariance::identity();
        let params = UkfParams::default();

        let dt = 1.0;
        let q = transition.process_noise(dt);

        let (pred_mean, _) = unscented_transform(
            &mean,
            &cov,
            &params,
            |x| transition.predict_nonlinear(x, dt).into_svector(),
            Some(q.as_matrix()),
        )
        .unwrap();

        // Turn radius r = v/omega = 10/(pi/2) ≈ 6.37
        let r = 10.0 / FRAC_PI_2;

        // Position should be at approximately (r, r)
        // Note: UKF accounts for uncertainty in omega, which affects the arc length.
        // With unit covariance on omega, the sigma points span a range of turn rates,
        // leading to different arc trajectories that average out.
        assert!(
            (pred_mean[0] - r).abs() < 1.5,
            "x: {} vs {}",
            pred_mean[0],
            r
        );
        assert!(
            (pred_mean[1] - r).abs() < 1.5,
            "y: {} vs {}",
            pred_mean[1],
            r
        );
    }
}
