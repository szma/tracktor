//! Gaussian components and mixtures
//!
//! This module provides weighted Gaussian components for representing
//! probability density functions in multi-target tracking.

use ::core::f64::consts::PI;
use nalgebra::RealField;
use num_traits::Float;

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use super::spaces::{StateVector, StateCovariance, Measurement, MeasurementCovariance};
use super::transforms::{
    ObservationMatrix, TransitionMatrix,
    compute_kalman_gain, compute_innovation_covariance, joseph_update,
};

// ============================================================================
// Gaussian State
// ============================================================================

/// A weighted Gaussian component representing a target state estimate.
///
/// In the PHD filter context, the weight represents the expected number
/// of targets in that region of state space.
#[derive(Debug, Clone, PartialEq)]
pub struct GaussianState<T: RealField, const N: usize> {
    /// Weight (expected target count in PHD context)
    pub weight: T,
    /// Mean state vector
    pub mean: StateVector<T, N>,
    /// State covariance matrix
    pub covariance: StateCovariance<T, N>,
}

impl<T: RealField + Copy, const N: usize> GaussianState<T, N> {
    /// Creates a new Gaussian component.
    #[inline]
    pub fn new(weight: T, mean: StateVector<T, N>, covariance: StateCovariance<T, N>) -> Self {
        Self { weight, mean, covariance }
    }

    /// Creates a Gaussian with unit weight.
    #[inline]
    pub fn with_unit_weight(mean: StateVector<T, N>, covariance: StateCovariance<T, N>) -> Self {
        Self {
            weight: T::one(),
            mean,
            covariance,
        }
    }

    /// Scales the weight of this component.
    #[inline]
    pub fn scale_weight(&self, factor: T) -> Self {
        Self {
            weight: self.weight * factor,
            mean: self.mean,
            covariance: self.covariance,
        }
    }

    /// Predicts this Gaussian through a linear transition model.
    #[inline]
    pub fn predict(
        &self,
        transition: &TransitionMatrix<T, N>,
        process_noise: &StateCovariance<T, N>,
    ) -> Self {
        let predicted_mean = transition.apply_state(&self.mean);
        let predicted_cov = transition.propagate_covariance(&self.covariance)
            .add(process_noise);

        Self {
            weight: self.weight,
            mean: predicted_mean,
            covariance: predicted_cov,
        }
    }

    /// Updates this Gaussian with a measurement using the Kalman filter equations.
    pub fn update<const M: usize>(
        &self,
        measurement: &Measurement<T, M>,
        obs_matrix: &ObservationMatrix<T, M, N>,
        meas_noise: &MeasurementCovariance<T, M>,
    ) -> Option<Self> {
        use super::spaces::ComputeInnovation;

        // Predicted measurement
        let predicted_meas = obs_matrix.observe(&self.mean);

        // Innovation
        let innovation = measurement.innovation(predicted_meas);

        // Innovation covariance
        let innovation_cov = compute_innovation_covariance(
            &self.covariance,
            obs_matrix,
            meas_noise,
        );

        // Kalman gain
        let kalman_gain = compute_kalman_gain(
            &self.covariance,
            obs_matrix,
            &innovation_cov,
        )?;

        // Updated mean
        let correction = kalman_gain.correct(&innovation);
        let updated_mean = StateVector::from_svector(
            self.mean.as_svector() + correction.as_svector()
        );

        // Updated covariance (Joseph form for numerical stability)
        let updated_cov = joseph_update(
            &self.covariance,
            &kalman_gain,
            obs_matrix,
            meas_noise,
        );

        Some(Self {
            weight: self.weight,
            mean: updated_mean,
            covariance: updated_cov,
        })
    }

    /// Computes the likelihood of a measurement given this Gaussian.
    ///
    /// Returns `None` if the innovation covariance is singular (numerical instability).
    pub fn measurement_likelihood<const M: usize>(
        &self,
        measurement: &Measurement<T, M>,
        obs_matrix: &ObservationMatrix<T, M, N>,
        meas_noise: &MeasurementCovariance<T, M>,
    ) -> Option<T>
    where
        T: Float,
    {
        use super::spaces::ComputeInnovation;

        // Predicted measurement
        let predicted_meas = obs_matrix.observe(&self.mean);

        // Innovation
        let innovation = measurement.innovation(predicted_meas);

        // Innovation covariance: S = H * P * H^T + R
        let innovation_cov = compute_innovation_covariance(
            &self.covariance,
            obs_matrix,
            meas_noise,
        );

        // Compute Gaussian likelihood - convert covariance to innovation space
        let innovation_cov_typed = super::spaces::Covariance::from_matrix(*innovation_cov.as_matrix());
        gaussian_likelihood(&innovation, &innovation_cov_typed)
    }
}

// ============================================================================
// Gaussian Likelihood Computation
// ============================================================================

/// Computes the multivariate Gaussian likelihood.
///
/// p(z) = (2π)^(-M/2) |S|^(-1/2) exp(-0.5 * z^T * S^{-1} * z)
///
/// This is generic over the vector space type. For innovation-based likelihood
/// calculations in Kalman filters, use `innovation_likelihood` instead.
///
/// Returns `None` if the covariance matrix is singular or not positive definite.
/// This explicit error propagation is preferred over silently returning zero.
pub fn gaussian_likelihood<T: RealField + Float + Copy, Space, const M: usize>(
    z: &super::spaces::Vector<T, M, Space>,
    covariance: &super::spaces::Covariance<T, M, Space>,
) -> Option<T> {
    // Determinant - returns None if not positive definite
    let det = covariance.determinant()?;

    if det <= T::zero() {
        return None;
    }

    // Inverse covariance
    let cov_inv = covariance.try_inverse()?;

    // Mahalanobis distance squared: z^T * S^{-1} * z
    let mahal_sq = (z.as_svector().transpose() * cov_inv.as_matrix() * z.as_svector())[(0, 0)];

    // Normalization constant
    let m = T::from(M).unwrap();
    let two_pi = T::from(2.0 * PI).unwrap();
    let norm = num_traits::Float::powf(two_pi, m / T::from(2.0).unwrap()) * num_traits::Float::sqrt(det);

    // Likelihood
    let half = T::from(0.5).unwrap();
    Some(num_traits::Float::exp(-half * mahal_sq) / norm)
}

/// Computes the log-likelihood of a Gaussian.
///
/// Returns `None` if the covariance matrix is singular or not positive definite.
pub fn gaussian_log_likelihood<T: RealField + Float + Copy, Space, const M: usize>(
    z: &super::spaces::Vector<T, M, Space>,
    covariance: &super::spaces::Covariance<T, M, Space>,
) -> Option<T> {
    // Determinant - returns None if not positive definite
    let det = covariance.determinant()?;

    if det <= T::zero() {
        return None;
    }

    // Inverse covariance
    let cov_inv = covariance.try_inverse()?;

    // Mahalanobis distance squared
    let mahal_sq = (z.as_svector().transpose() * cov_inv.as_matrix() * z.as_svector())[(0, 0)];

    // Log normalization constant
    let m = T::from(M).unwrap();
    let two_pi = T::from(2.0 * PI).unwrap();
    let log_norm = (m / T::from(2.0).unwrap()) * num_traits::Float::ln(two_pi) + num_traits::Float::ln(det) / T::from(2.0).unwrap();

    // Log likelihood
    let half = T::from(0.5).unwrap();
    Some(-half * mahal_sq - log_norm)
}

/// Computes the Gaussian likelihood for an innovation vector.
///
/// This is a convenience function for Kalman filter update steps where the
/// innovation vector (measurement residual) needs to be evaluated against
/// the innovation covariance S = H*P*H' + R.
///
/// Since innovation vectors are in InnovationSpace but the covariance S is
/// typically computed in MeasurementSpace, this function accepts the raw
/// matrix data to avoid type conversion issues.
pub fn innovation_likelihood<T: RealField + Float + Copy, const M: usize>(
    innovation: &super::spaces::Innovation<T, M>,
    innovation_cov_matrix: &nalgebra::SMatrix<T, M, M>,
) -> T {
    // Use Cholesky decomposition for stability
    let chol = match nalgebra::Cholesky::new(*innovation_cov_matrix) {
        Some(c) => c,
        None => return T::zero(),
    };

    // Determinant via Cholesky: det(S) = det(L)^2
    let l = chol.l();
    let mut det_l = T::one();
    for i in 0..M {
        det_l = det_l * l[(i, i)];
    }
    let det = det_l * det_l;

    if det <= T::zero() {
        return T::zero();
    }

    // Solve L * y = z for y, then ||y||^2 = z^T * S^{-1} * z
    let z = innovation.as_svector();
    let y = chol.l().solve_lower_triangular(z);
    let mahal_sq = match y {
        Some(y_vec) => y_vec.norm_squared(),
        None => return T::zero(),
    };

    // Normalization constant
    let m = T::from(M).unwrap();
    let two_pi = T::from(2.0 * PI).unwrap();
    let norm = num_traits::Float::powf(two_pi, m / T::from(2.0).unwrap()) * num_traits::Float::sqrt(det);

    // Likelihood
    let half = T::from(0.5).unwrap();
    num_traits::Float::exp(-half * mahal_sq) / norm
}

// ============================================================================
// Gaussian Mixture
// ============================================================================

/// A Gaussian mixture representing a multi-modal probability distribution.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct GaussianMixture<T: RealField, const N: usize> {
    /// The Gaussian components
    pub components: Vec<GaussianState<T, N>>,
}

#[cfg(feature = "alloc")]
impl<T: RealField + Copy, const N: usize> GaussianMixture<T, N> {
    /// Creates an empty mixture.
    #[inline]
    pub fn new() -> Self {
        Self { components: Vec::new() }
    }

    /// Creates a mixture with the given capacity.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self { components: Vec::with_capacity(capacity) }
    }

    /// Creates a mixture from a vector of components.
    #[inline]
    pub fn from_components(components: Vec<GaussianState<T, N>>) -> Self {
        Self { components }
    }

    /// Returns the number of components.
    #[inline]
    pub fn len(&self) -> usize {
        self.components.len()
    }

    /// Returns true if the mixture is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.components.is_empty()
    }

    /// Adds a component to the mixture.
    #[inline]
    pub fn push(&mut self, component: GaussianState<T, N>) {
        self.components.push(component);
    }

    /// Extends the mixture with components from an iterator.
    #[inline]
    pub fn extend<I: IntoIterator<Item = GaussianState<T, N>>>(&mut self, iter: I) {
        self.components.extend(iter);
    }

    /// Returns the total weight (expected number of targets for PHD).
    pub fn total_weight(&self) -> T {
        self.components.iter().fold(T::zero(), |acc, c| acc + c.weight)
    }

    /// Iterates over the components.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &GaussianState<T, N>> {
        self.components.iter()
    }

    /// Mutably iterates over the components.
    #[inline]
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut GaussianState<T, N>> {
        self.components.iter_mut()
    }

    /// Clears all components.
    #[inline]
    pub fn clear(&mut self) {
        self.components.clear();
    }

    /// Scales all component weights by a factor.
    pub fn scale_weights(&mut self, factor: T) {
        for component in &mut self.components {
            component.weight = component.weight * factor;
        }
    }
}

#[cfg(feature = "alloc")]
impl<T: RealField + Copy, const N: usize> Default for GaussianMixture<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Fixed-Size Gaussian Mixture (no_std without alloc)
// ============================================================================

/// A fixed-capacity Gaussian mixture for no_std environments.
///
/// Maximum number of components is determined at compile time.
/// Uses a compact storage format without Option overhead.
pub struct FixedGaussianMixture<T: RealField, const N: usize, const MAX: usize> {
    /// Contiguous storage for components (only first `len` elements are valid)
    components: [core::mem::MaybeUninit<GaussianState<T, N>>; MAX],
    /// Number of valid components
    len: usize,
}

impl<T: RealField + Copy, const N: usize, const MAX: usize> FixedGaussianMixture<T, N, MAX> {
    /// Creates an empty fixed-size mixture.
    pub fn new() -> Self {
        Self {
            // SAFETY: MaybeUninit doesn't require initialization
            components: unsafe { core::mem::MaybeUninit::uninit().assume_init() },
            len: 0,
        }
    }

    /// Returns the number of components.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true if the mixture is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the maximum capacity.
    #[inline]
    pub fn capacity(&self) -> usize {
        MAX
    }

    /// Attempts to add a component. Returns Err if at capacity.
    pub fn try_push(&mut self, component: GaussianState<T, N>) -> Result<(), crate::TracktorError> {
        if self.len >= MAX {
            return Err(crate::TracktorError::MaxComponentsExceeded);
        }
        self.components[self.len].write(component);
        self.len += 1;
        Ok(())
    }

    /// Returns the total weight.
    pub fn total_weight(&self) -> T {
        self.iter().fold(T::zero(), |acc, c| acc + c.weight)
    }

    /// Iterates over the components.
    pub fn iter(&self) -> impl Iterator<Item = &GaussianState<T, N>> {
        // SAFETY: elements 0..self.len are initialized
        self.components[..self.len].iter().map(|c| unsafe { c.assume_init_ref() })
    }

    /// Returns a slice of the components.
    pub fn as_slice(&self) -> &[GaussianState<T, N>] {
        // SAFETY: elements 0..self.len are initialized and contiguous
        unsafe {
            core::slice::from_raw_parts(
                self.components.as_ptr() as *const GaussianState<T, N>,
                self.len,
            )
        }
    }

    /// Clears all components.
    pub fn clear(&mut self) {
        // Drop existing components
        for i in 0..self.len {
            // SAFETY: elements 0..self.len are initialized
            unsafe { self.components[i].assume_init_drop() };
        }
        self.len = 0;
    }

    /// Gets a reference to a component at the given index.
    pub fn get(&self, index: usize) -> Option<&GaussianState<T, N>> {
        if index < self.len {
            // SAFETY: elements 0..self.len are initialized
            Some(unsafe { self.components[index].assume_init_ref() })
        } else {
            None
        }
    }
}

impl<T: RealField + Copy, const N: usize, const MAX: usize> Default for FixedGaussianMixture<T, N, MAX> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: RealField, const N: usize, const MAX: usize> Drop for FixedGaussianMixture<T, N, MAX> {
    fn drop(&mut self) {
        // Drop existing components
        for i in 0..self.len {
            // SAFETY: elements 0..self.len are initialized
            unsafe { self.components[i].assume_init_drop() };
        }
        self.len = 0;
    }
}

impl<T: RealField + Copy, const N: usize, const MAX: usize> Clone for FixedGaussianMixture<T, N, MAX> {
    fn clone(&self) -> Self {
        let mut new = Self::new();
        for component in self.iter() {
            // This cannot fail since we're cloning into the same capacity (MAX).
            // If it somehow did fail, that would indicate a serious bug.
            new.try_push(component.clone())
                .expect("Clone into same-capacity FixedGaussianMixture should never fail");
        }
        new
    }
}

impl<T: RealField + Copy + core::fmt::Debug, const N: usize, const MAX: usize> core::fmt::Debug for FixedGaussianMixture<T, N, MAX> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("FixedGaussianMixture")
            .field("components", &self.as_slice())
            .field("len", &self.len)
            .field("capacity", &MAX)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_state_creation() {
        let mean: StateVector<f64, 4> = StateVector::from_array([0.0, 0.0, 1.0, 0.0]);
        let cov: StateCovariance<f64, 4> = StateCovariance::identity();

        let gs = GaussianState::new(0.5, mean, cov);
        assert!((gs.weight - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_gaussian_predict() {
        let mean: StateVector<f64, 4> = StateVector::from_array([0.0, 0.0, 1.0, 2.0]);
        let cov: StateCovariance<f64, 4> = StateCovariance::identity();
        let gs = GaussianState::new(1.0, mean, cov);

        // Constant velocity transition
        let dt = 1.0;
        let f = TransitionMatrix::from_matrix(nalgebra::matrix![
            1.0, 0.0, dt, 0.0;
            0.0, 1.0, 0.0, dt;
            0.0, 0.0, 1.0, 0.0;
            0.0, 0.0, 0.0, 1.0
        ]);
        let q = StateCovariance::from_matrix(nalgebra::SMatrix::<f64, 4, 4>::identity().scale(0.01));

        let predicted = gs.predict(&f, &q);

        // Position should have moved by velocity
        assert!((predicted.mean.index(0) - 1.0).abs() < 1e-10);
        assert!((predicted.mean.index(1) - 2.0).abs() < 1e-10);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_gaussian_mixture() {
        let mut mixture = GaussianMixture::new();

        let mean1: StateVector<f64, 2> = StateVector::from_array([0.0, 0.0]);
        let cov1: StateCovariance<f64, 2> = StateCovariance::identity();
        mixture.push(GaussianState::new(0.3, mean1, cov1));

        let mean2: StateVector<f64, 2> = StateVector::from_array([5.0, 5.0]);
        let cov2: StateCovariance<f64, 2> = StateCovariance::identity();
        mixture.push(GaussianState::new(0.7, mean2, cov2));

        assert_eq!(mixture.len(), 2);
        assert!((mixture.total_weight() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_fixed_gaussian_mixture() {
        let mut mixture: FixedGaussianMixture<f64, 2, 10> = FixedGaussianMixture::new();

        let mean: StateVector<f64, 2> = StateVector::from_array([0.0, 0.0]);
        let cov: StateCovariance<f64, 2> = StateCovariance::identity();

        assert!(mixture.try_push(GaussianState::new(0.5, mean, cov)).is_ok());
        assert_eq!(mixture.len(), 1);
    }
}
