//! Vector space markers and typed vectors
//!
//! This module provides type-safe vectors that cannot be accidentally mixed
//! across different mathematical spaces (state, measurement, innovation).

use ::core::marker::PhantomData;
use ::core::ops::{Add, Mul, Neg, Sub};
use nalgebra::{RealField, SVector, Scalar};

// ============================================================================
// Vector Space Markers
// ============================================================================

/// Marker type for state space vectors (e.g., position, velocity)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StateSpace;

/// Marker type for measurement space vectors (e.g., sensor observations)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MeasurementSpace;

/// Marker type for innovation vectors (measurement - predicted measurement)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InnovationSpace;

// ============================================================================
// Typed Vector
// ============================================================================

/// A vector parameterized by scalar type, dimension, and mathematical space.
///
/// The `Space` parameter ensures that vectors from different spaces cannot
/// be accidentally mixed in operations.
///
/// # Type Parameters
///
/// - `T`: The scalar type (typically `f32` or `f64`)
/// - `N`: The dimension of the vector (const generic)
/// - `Space`: A marker type indicating which mathematical space this vector belongs to
#[repr(transparent)]
#[derive(Debug, Clone, PartialEq)]
pub struct Vector<T: Scalar, const N: usize, Space> {
    inner: SVector<T, N>,
    _marker: PhantomData<Space>,
}

impl<T: Scalar, const N: usize, Space> Vector<T, N, Space> {
    /// Creates a new vector from raw components.
    #[inline]
    pub fn from_array(data: [T; N]) -> Self {
        Self {
            inner: SVector::from(data),
            _marker: PhantomData,
        }
    }

    /// Creates a new vector from an nalgebra SVector.
    #[inline]
    pub fn from_svector(inner: SVector<T, N>) -> Self {
        Self {
            inner,
            _marker: PhantomData,
        }
    }

    /// Returns a reference to the underlying nalgebra vector.
    #[inline]
    pub fn as_svector(&self) -> &SVector<T, N> {
        &self.inner
    }

    /// Consumes self and returns the underlying nalgebra vector.
    #[inline]
    pub fn into_svector(self) -> SVector<T, N> {
        self.inner
    }

    /// Returns a reference to the raw data array.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        self.inner.as_slice()
    }

    /// Access element at index.
    #[inline]
    pub fn get(&self, index: usize) -> Option<&T> {
        self.inner.get(index)
    }

    /// Access element at index (unchecked).
    ///
    /// # Panics
    /// Panics if index is out of bounds.
    #[inline]
    #[allow(clippy::should_implement_trait)]
    pub fn index(&self, index: usize) -> &T {
        &self.inner[index]
    }
}

impl<T: Scalar + Copy, const N: usize, Space: Clone> Copy for Vector<T, N, Space> {}

impl<T: RealField + Copy, const N: usize, Space> Vector<T, N, Space> {
    /// Creates a zero vector.
    #[inline]
    pub fn zeros() -> Self {
        Self {
            inner: SVector::zeros(),
            _marker: PhantomData,
        }
    }

    /// Computes the squared Euclidean norm.
    #[inline]
    pub fn norm_squared(&self) -> T {
        self.inner.norm_squared()
    }

    /// Computes the Euclidean norm.
    #[inline]
    pub fn norm(&self) -> T {
        self.inner.norm()
    }

    /// Scales the vector by a scalar.
    #[inline]
    pub fn scale(&self, s: T) -> Self {
        Self {
            inner: self.inner.scale(s),
            _marker: PhantomData,
        }
    }
}

// ============================================================================
// Type Aliases
// ============================================================================

/// A state vector in state space.
pub type StateVector<T, const N: usize> = Vector<T, N, StateSpace>;

/// A measurement vector in measurement space.
pub type Measurement<T, const M: usize> = Vector<T, M, MeasurementSpace>;

/// An innovation vector (measurement residual) in innovation space.
pub type Innovation<T, const M: usize> = Vector<T, M, InnovationSpace>;

// ============================================================================
// Operations: Same-Space Addition/Subtraction
// ============================================================================

impl<T: RealField + Copy, const N: usize, Space> Add for Vector<T, N, Space> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            inner: self.inner + rhs.inner,
            _marker: PhantomData,
        }
    }
}

impl<T: RealField + Copy, const N: usize, Space> Sub for Vector<T, N, Space> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            inner: self.inner - rhs.inner,
            _marker: PhantomData,
        }
    }
}

impl<T: RealField + Copy, const N: usize, Space> Neg for Vector<T, N, Space> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        Self {
            inner: -self.inner,
            _marker: PhantomData,
        }
    }
}

impl<T: RealField + Copy, const N: usize, Space> Mul<T> for Vector<T, N, Space> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: T) -> Self::Output {
        Self {
            inner: self.inner * rhs,
            _marker: PhantomData,
        }
    }
}

// ============================================================================
// Special Operation: Measurement - Measurement = Innovation
// ============================================================================

/// Trait for computing innovation (residual) from measurements.
///
/// This is a separate trait because subtracting two measurements
/// produces an innovation vector, not another measurement.
pub trait ComputeInnovation<T: RealField, const M: usize> {
    /// Computes the innovation (residual) between this measurement and a predicted measurement.
    fn innovation(self, predicted: Measurement<T, M>) -> Innovation<T, M>;
}

impl<T: RealField + Copy, const M: usize> ComputeInnovation<T, M> for Measurement<T, M> {
    #[inline]
    fn innovation(self, predicted: Measurement<T, M>) -> Innovation<T, M> {
        Innovation {
            inner: self.inner - predicted.inner,
            _marker: PhantomData,
        }
    }
}

// ============================================================================
// Covariance Matrix
// ============================================================================

/// A covariance matrix bound to a specific vector space.
///
/// Covariance matrices are symmetric positive semi-definite matrices
/// that describe the uncertainty in a vector estimate.
#[repr(transparent)]
#[derive(Debug, Clone, PartialEq)]
pub struct Covariance<T: Scalar, const N: usize, Space> {
    inner: nalgebra::SMatrix<T, N, N>,
    _marker: PhantomData<Space>,
}

impl<T: Scalar, const N: usize, Space> Covariance<T, N, Space> {
    /// Creates a covariance matrix from a raw matrix.
    ///
    /// # Safety (logical)
    /// The caller should ensure the matrix is symmetric and positive semi-definite.
    #[inline]
    pub fn from_matrix(inner: nalgebra::SMatrix<T, N, N>) -> Self {
        Self {
            inner,
            _marker: PhantomData,
        }
    }

    /// Returns a reference to the underlying matrix.
    #[inline]
    pub fn as_matrix(&self) -> &nalgebra::SMatrix<T, N, N> {
        &self.inner
    }

    /// Consumes self and returns the underlying matrix.
    #[inline]
    pub fn into_matrix(self) -> nalgebra::SMatrix<T, N, N> {
        self.inner
    }
}

impl<T: Scalar + Copy, const N: usize, Space: Clone> Copy for Covariance<T, N, Space> where
    nalgebra::SMatrix<T, N, N>: Copy
{
}

impl<T: RealField + Copy, const N: usize, Space> Covariance<T, N, Space> {
    /// Creates a zero covariance matrix.
    #[inline]
    pub fn zeros() -> Self {
        Self {
            inner: nalgebra::SMatrix::zeros(),
            _marker: PhantomData,
        }
    }

    /// Creates an identity covariance matrix.
    #[inline]
    pub fn identity() -> Self {
        Self {
            inner: nalgebra::SMatrix::identity(),
            _marker: PhantomData,
        }
    }

    /// Creates a diagonal covariance matrix.
    #[inline]
    pub fn from_diagonal(diag: &SVector<T, N>) -> Self {
        Self {
            inner: nalgebra::SMatrix::from_diagonal(diag),
            _marker: PhantomData,
        }
    }

    /// Scales the covariance matrix.
    #[inline]
    pub fn scale(&self, s: T) -> Self {
        Self {
            inner: self.inner.scale(s),
            _marker: PhantomData,
        }
    }

    /// Adds two covariance matrices.
    #[inline]
    pub fn add(&self, other: &Self) -> Self {
        Self {
            inner: self.inner + other.inner,
            _marker: PhantomData,
        }
    }

    /// Computes the trace of the covariance matrix.
    #[inline]
    pub fn trace(&self) -> T {
        self.inner.trace()
    }

    /// Computes the determinant of the covariance matrix via Cholesky decomposition.
    ///
    /// For a positive definite matrix, det(A) = det(L)^2 where L is lower triangular.
    /// Returns None if the matrix is not positive definite.
    #[inline]
    pub fn determinant_cholesky(&self) -> Option<T> {
        let chol = nalgebra::Cholesky::new(self.inner)?;
        let l = chol.l();
        // det(A) = det(L)^2, and det(L) = product of diagonal elements
        let mut det_l = T::one();
        for i in 0..N {
            det_l *= l[(i, i)];
        }
        Some(det_l * det_l)
    }

    /// Computes the determinant assuming the matrix is positive definite.
    ///
    /// Returns `None` if the matrix is not positive definite (Cholesky decomposition fails).
    /// This is preferred over silently returning zero, which could mask numerical issues.
    #[inline]
    pub fn determinant(&self) -> Option<T> {
        self.determinant_cholesky()
    }

    /// Attempts to compute the inverse of the covariance matrix.
    #[inline]
    pub fn try_inverse(&self) -> Option<Self> {
        self.inner.try_inverse().map(|inner| Self {
            inner,
            _marker: PhantomData,
        })
    }

    /// Computes the Cholesky decomposition (lower triangular).
    #[inline]
    pub fn cholesky(&self) -> Option<nalgebra::SMatrix<T, N, N>> {
        nalgebra::Cholesky::new(self.inner).map(|c| c.l())
    }
}

impl<T: RealField + Copy, const N: usize, Space> Add for Covariance<T, N, Space> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            inner: self.inner + rhs.inner,
            _marker: PhantomData,
        }
    }
}

// ============================================================================
// Type Aliases for Covariance
// ============================================================================

/// Covariance matrix in state space.
pub type StateCovariance<T, const N: usize> = Covariance<T, N, StateSpace>;

/// Covariance matrix in measurement space.
pub type MeasurementCovariance<T, const M: usize> = Covariance<T, M, MeasurementSpace>;

/// Covariance matrix in innovation space.
///
/// Note: The innovation covariance S = H*P*H' + R is computed in measurement space
/// and used for innovation vectors. We use `MeasurementCovariance` for S throughout
/// the codebase for consistency with the Kalman filter literature, where S is
/// considered to be in measurement space.
pub type InnovationCovariance<T, const M: usize> = Covariance<T, M, InnovationSpace>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_vector_operations() {
        let v1: StateVector<f64, 4> = StateVector::from_array([1.0, 2.0, 3.0, 4.0]);
        let v2: StateVector<f64, 4> = StateVector::from_array([0.5, 1.0, 1.5, 2.0]);

        let sum = v1 + v2;
        assert!((sum.index(0) - 1.5).abs() < 1e-10);
        assert!((sum.index(1) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_measurement_to_innovation() {
        let actual: Measurement<f64, 2> = Measurement::from_array([10.0, 20.0]);
        let predicted: Measurement<f64, 2> = Measurement::from_array([9.5, 19.0]);

        let innovation = actual.innovation(predicted);
        assert!((innovation.index(0) - 0.5).abs() < 1e-10);
        assert!((innovation.index(1) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_covariance_operations() {
        let cov: StateCovariance<f64, 2> = StateCovariance::identity();
        assert!((cov.trace() - 2.0).abs() < 1e-10);
        assert!((cov.determinant().unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_singular_covariance_determinant() {
        // A singular matrix should return None
        let singular: StateCovariance<f64, 2> =
            StateCovariance::from_matrix(nalgebra::matrix![1.0, 1.0; 1.0, 1.0]);
        assert!(singular.determinant().is_none());
    }
}
