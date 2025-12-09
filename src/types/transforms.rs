//! Typed transformation matrices
//!
//! Matrices that transform vectors between spaces, with type-level
//! encoding of source and target spaces.

use ::core::marker::PhantomData;
use nalgebra::{SMatrix, RealField, Scalar};

use super::spaces::{
    Vector, StateSpace, MeasurementSpace, InnovationSpace,
    StateVector, Measurement, Innovation,
    StateCovariance, MeasurementCovariance,
};

// ============================================================================
// Transform Matrix
// ============================================================================

/// A transformation matrix that maps vectors from one space to another.
///
/// # Type Parameters
///
/// - `T`: Scalar type
/// - `ROWS`: Number of rows (dimension of target space)
/// - `COLS`: Number of columns (dimension of source space)
/// - `To`: Target space marker
/// - `From`: Source space marker
#[repr(transparent)]
#[derive(Debug, Clone, PartialEq)]
pub struct Transform<T: Scalar, const ROWS: usize, const COLS: usize, To, From> {
    inner: SMatrix<T, ROWS, COLS>,
    _marker: PhantomData<(To, From)>,
}

impl<T: Scalar, const ROWS: usize, const COLS: usize, To, From> Transform<T, ROWS, COLS, To, From> {
    /// Creates a transform from a raw matrix.
    #[inline]
    pub fn from_matrix(inner: SMatrix<T, ROWS, COLS>) -> Self {
        Self {
            inner,
            _marker: PhantomData,
        }
    }

    /// Returns a reference to the underlying matrix.
    #[inline]
    pub fn as_matrix(&self) -> &SMatrix<T, ROWS, COLS> {
        &self.inner
    }

    /// Consumes self and returns the underlying matrix.
    #[inline]
    pub fn into_matrix(self) -> SMatrix<T, ROWS, COLS> {
        self.inner
    }
}

impl<T: Scalar + Copy, const ROWS: usize, const COLS: usize, To: Clone, From: Clone> Copy
    for Transform<T, ROWS, COLS, To, From>
where SMatrix<T, ROWS, COLS>: Copy {}

impl<T: RealField + Copy, const ROWS: usize, const COLS: usize, To, From>
    Transform<T, ROWS, COLS, To, From>
{
    /// Returns the transpose of this transform.
    ///
    /// The transpose maps from `To` to `From` (reversed).
    #[inline]
    pub fn transpose(&self) -> Transform<T, COLS, ROWS, From, To> {
        Transform {
            inner: self.inner.transpose(),
            _marker: PhantomData,
        }
    }
}

// ============================================================================
// Type Aliases
// ============================================================================

/// State transition matrix: StateSpace -> StateSpace
pub type TransitionMatrix<T, const N: usize> = Transform<T, N, N, StateSpace, StateSpace>;

/// Observation matrix: StateSpace -> MeasurementSpace
pub type ObservationMatrix<T, const M: usize, const N: usize> =
    Transform<T, M, N, MeasurementSpace, StateSpace>;

/// Kalman gain: InnovationSpace -> StateSpace
pub type KalmanGain<T, const N: usize, const M: usize> =
    Transform<T, N, M, StateSpace, InnovationSpace>;

// ============================================================================
// Matrix-Vector Multiplication: Transform * Vector
// ============================================================================

/// Trait for applying a transformation to a vector.
pub trait ApplyTransform<T: RealField, const ROWS: usize, const COLS: usize, To, From> {
    type Output;
    fn apply(&self, v: &Vector<T, COLS, From>) -> Self::Output;
}

impl<T: RealField + Copy, const ROWS: usize, const COLS: usize, To, From>
    ApplyTransform<T, ROWS, COLS, To, From> for Transform<T, ROWS, COLS, To, From>
{
    type Output = Vector<T, ROWS, To>;

    #[inline]
    fn apply(&self, v: &Vector<T, COLS, From>) -> Self::Output {
        Vector::from_svector(self.inner * v.as_svector())
    }
}

// ============================================================================
// Specific Transform Applications
// ============================================================================

impl<T: RealField + Copy, const N: usize> TransitionMatrix<T, N> {
    /// Creates an identity transition matrix.
    #[inline]
    pub fn identity() -> Self {
        Self {
            inner: SMatrix::identity(),
            _marker: PhantomData,
        }
    }

    /// Creates a zero transition matrix.
    #[inline]
    pub fn zeros() -> Self {
        Self {
            inner: SMatrix::zeros(),
            _marker: PhantomData,
        }
    }

    /// Applies the transition to a state vector.
    #[inline]
    pub fn apply_state(&self, state: &StateVector<T, N>) -> StateVector<T, N> {
        StateVector::from_svector(self.inner * state.as_svector())
    }

    /// Propagates a covariance matrix: F * P * F^T
    #[inline]
    pub fn propagate_covariance(&self, cov: &StateCovariance<T, N>) -> StateCovariance<T, N> {
        StateCovariance::from_matrix(
            self.inner * cov.as_matrix() * self.inner.transpose()
        )
    }
}

impl<T: RealField + Copy, const M: usize, const N: usize> ObservationMatrix<T, M, N> {
    /// Creates a zero observation matrix.
    #[inline]
    pub fn zeros() -> Self {
        Self {
            inner: SMatrix::zeros(),
            _marker: PhantomData,
        }
    }

    /// Applies the observation model to a state vector.
    #[inline]
    pub fn observe(&self, state: &StateVector<T, N>) -> Measurement<T, M> {
        Measurement::from_svector(self.inner * state.as_svector())
    }

    /// Projects state covariance to measurement space: H * P * H^T
    #[inline]
    pub fn project_covariance(&self, cov: &StateCovariance<T, N>) -> MeasurementCovariance<T, M> {
        MeasurementCovariance::from_matrix(
            self.inner * cov.as_matrix() * self.inner.transpose()
        )
    }
}

impl<T: RealField + Copy, const N: usize, const M: usize> KalmanGain<T, N, M> {
    /// Creates a zero Kalman gain matrix.
    #[inline]
    pub fn zeros() -> Self {
        Self {
            inner: SMatrix::zeros(),
            _marker: PhantomData,
        }
    }

    /// Applies the Kalman gain to an innovation vector.
    #[inline]
    pub fn correct(&self, innovation: &Innovation<T, M>) -> StateVector<T, N> {
        StateVector::from_svector(self.inner * innovation.as_svector())
    }
}

// ============================================================================
// Kalman Gain Computation
// ============================================================================

/// Computes the Kalman gain matrix.
///
/// K = P * H^T * S^{-1}
///
/// where:
/// - P is the predicted state covariance
/// - H is the observation matrix
/// - S is the innovation covariance (H * P * H^T + R)
pub fn compute_kalman_gain<T: RealField + Copy, const N: usize, const M: usize>(
    state_cov: &StateCovariance<T, N>,
    obs_matrix: &ObservationMatrix<T, M, N>,
    innovation_cov: &MeasurementCovariance<T, M>,
) -> Option<KalmanGain<T, N, M>> {
    // S^{-1}
    let s_inv = innovation_cov.as_matrix().try_inverse()?;

    // K = P * H^T * S^{-1}
    let k = state_cov.as_matrix() * obs_matrix.as_matrix().transpose() * s_inv;

    Some(KalmanGain::from_matrix(k))
}

/// Computes the innovation covariance.
///
/// S = H * P * H^T + R
pub fn compute_innovation_covariance<T: RealField + Copy, const N: usize, const M: usize>(
    state_cov: &StateCovariance<T, N>,
    obs_matrix: &ObservationMatrix<T, M, N>,
    meas_noise: &MeasurementCovariance<T, M>,
) -> MeasurementCovariance<T, M> {
    let h_p_ht = obs_matrix.project_covariance(state_cov);
    MeasurementCovariance::from_matrix(h_p_ht.as_matrix() + meas_noise.as_matrix())
}

/// Updates state covariance using Joseph form for numerical stability.
///
/// P_updated = (I - K*H) * P * (I - K*H)^T + K * R * K^T
pub fn joseph_update<T: RealField + Copy, const N: usize, const M: usize>(
    state_cov: &StateCovariance<T, N>,
    kalman_gain: &KalmanGain<T, N, M>,
    obs_matrix: &ObservationMatrix<T, M, N>,
    meas_noise: &MeasurementCovariance<T, M>,
) -> StateCovariance<T, N> {
    let i: SMatrix<T, N, N> = SMatrix::identity();
    let k_h = kalman_gain.as_matrix() * obs_matrix.as_matrix();
    let i_kh = i - k_h;

    let term1 = i_kh * state_cov.as_matrix() * i_kh.transpose();
    let term2 = kalman_gain.as_matrix() * meas_noise.as_matrix() * kalman_gain.as_matrix().transpose();

    StateCovariance::from_matrix(term1 + term2)
}

/// Simple covariance update (less numerically stable but faster).
///
/// P_updated = (I - K*H) * P
pub fn simple_covariance_update<T: RealField + Copy, const N: usize, const M: usize>(
    state_cov: &StateCovariance<T, N>,
    kalman_gain: &KalmanGain<T, N, M>,
    obs_matrix: &ObservationMatrix<T, M, N>,
) -> StateCovariance<T, N> {
    let i: SMatrix<T, N, N> = SMatrix::identity();
    let k_h = kalman_gain.as_matrix() * obs_matrix.as_matrix();
    let i_kh = i - k_h;

    StateCovariance::from_matrix(i_kh * state_cov.as_matrix())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transition_matrix() {
        // Simple 2D constant velocity model
        let dt = 1.0_f64;
        let f = TransitionMatrix::<f64, 4>::from_matrix(nalgebra::matrix![
            1.0, 0.0, dt, 0.0;
            0.0, 1.0, 0.0, dt;
            0.0, 0.0, 1.0, 0.0;
            0.0, 0.0, 0.0, 1.0
        ]);

        let state = StateVector::from_array([0.0, 0.0, 1.0, 2.0]);
        let predicted = f.apply_state(&state);

        assert!((predicted.index(0) - 1.0).abs() < 1e-10);
        assert!((predicted.index(1) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_observation_matrix() {
        // Observe position only from [x, y, vx, vy]
        let h = ObservationMatrix::<f64, 2, 4>::from_matrix(nalgebra::matrix![
            1.0, 0.0, 0.0, 0.0;
            0.0, 1.0, 0.0, 0.0
        ]);

        let state = StateVector::from_array([10.0, 20.0, 1.0, 2.0]);
        let measurement = h.observe(&state);

        assert!((measurement.index(0) - 10.0).abs() < 1e-10);
        assert!((measurement.index(1) - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_kalman_gain_application() {
        let k = KalmanGain::<f64, 4, 2>::from_matrix(nalgebra::matrix![
            0.5, 0.0;
            0.0, 0.5;
            0.1, 0.0;
            0.0, 0.1
        ]);

        let innovation = Innovation::from_array([2.0, 4.0]);
        let correction = k.correct(&innovation);

        assert!((correction.index(0) - 1.0).abs() < 1e-10);
        assert!((correction.index(1) - 2.0).abs() < 1e-10);
    }
}
