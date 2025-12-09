//! Association algorithm traits and result types
//!
//! Defines the common interface for data association algorithms used
//! in multi-target tracking.

use nalgebra::{ComplexField, RealField};
use num_traits::Float;

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use crate::TracktorError;

// ============================================================================
// Association Result
// ============================================================================

/// Result of a data association computation.
///
/// Contains marginal association weights, miss probabilities, and
/// optionally sampled associations for stochastic algorithms.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct AssociationResult<T: RealField> {
    /// Marginal association weights.
    /// `weights[i][j]` = P(track i associated with measurement j)
    /// where j=0 means "miss" (no detection).
    /// Shape: [n_tracks][n_measurements + 1]
    pub weights: Vec<Vec<T>>,

    /// Updated existence probabilities for each track.
    pub existence: Vec<T>,

    /// Number of iterations used by the algorithm.
    pub iterations: usize,

    /// Whether the algorithm converged (for iterative methods).
    pub converged: bool,

    /// Sampled associations (for stochastic algorithms).
    /// Each sample is a vector where `sample[i]` = measurement index for track i
    /// (-1 means miss detection).
    pub samples: Option<Vec<Vec<i32>>>,
}

#[cfg(feature = "alloc")]
impl<T: RealField + Copy> AssociationResult<T> {
    /// Creates a new association result.
    pub fn new(
        weights: Vec<Vec<T>>,
        existence: Vec<T>,
        iterations: usize,
        converged: bool,
    ) -> Self {
        Self {
            weights,
            existence,
            iterations,
            converged,
            samples: None,
        }
    }

    /// Creates a result with samples.
    pub fn with_samples(
        weights: Vec<Vec<T>>,
        existence: Vec<T>,
        iterations: usize,
        converged: bool,
        samples: Vec<Vec<i32>>,
    ) -> Self {
        Self {
            weights,
            existence,
            iterations,
            converged,
            samples: Some(samples),
        }
    }

    /// Returns the number of tracks.
    pub fn num_tracks(&self) -> usize {
        self.weights.len()
    }

    /// Returns the number of measurements.
    pub fn num_measurements(&self) -> usize {
        if self.weights.is_empty() {
            0
        } else {
            self.weights[0].len().saturating_sub(1)
        }
    }

    /// Returns the miss probability for a track.
    pub fn miss_probability(&self, track_idx: usize) -> Option<T> {
        self.weights.get(track_idx).and_then(|w| w.first().copied())
    }

    /// Returns the detection probability for a track-measurement pair.
    pub fn detection_probability(&self, track_idx: usize, meas_idx: usize) -> Option<T> {
        self.weights
            .get(track_idx)
            .and_then(|w| w.get(meas_idx + 1).copied())
    }
}

// ============================================================================
// Fixed-Size Association Result (no_std)
// ============================================================================

/// Fixed-size association result for no_std environments.
///
/// # Type Parameters
///
/// * `T` - Scalar type
/// * `N` - Maximum number of tracks
/// * `M` - Maximum number of measurements
pub struct FixedAssociationResult<T: RealField, const N: usize, const M: usize> {
    /// Marginal weights [track][measurement+1] (index 0 = miss)
    pub weights: [[T; M]; N],
    /// Updated existence probabilities
    pub existence: [T; N],
    /// Number of valid tracks
    pub num_tracks: usize,
    /// Number of valid measurements
    pub num_measurements: usize,
    /// Iterations used
    pub iterations: usize,
    /// Whether converged
    pub converged: bool,
}

impl<T: RealField + Copy, const N: usize, const M: usize> FixedAssociationResult<T, N, M> {
    /// Creates a new fixed-size result.
    pub fn new(num_tracks: usize, num_measurements: usize) -> Self {
        Self {
            weights: [[T::zero(); M]; N],
            existence: [T::zero(); N],
            num_tracks,
            num_measurements,
            iterations: 0,
            converged: false,
        }
    }
}

// ============================================================================
// Association Matrices
// ============================================================================

/// Precomputed matrices for data association algorithms.
///
/// These matrices are computed from track predictions and measurements
/// before running the association algorithm.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct AssociationMatrices<T: RealField> {
    /// Psi matrix: likelihood ratios.
    /// `psi[i][j]` = r_i * p_d * L(z_j | x_i) / kappa_j
    pub psi: Vec<Vec<T>>,

    /// Phi vector: missed detection factors.
    /// `phi[i]` = r_i * (1 - p_d)
    pub phi: Vec<T>,

    /// Eta vector: normalization factors.
    /// `eta[i]` = 1 - r_i * p_d
    pub eta: Vec<T>,

    /// Cost matrix for assignment algorithms.
    /// `cost[i][j]` = -log(L(z_j | x_i) * p_d / kappa_j)
    pub cost: Vec<Vec<f64>>,

    /// Detection probabilities for each track.
    pub detection_probs: Vec<T>,

    /// Clutter intensities for each measurement.
    pub clutter_intensities: Vec<T>,
}

#[cfg(feature = "alloc")]
impl<T: RealField + Float + Copy> AssociationMatrices<T> {
    /// Creates empty association matrices.
    pub fn new(num_tracks: usize, num_measurements: usize) -> Self {
        Self {
            psi: vec![vec![T::zero(); num_measurements]; num_tracks],
            phi: vec![T::zero(); num_tracks],
            eta: vec![T::zero(); num_tracks],
            cost: vec![vec![0.0; num_measurements]; num_tracks],
            detection_probs: vec![T::zero(); num_tracks],
            clutter_intensities: vec![T::zero(); num_measurements],
        }
    }

    /// Returns the number of tracks.
    pub fn num_tracks(&self) -> usize {
        self.psi.len()
    }

    /// Returns the number of measurements.
    pub fn num_measurements(&self) -> usize {
        if self.psi.is_empty() {
            0
        } else {
            self.psi[0].len()
        }
    }
}

// ============================================================================
// Associator Trait
// ============================================================================

/// Trait for data association algorithms.
///
/// Associators compute the marginal probabilities of track-to-measurement
/// associations from precomputed likelihood matrices.
#[cfg(feature = "alloc")]
pub trait Associator<T: RealField> {
    /// Configuration type for this associator.
    type Config;

    /// Computes association probabilities.
    ///
    /// # Arguments
    ///
    /// * `matrices` - Precomputed association matrices
    /// * `existence_probs` - Prior existence probabilities
    /// * `config` - Algorithm configuration
    ///
    /// # Returns
    ///
    /// Association result with marginal weights and updated existence.
    fn associate(
        &self,
        matrices: &AssociationMatrices<T>,
        existence_probs: &[T],
        config: &Self::Config,
    ) -> Result<AssociationResult<T>, TracktorError>;
}

// ============================================================================
// Stochastic Associator Trait
// ============================================================================

// Note: StochasticAssociator is not included in the base crate as it requires
// the `rand` crate. Stochastic associators like Gibbs sampling can be implemented
// by users when they add `rand` as a dependency.

// ============================================================================
// LBP Associator
// ============================================================================

/// Loopy Belief Propagation configuration.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct LbpConfig<T> {
    /// Convergence tolerance.
    pub tolerance: T,
    /// Maximum number of iterations.
    pub max_iterations: usize,
}

#[cfg(feature = "alloc")]
impl<T: RealField> Default for LbpConfig<T> {
    fn default() -> Self {
        Self {
            tolerance: T::from_f64(1e-6).unwrap(),
            max_iterations: 50,
        }
    }
}

/// Loopy Belief Propagation associator.
///
/// A deterministic message-passing algorithm for computing marginal
/// association probabilities. Fast and typically converges quickly.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, Default)]
pub struct LbpAssociator;

#[cfg(feature = "alloc")]
impl<T: RealField + Float + Copy> Associator<T> for LbpAssociator {
    type Config = LbpConfig<T>;

    fn associate(
        &self,
        matrices: &AssociationMatrices<T>,
        existence_probs: &[T],
        config: &Self::Config,
    ) -> Result<AssociationResult<T>, TracktorError> {
        let n_tracks = matrices.num_tracks();
        let n_meas = matrices.num_measurements();

        if n_tracks == 0 {
            return Ok(AssociationResult::new(Vec::new(), Vec::new(), 0, true));
        }

        // Initialize messages
        let mut sigma_mt: Vec<Vec<T>> = vec![vec![T::one(); n_tracks]; n_meas];
        let mut sigma_tm: Vec<Vec<T>> = vec![vec![T::one(); n_meas]; n_tracks];

        let mut iterations = 0;
        let mut converged = false;

        for iter in 0..config.max_iterations {
            iterations = iter + 1;
            let sigma_mt_old = sigma_mt.clone();

            // Compute B = psi .* sigma_mt
            let mut b: Vec<Vec<T>> = vec![vec![T::zero(); n_meas]; n_tracks];
            for i in 0..n_tracks {
                for j in 0..n_meas {
                    b[i][j] = matrices.psi[i][j] * sigma_mt[j][i];
                }
            }

            // Row sums of B
            let b_row_sums: Vec<T> = b
                .iter()
                .map(|row| row.iter().fold(T::zero(), |acc, &x| acc + x))
                .collect();

            // Update sigma_tm
            for i in 0..n_tracks {
                for j in 0..n_meas {
                    let denom = -b[i][j] + b_row_sums[i] + T::one();
                    sigma_tm[i][j] = if denom > T::zero() {
                        matrices.psi[i][j] / denom
                    } else {
                        T::one()
                    };
                }
            }

            // Column sums of sigma_tm
            let mut sigma_tm_col_sums = vec![T::zero(); n_meas];
            for row in sigma_tm.iter().take(n_tracks) {
                for j in 0..n_meas {
                    sigma_tm_col_sums[j] += row[j];
                }
            }

            // Update sigma_mt
            for j in 0..n_meas {
                for i in 0..n_tracks {
                    let denom = -sigma_tm[i][j] + sigma_tm_col_sums[j] + T::one();
                    sigma_mt[j][i] = if denom > T::zero() {
                        T::one() / denom
                    } else {
                        T::one()
                    };
                }
            }

            // Check convergence
            let mut max_delta = T::zero();
            for j in 0..n_meas {
                for i in 0..n_tracks {
                    let delta = ComplexField::abs(sigma_mt[j][i] - sigma_mt_old[j][i]);
                    if delta > max_delta {
                        max_delta = delta;
                    }
                }
            }

            if max_delta < config.tolerance {
                converged = true;
                break;
            }
        }

        // Compute final B
        let mut b: Vec<Vec<T>> = vec![vec![T::zero(); n_meas]; n_tracks];
        for i in 0..n_tracks {
            for j in 0..n_meas {
                b[i][j] = matrices.psi[i][j] * sigma_mt[j][i];
            }
        }

        // Compute marginal weights and existence
        let mut weights = Vec::with_capacity(n_tracks);
        let mut updated_existence = Vec::with_capacity(n_tracks);

        for i in 0..n_tracks {
            let mut gamma = Vec::with_capacity(n_meas + 1);
            gamma.push(matrices.phi[i]); // Miss

            for b_val in b[i].iter().take(n_meas) {
                gamma.push(*b_val * matrices.eta[i]);
            }

            let gamma_sum: T = gamma.iter().fold(T::zero(), |acc, &x| acc + x);

            // Marginal weights
            let w: Vec<T> = if gamma_sum > T::zero() {
                gamma.iter().map(|&g| g / gamma_sum).collect()
            } else {
                let mut w = vec![T::zero(); n_meas + 1];
                w[0] = T::one();
                w
            };
            weights.push(w);

            // Updated existence
            let denom = matrices.eta[i] + gamma_sum - matrices.phi[i];
            let r_updated = if denom > T::zero() {
                gamma_sum / denom
            } else {
                existence_probs[i]
            };
            updated_existence.push(r_updated);
        }

        Ok(AssociationResult::new(
            weights,
            updated_existence,
            iterations,
            converged,
        ))
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "alloc")]
    #[test]
    fn test_association_result() {
        let weights = vec![
            vec![0.3, 0.7], // Track 0: 30% miss, 70% detection
            vec![0.8, 0.2], // Track 1: 80% miss, 20% detection
        ];
        let existence = vec![0.9, 0.5];

        let result = AssociationResult::new(weights, existence, 10, true);

        assert_eq!(result.num_tracks(), 2);
        assert_eq!(result.num_measurements(), 1);
        assert!(Float::abs(result.miss_probability(0).unwrap() - 0.3) < 1e-10);
        assert!(Float::abs(result.detection_probability(0, 0).unwrap() - 0.7) < 1e-10);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_lbp_empty() {
        let matrices = AssociationMatrices::<f64>::new(0, 0);
        let associator = LbpAssociator;
        let config = LbpConfig::default();

        let result = associator.associate(&matrices, &[], &config).unwrap();

        assert_eq!(result.num_tracks(), 0);
        assert!(result.converged);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_lbp_single_track_single_measurement() {
        let mut matrices = AssociationMatrices::<f64>::new(1, 1);

        // High likelihood ratio
        matrices.psi[0][0] = 10.0; // Strong association
        matrices.phi[0] = 0.05; // Low miss probability component
        matrices.eta[0] = 0.95; // High eta

        let existence_probs = vec![0.9];
        let associator = LbpAssociator;
        let config = LbpConfig::default();

        let result = associator
            .associate(&matrices, &existence_probs, &config)
            .unwrap();

        assert_eq!(result.num_tracks(), 1);
        assert!(result.converged);

        // Should have high detection probability
        let miss_prob = result.miss_probability(0).unwrap();
        let det_prob = result.detection_probability(0, 0).unwrap();
        assert!(det_prob > miss_prob);
    }
}
