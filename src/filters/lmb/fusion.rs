//! Multi-sensor fusion strategies for LMB filters
//!
//! Implements various track fusion methods for combining information
//! from multiple sensors.

use nalgebra::RealField;
use num_traits::Float;

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use super::types::LmbTrack;
use crate::types::gaussian::GaussianState;
#[allow(unused_imports)]
use crate::types::labels::Label;
use crate::types::spaces::{StateCovariance, StateVector};

// ============================================================================
// Merger Trait
// ============================================================================

/// Trait for merging tracks from multiple sensors.
///
/// Different fusion strategies handle sensor correlation differently:
/// - [`ArithmeticAverageMerger`]: Simple weighted average (conservative)
/// - [`GeometricAverageMerger`]: Covariance intersection (robust to correlation)
/// - [`ParallelUpdateMerger`]: Information filter fusion (optimal for independent sensors)
/// - [`IteratedCorrectorMerger`]: Sequential Bayesian updates
#[cfg(feature = "alloc")]
pub trait Merger<T: RealField, const N: usize> {
    /// Merge tracks from multiple sensors for the same label.
    ///
    /// # Arguments
    ///
    /// * `tracks` - Tracks from each sensor (same label)
    /// * `weights` - Relative weights for each sensor
    ///
    /// # Returns
    ///
    /// Fused track combining information from all sensors
    fn merge(&self, tracks: &[&LmbTrack<T, N>], weights: &[T]) -> LmbTrack<T, N>;
}

// ============================================================================
// Arithmetic Average Merger (AA-LMB)
// ============================================================================

/// Arithmetic average fusion strategy.
///
/// Computes a weighted average of existence probabilities and
/// concatenates Gaussian mixture components.
///
/// Pros: Fast, robust, simple
/// Cons: Ignores sensor correlation, may inflate uncertainties
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, Default)]
pub struct ArithmeticAverageMerger {
    /// Maximum number of components in fused mixture
    pub max_components: usize,
}

#[cfg(feature = "alloc")]
impl ArithmeticAverageMerger {
    /// Creates a new arithmetic average merger.
    pub fn new(max_components: usize) -> Self {
        Self { max_components }
    }
}

#[cfg(feature = "alloc")]
impl<T: RealField + Float + Copy, const N: usize> Merger<T, N> for ArithmeticAverageMerger {
    fn merge(&self, tracks: &[&LmbTrack<T, N>], weights: &[T]) -> LmbTrack<T, N> {
        assert!(!tracks.is_empty());
        assert_eq!(tracks.len(), weights.len());

        let label = tracks[0].label;

        // Normalize weights
        let weight_sum: T = weights.iter().fold(T::zero(), |acc, &w| acc + w);
        let normalized_weights: Vec<T> = if weight_sum > T::zero() {
            weights.iter().map(|&w| w / weight_sum).collect()
        } else {
            vec![T::one() / T::from(weights.len()).unwrap(); weights.len()]
        };

        // Weighted average of existence
        let fused_existence = tracks
            .iter()
            .zip(&normalized_weights)
            .fold(T::zero(), |acc, (track, &w)| acc + w * track.existence);

        // Concatenate and weight components
        let mut all_components = crate::types::gaussian::GaussianMixture::new();
        for (track, &sensor_weight) in tracks.iter().zip(&normalized_weights) {
            for component in track.components.iter() {
                all_components.push(GaussianState {
                    weight: component.weight * sensor_weight,
                    mean: component.mean,
                    covariance: component.covariance,
                });
            }
        }

        // Sort by weight and truncate
        all_components
            .components
            .sort_by(|a, b| b.weight.partial_cmp(&a.weight).unwrap());
        if all_components.len() > self.max_components {
            all_components.components.truncate(self.max_components);
        }

        // Normalize weights
        let total = all_components.total_weight();
        if total > T::zero() {
            for c in all_components.iter_mut() {
                c.weight /= total;
            }
        }

        LmbTrack {
            label,
            existence: fused_existence,
            components: all_components,
        }
    }
}

// ============================================================================
// Geometric Average Merger (GA-LMB)
// ============================================================================

/// Geometric average fusion using covariance intersection.
///
/// Computes fusion in canonical (information) form to handle
/// unknown sensor correlations conservatively.
///
/// Pros: Conservative, handles unknown correlations
/// Cons: Slower, requires matrix inversion
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, Default)]
pub struct GeometricAverageMerger;

#[cfg(feature = "alloc")]
impl GeometricAverageMerger {
    /// Creates a new geometric average merger.
    pub fn new() -> Self {
        Self
    }
}

#[cfg(feature = "alloc")]
impl<T: RealField + Float + Copy, const N: usize> Merger<T, N> for GeometricAverageMerger {
    fn merge(&self, tracks: &[&LmbTrack<T, N>], weights: &[T]) -> LmbTrack<T, N> {
        assert!(!tracks.is_empty());
        assert_eq!(tracks.len(), weights.len());

        let label = tracks[0].label;

        // Normalize weights
        let weight_sum: T = weights.iter().fold(T::zero(), |acc, &w| acc + w);
        let normalized_weights: Vec<T> = if weight_sum > T::zero() {
            weights.iter().map(|&w| w / weight_sum).collect()
        } else {
            vec![T::one() / T::from(weights.len()).unwrap(); weights.len()]
        };

        // Geometric mean of existence: r_fused = prod(r_i^w_i)
        let epsilon = T::from_f64(1e-10).unwrap();
        let fused_existence =
            tracks
                .iter()
                .zip(&normalized_weights)
                .fold(T::one(), |acc, (track, &w)| {
                    let clamped = if track.existence > epsilon {
                        track.existence
                    } else {
                        epsilon
                    };
                    acc * Float::powf(clamped, w)
                });

        // Fuse best components using covariance intersection
        // Get best component from each track
        let best_components: Vec<_> = tracks.iter().filter_map(|t| t.best_estimate()).collect();

        if best_components.is_empty() {
            return LmbTrack::new(
                label,
                fused_existence,
                GaussianState::new(T::one(), StateVector::zeros(), StateCovariance::identity()),
            );
        }

        // Covariance intersection in canonical form
        // K = sum(w_i * Sigma_i^{-1})
        // h = sum(w_i * Sigma_i^{-1} * mu_i)
        // Sigma_fused = K^{-1}
        // mu_fused = Sigma_fused * h

        let mut k_sum = nalgebra::SMatrix::<T, N, N>::zeros();
        let mut h_sum = nalgebra::SVector::<T, N>::zeros();

        for (component, &w) in best_components.iter().zip(&normalized_weights) {
            if let Some(cov_inv) = component.covariance.try_inverse() {
                k_sum += cov_inv.as_matrix().scale(w);
                h_sum += (cov_inv.as_matrix() * component.mean.as_svector()).scale(w);
            }
        }

        // Invert to get fused covariance
        let (fused_mean, fused_cov) = if let Some(k_inv) = k_sum.try_inverse() {
            let mean = StateVector::from_svector(k_inv * h_sum);
            let cov = StateCovariance::from_matrix(k_inv);
            (mean, cov)
        } else {
            // Fallback: use first track's best estimate
            let first = best_components[0];
            (first.mean, first.covariance)
        };

        let fused_state = GaussianState::new(T::one(), fused_mean, fused_cov);
        LmbTrack::new(label, fused_existence, fused_state)
    }
}

// ============================================================================
// Parallel Update Merger (PU-LMB)
// ============================================================================

/// Parallel update fusion using information filter.
///
/// Optimal for conditionally independent sensors. Fuses information
/// contributions while decorrelating common prior.
///
/// Pros: Theoretically optimal for independent sensors
/// Cons: Complex, slower
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct ParallelUpdateMerger<T: RealField, const N: usize> {
    /// Reference (prior) information matrix
    pub reference_info: StateCovariance<T, N>,
}

#[cfg(feature = "alloc")]
impl<T: RealField + Copy, const N: usize> ParallelUpdateMerger<T, N> {
    /// Creates a new parallel update merger with the given prior covariance.
    pub fn new(prior_covariance: StateCovariance<T, N>) -> Self {
        Self {
            reference_info: prior_covariance,
        }
    }

    /// Creates a merger with identity prior.
    pub fn with_identity_prior() -> Self {
        Self {
            reference_info: StateCovariance::identity(),
        }
    }
}

#[cfg(feature = "alloc")]
impl<T: RealField + Float + Copy, const N: usize> Merger<T, N> for ParallelUpdateMerger<T, N> {
    fn merge(&self, tracks: &[&LmbTrack<T, N>], weights: &[T]) -> LmbTrack<T, N> {
        assert!(!tracks.is_empty());
        assert_eq!(tracks.len(), weights.len());

        let label = tracks[0].label;
        let n_sensors = tracks.len();

        // Normalize weights
        let weight_sum: T = weights.iter().fold(T::zero(), |acc, &w| acc + w);
        let normalized_weights: Vec<T> = if weight_sum > T::zero() {
            weights.iter().map(|&w| w / weight_sum).collect()
        } else {
            vec![T::one() / T::from(n_sensors).unwrap(); n_sensors]
        };

        // Weighted average existence
        let fused_existence = tracks
            .iter()
            .zip(&normalized_weights)
            .fold(T::zero(), |acc, (track, &w)| acc + w * track.existence);

        // Get best components
        let best_components: Vec<_> = tracks.iter().filter_map(|t| t.best_estimate()).collect();

        if best_components.is_empty() {
            return LmbTrack::new(
                label,
                fused_existence,
                GaussianState::new(T::one(), StateVector::zeros(), StateCovariance::identity()),
            );
        }

        // Information filter fusion
        // Y_fused = sum(Y_i) - (S-1) * Y_ref
        // y_fused = sum(y_i) - (S-1) * y_ref
        // where Y_i = Sigma_i^{-1}, y_i = Y_i * mu_i

        let mut y_sum = nalgebra::SMatrix::<T, N, N>::zeros();
        let mut info_vec_sum = nalgebra::SVector::<T, N>::zeros();

        for component in &best_components {
            if let Some(cov_inv) = component.covariance.try_inverse() {
                y_sum += *cov_inv.as_matrix();
                info_vec_sum += cov_inv.as_matrix() * component.mean.as_svector();
            }
        }

        // Decorrelate by subtracting (S-1) times reference
        let s_minus_1 = T::from(n_sensors.saturating_sub(1)).unwrap();
        if let Some(ref_inv) = self.reference_info.try_inverse() {
            y_sum -= ref_inv.as_matrix().scale(s_minus_1);
            // Note: We don't have a reference mean, assume it's zero
        }

        // Convert back to moment form
        let (fused_mean, fused_cov) = if let Some(y_inv) = y_sum.try_inverse() {
            let mean = StateVector::from_svector(y_inv * info_vec_sum);
            let cov = StateCovariance::from_matrix(y_inv);
            (mean, cov)
        } else {
            let first = best_components[0];
            (first.mean, first.covariance)
        };

        let fused_state = GaussianState::new(T::one(), fused_mean, fused_cov);
        LmbTrack::new(label, fused_existence, fused_state)
    }
}

// ============================================================================
// Iterated Corrector Merger (IC-LMB)
// ============================================================================

/// Iterated corrector fusion using sequential Bayesian updates.
///
/// Applies sensor updates sequentially. Simple but order-dependent.
///
/// Pros: Simple
/// Cons: Order-dependent, not truly parallel
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, Default)]
pub struct IteratedCorrectorMerger;

#[cfg(feature = "alloc")]
impl IteratedCorrectorMerger {
    /// Creates a new iterated corrector merger.
    pub fn new() -> Self {
        Self
    }
}

#[cfg(feature = "alloc")]
impl<T: RealField + Float + Copy, const N: usize> Merger<T, N> for IteratedCorrectorMerger {
    fn merge(&self, tracks: &[&LmbTrack<T, N>], weights: &[T]) -> LmbTrack<T, N> {
        assert!(!tracks.is_empty());

        // Start with first track and sequentially update
        let mut current = tracks[0].clone();

        for (i, track) in tracks.iter().enumerate().skip(1) {
            let w = weights.get(i).copied().unwrap_or(T::one());

            // Update existence: blend toward track's existence
            current.existence = (T::one() - w) * current.existence + w * track.existence;

            // Fuse best component with Kalman-like update
            if let (Some(curr_best), Some(track_best)) =
                (current.best_estimate(), track.best_estimate())
            {
                // Treat track_best as a "measurement" with its covariance
                // This is a simplified fusion that blends means and covariances
                let alpha = w;
                let beta = T::one() - w;

                let fused_mean = StateVector::from_svector(
                    curr_best.mean.as_svector().scale(beta)
                        + track_best.mean.as_svector().scale(alpha),
                );

                let fused_cov = StateCovariance::from_matrix(
                    curr_best.covariance.as_matrix().scale(beta)
                        + track_best.covariance.as_matrix().scale(alpha),
                );

                current.components =
                    crate::types::gaussian::GaussianMixture::from_components(vec![
                        GaussianState::new(T::one(), fused_mean, fused_cov),
                    ]);
            }
        }

        current
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_track(label: Label, x: f64, existence: f64) -> LmbTrack<f64, 4> {
        let mean: StateVector<f64, 4> = StateVector::from_array([x, 0.0, 0.0, 0.0]);
        let cov: StateCovariance<f64, 4> = StateCovariance::identity();
        let state = GaussianState::new(1.0, mean, cov);
        LmbTrack::new(label, existence, state)
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_arithmetic_average_merger() {
        let label = Label::new(0, 0);
        let track1 = create_test_track(label, 1.0, 0.8);
        let track2 = create_test_track(label, 3.0, 0.6);

        let merger = ArithmeticAverageMerger::new(10);
        let fused = merger.merge(&[&track1, &track2], &[0.5, 0.5]);

        assert_eq!(fused.label, label);
        // Existence should be average: (0.8 + 0.6) / 2 = 0.7
        assert!((fused.existence - 0.7).abs() < 1e-10);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_geometric_average_merger() {
        let label = Label::new(0, 0);
        let track1 = create_test_track(label, 1.0, 0.9);
        let track2 = create_test_track(label, 2.0, 0.9);

        let merger = GeometricAverageMerger::new();
        let fused = merger.merge(&[&track1, &track2], &[0.5, 0.5]);

        assert_eq!(fused.label, label);
        // With equal weights and existence, geometric mean = 0.9
        assert!((fused.existence - 0.9).abs() < 1e-10);

        // Mean should be between 1.0 and 2.0
        let fused_mean = fused.weighted_mean();
        let fused_x = fused_mean.index(0);
        assert!(*fused_x >= 1.0 && *fused_x <= 2.0);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_parallel_update_merger() {
        let label = Label::new(0, 0);
        let track1 = create_test_track(label, 1.0, 0.8);
        let track2 = create_test_track(label, 2.0, 0.8);

        let merger = ParallelUpdateMerger::<f64, 4>::with_identity_prior();
        let fused = merger.merge(&[&track1, &track2], &[0.5, 0.5]);

        assert_eq!(fused.label, label);
        assert!((fused.existence - 0.8).abs() < 1e-10);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_iterated_corrector_merger() {
        let label = Label::new(0, 0);
        let track1 = create_test_track(label, 1.0, 0.8);
        let track2 = create_test_track(label, 3.0, 0.6);

        let merger = IteratedCorrectorMerger::new();
        let fused = merger.merge(&[&track1, &track2], &[0.5, 0.5]);

        assert_eq!(fused.label, label);
        // Should be somewhere between the two
        assert!(fused.existence > 0.5 && fused.existence < 0.9);
    }
}
