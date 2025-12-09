//! Track updaters for LMB filters
//!
//! Different strategies for updating tracks based on association results.

use nalgebra::RealField;
use num_traits::Float;

#[cfg(feature = "alloc")]
#[allow(unused_imports)]
use alloc::vec::Vec;

use super::types::LmbTrack;
use crate::types::gaussian::GaussianState;
use crate::types::labels::BernoulliTrack;
use crate::types::spaces::{StateCovariance, StateVector};

// ============================================================================
// Track Updater Trait
// ============================================================================

/// Trait for updating tracks based on association results.
///
/// Different update strategies produce different posterior representations:
/// - [`MarginalUpdater`]: Creates Gaussian mixtures (for LMB)
/// - [`HardAssignmentUpdater`]: Uses single assignments (for LMBM)
#[cfg(feature = "alloc")]
pub trait TrackUpdater<T: RealField, const N: usize> {
    /// The type of track output by this updater
    type OutputTrack;

    /// Update a single track based on association weights and posteriors.
    ///
    /// # Arguments
    ///
    /// * `track` - The predicted track to update
    /// * `marginal_weights` - Association weights [miss, meas_0, meas_1, ...]
    /// * `posteriors` - Precomputed Kalman posteriors for each measurement
    fn update_track(
        &self,
        track: &LmbTrack<T, N>,
        marginal_weights: &[T],
        posteriors: &[(StateVector<T, N>, StateCovariance<T, N>, T)],
        updated_existence: T,
    ) -> Self::OutputTrack;
}

// ============================================================================
// Marginal Updater (for LMB)
// ============================================================================

/// Marginal updater that creates Gaussian mixture posteriors.
///
/// Used by the standard LMB filter. Each association event creates
/// a weighted component in the posterior mixture.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, Default)]
pub struct MarginalUpdater {
    /// Minimum weight to include a component
    pub weight_threshold: f64,
    /// Maximum components per track
    pub max_components: usize,
}

#[cfg(feature = "alloc")]
impl MarginalUpdater {
    /// Creates a new marginal updater with default settings.
    pub fn new() -> Self {
        Self {
            weight_threshold: 1e-10,
            max_components: 100,
        }
    }

    /// Creates a marginal updater with custom settings.
    pub fn with_settings(weight_threshold: f64, max_components: usize) -> Self {
        Self {
            weight_threshold,
            max_components,
        }
    }
}

#[cfg(feature = "alloc")]
impl<T: RealField + Float + Copy, const N: usize> TrackUpdater<T, N> for MarginalUpdater {
    type OutputTrack = LmbTrack<T, N>;

    fn update_track(
        &self,
        track: &LmbTrack<T, N>,
        marginal_weights: &[T],
        posteriors: &[(StateVector<T, N>, StateCovariance<T, N>, T)],
        updated_existence: T,
    ) -> LmbTrack<T, N> {
        let threshold = T::from(self.weight_threshold).unwrap();
        let mut new_components = crate::types::gaussian::GaussianMixture::with_capacity(
            posteriors.len() + track.num_components(),
        );

        // Miss detection component (index 0 in marginal_weights)
        let miss_weight = if marginal_weights.is_empty() {
            T::one()
        } else {
            marginal_weights[0]
        };

        if miss_weight > threshold {
            for component in track.components.iter() {
                new_components.push(GaussianState {
                    weight: component.weight * miss_weight,
                    mean: component.mean,
                    covariance: component.covariance,
                });
            }
        }

        // Detection components
        for (j, (mean, cov, _likelihood)) in posteriors.iter().enumerate() {
            let det_weight = marginal_weights.get(j + 1).copied().unwrap_or(T::zero());
            if det_weight > threshold {
                new_components.push(GaussianState {
                    weight: det_weight,
                    mean: *mean,
                    covariance: *cov,
                });
            }
        }

        // Normalize and truncate
        let total_weight = new_components.total_weight();
        if total_weight > T::zero() {
            for component in new_components.iter_mut() {
                component.weight /= total_weight;
            }
        }

        // Sort by weight and truncate if needed
        if new_components.len() > self.max_components {
            new_components
                .components
                .sort_by(|a, b| b.weight.partial_cmp(&a.weight).unwrap());
            new_components.components.truncate(self.max_components);

            // Renormalize after truncation
            let new_total = new_components.total_weight();
            if new_total > T::zero() {
                for component in new_components.iter_mut() {
                    component.weight /= new_total;
                }
            }
        }

        LmbTrack {
            label: track.label,
            existence: updated_existence,
            components: new_components,
        }
    }
}

// ============================================================================
// Hard Assignment Updater (for LMBM)
// ============================================================================

/// Hard assignment updater that creates single-component tracks.
///
/// Used by the LMBM filter. Takes the most likely association
/// and creates a single Gaussian posterior.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, Default)]
pub struct HardAssignmentUpdater;

#[cfg(feature = "alloc")]
impl HardAssignmentUpdater {
    /// Creates a new hard assignment updater.
    pub fn new() -> Self {
        Self
    }
}

#[cfg(feature = "alloc")]
impl<T: RealField + Float + Copy, const N: usize> TrackUpdater<T, N> for HardAssignmentUpdater {
    type OutputTrack = BernoulliTrack<T, N>;

    fn update_track(
        &self,
        track: &LmbTrack<T, N>,
        marginal_weights: &[T],
        posteriors: &[(StateVector<T, N>, StateCovariance<T, N>, T)],
        updated_existence: T,
    ) -> BernoulliTrack<T, N> {
        // Find the best assignment (highest weight)
        let (best_idx, _) = marginal_weights
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap_or((0, &T::zero()));

        let state = if best_idx == 0 {
            // Miss detection - use best existing component
            track.best_estimate().cloned().unwrap_or_else(|| {
                GaussianState::new(T::one(), StateVector::zeros(), StateCovariance::identity())
            })
        } else {
            // Detection - use posterior for measurement (best_idx - 1)
            let meas_idx = best_idx - 1;
            if meas_idx < posteriors.len() {
                let (mean, cov, _) = &posteriors[meas_idx];
                GaussianState::new(T::one(), *mean, *cov)
            } else {
                track.best_estimate().cloned().unwrap_or_else(|| {
                    GaussianState::new(T::one(), StateVector::zeros(), StateCovariance::identity())
                })
            }
        };

        BernoulliTrack {
            label: track.label,
            existence: updated_existence,
            state,
        }
    }
}

// ============================================================================
// Sampled Assignment Updater (for LMBM with Gibbs)
// ============================================================================

/// Sampled assignment updater for LMBM filter with Gibbs sampling.
///
/// Takes a specific assignment index (from a sample) rather than
/// the marginal-best assignment.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct SampledAssignmentUpdater {
    /// Assignment index: -1 for miss, >= 0 for measurement index
    pub assignment: i32,
}

#[cfg(feature = "alloc")]
impl SampledAssignmentUpdater {
    /// Creates an updater for a specific assignment.
    pub fn new(assignment: i32) -> Self {
        Self { assignment }
    }
}

#[cfg(feature = "alloc")]
impl<T: RealField + Float + Copy, const N: usize> TrackUpdater<T, N> for SampledAssignmentUpdater {
    type OutputTrack = BernoulliTrack<T, N>;

    fn update_track(
        &self,
        track: &LmbTrack<T, N>,
        _marginal_weights: &[T],
        posteriors: &[(StateVector<T, N>, StateCovariance<T, N>, T)],
        updated_existence: T,
    ) -> BernoulliTrack<T, N> {
        let state = if self.assignment < 0 {
            // Miss detection
            track.best_estimate().cloned().unwrap_or_else(|| {
                GaussianState::new(T::one(), StateVector::zeros(), StateCovariance::identity())
            })
        } else {
            // Detection
            let meas_idx = self.assignment as usize;
            if meas_idx < posteriors.len() {
                let (mean, cov, _) = &posteriors[meas_idx];
                GaussianState::new(T::one(), *mean, *cov)
            } else {
                track.best_estimate().cloned().unwrap_or_else(|| {
                    GaussianState::new(T::one(), StateVector::zeros(), StateCovariance::identity())
                })
            }
        };

        BernoulliTrack {
            label: track.label,
            existence: updated_existence,
            state,
        }
    }
}

// ============================================================================
// Existence Update Functions
// ============================================================================

/// Computes the updated existence probability for a missed detection.
///
/// r' = r * (1 - p_d) / (1 - r * p_d)
pub fn existence_update_miss<T: RealField + Float + Copy>(existence: T, detection_prob: T) -> T {
    let numerator = existence * (T::one() - detection_prob);
    let denominator = T::one() - existence * detection_prob;
    if denominator > T::zero() {
        numerator / denominator
    } else {
        existence
    }
}

/// Computes the updated existence probability with detection evidence.
///
/// r' = r * sum_weight / (1 - r + r * sum_weight)
/// where sum_weight includes both miss and detection terms.
pub fn existence_update_detection<T: RealField + Copy>(
    existence: T,
    detection_prob: T,
    likelihood_sum: T,
) -> T {
    let miss_factor = T::one() - detection_prob;
    let sum_weight = miss_factor + likelihood_sum;
    let denominator = T::one() - existence + existence * sum_weight;
    if denominator > T::zero() {
        existence * sum_weight / denominator
    } else {
        existence
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::labels::Label;

    #[cfg(feature = "alloc")]
    #[test]
    fn test_marginal_updater() {
        let label = Label::new(0, 0);
        let mean: StateVector<f64, 4> = StateVector::from_array([1.0, 2.0, 0.0, 0.0]);
        let cov: StateCovariance<f64, 4> = StateCovariance::identity();
        let state = GaussianState::new(1.0, mean, cov);
        let track = LmbTrack::new(label, 0.9, state);

        let updater = MarginalUpdater::new();
        let marginal_weights = vec![0.3, 0.7]; // 30% miss, 70% detection

        let posterior_mean: StateVector<f64, 4> = StateVector::from_array([1.5, 2.5, 0.1, 0.1]);
        let posterior_cov: StateCovariance<f64, 4> = StateCovariance::identity();
        let posteriors = vec![(posterior_mean, posterior_cov, 0.5)];

        let updated = updater.update_track(&track, &marginal_weights, &posteriors, 0.85);

        assert_eq!(updated.label, label);
        assert!((updated.existence - 0.85).abs() < 1e-10);
        assert_eq!(updated.num_components(), 2); // miss + detection
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_hard_assignment_updater() {
        let label = Label::new(0, 0);
        let mean: StateVector<f64, 4> = StateVector::from_array([1.0, 2.0, 0.0, 0.0]);
        let cov: StateCovariance<f64, 4> = StateCovariance::identity();
        let state = GaussianState::new(1.0, mean, cov);
        let track = LmbTrack::new(label, 0.9, state);

        let updater = HardAssignmentUpdater::new();
        let marginal_weights = vec![0.3, 0.7]; // Detection wins

        let posterior_mean: StateVector<f64, 4> = StateVector::from_array([1.5, 2.5, 0.1, 0.1]);
        let posterior_cov: StateCovariance<f64, 4> = StateCovariance::identity();
        let posteriors = vec![(posterior_mean, posterior_cov, 0.5)];

        let updated = updater.update_track(&track, &marginal_weights, &posteriors, 0.85);

        assert_eq!(updated.label, label);
        assert!((updated.existence - 0.85).abs() < 1e-10);
        // Should use posterior since detection weight > miss weight
        assert!((updated.state.mean.index(0) - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_existence_update_miss() {
        let r = 0.9_f64;
        let p_d = 0.95;
        let r_updated = existence_update_miss(r, p_d);

        // With high detection probability and miss, existence should decrease
        assert!(r_updated < r);
        assert!(r_updated > 0.0);
    }

    #[test]
    fn test_existence_update_detection() {
        let r = 0.5_f64;
        let p_d = 0.95;
        let likelihood_sum = 2.0; // Strong detection evidence

        let r_updated = existence_update_detection(r, p_d, likelihood_sum);

        // With strong detection evidence, existence should increase
        assert!(r_updated > r);
        assert!(r_updated <= 1.0);
    }
}
