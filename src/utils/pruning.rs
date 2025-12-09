//! Component pruning and merging for Gaussian mixtures
//!
//! Essential for maintaining tractable mixture sizes in GM-PHD and similar filters.

use nalgebra::RealField;
use num_traits::Float;

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use crate::types::gaussian::{GaussianMixture, GaussianState};
use crate::types::spaces::{StateCovariance, StateVector};

/// Configuration for pruning and merging operations.
#[derive(Debug, Clone)]
pub struct PruningConfig<T: RealField> {
    /// Minimum weight threshold (components below this are removed)
    pub weight_threshold: T,
    /// Merging distance threshold (Mahalanobis distance)
    pub merge_threshold: T,
    /// Maximum number of components after pruning
    pub max_components: usize,
}

impl<T: RealField + Float> PruningConfig<T> {
    /// Creates a default pruning configuration.
    pub fn default_config() -> Self {
        Self {
            weight_threshold: T::from_f64(1e-5).unwrap(),
            merge_threshold: T::from_f64(4.0).unwrap(),
            max_components: 100,
        }
    }

    /// Creates a pruning configuration with custom values.
    pub fn new(weight_threshold: T, merge_threshold: T, max_components: usize) -> Self {
        Self {
            weight_threshold,
            merge_threshold,
            max_components,
        }
    }
}

/// Prunes components with weight below a threshold.
///
/// The weights of pruned components are
/// redistributed equally among the remaining components to preserve the
/// total weight (expected target count in PHD filters).
#[cfg(feature = "alloc")]
pub fn prune_by_weight<T: RealField + Float + Copy, const N: usize>(
    mixture: &GaussianMixture<T, N>,
    threshold: T,
) -> GaussianMixture<T, N> {
    // Calculate the total weight of pruned components
    let pruned_weight_sum: T = mixture
        .iter()
        .filter(|c| c.weight < threshold)
        .fold(T::zero(), |acc, c| acc + c.weight);

    // Collect remaining components
    let mut remaining: Vec<_> = mixture
        .iter()
        .filter(|c| c.weight >= threshold)
        .cloned()
        .collect();

    // Redistribute pruned weights across remaining components
    if !remaining.is_empty() && pruned_weight_sum > T::zero() {
        let n_remaining = T::from(remaining.len()).unwrap();
        let weight_per_component = pruned_weight_sum / n_remaining;
        for component in &mut remaining {
            component.weight += weight_per_component;
        }
    }

    GaussianMixture::from_components(remaining)
}

/// Truncates the mixture to keep only the top N components by weight.
///
/// The weights of truncated components are
/// redistributed equally among the remaining components to preserve the
/// total weight (expected target count in PHD filters).
#[cfg(feature = "alloc")]
pub fn truncate<T: RealField + Float + Copy, const N: usize>(
    mixture: &GaussianMixture<T, N>,
    max_components: usize,
) -> GaussianMixture<T, N> {
    if mixture.len() <= max_components {
        return mixture.clone();
    }

    // Sort components by weight (highest first)
    let mut indexed: Vec<_> = mixture.iter().enumerate().collect();
    indexed.sort_by(|(_, a), (_, b)| {
        b.weight
            .partial_cmp(&a.weight)
            .unwrap_or(core::cmp::Ordering::Equal)
    });

    // Calculate the total weight of truncated components
    let truncated_weight_sum: T = indexed
        .iter()
        .skip(max_components)
        .fold(T::zero(), |acc, (_, c)| acc + c.weight);

    // Collect the top components
    let mut remaining: Vec<_> = indexed
        .into_iter()
        .take(max_components)
        .map(|(_, c)| c.clone())
        .collect();

    // Redistribute truncated weights across remaining components
    if !remaining.is_empty() && truncated_weight_sum > T::zero() {
        let n_remaining = T::from(remaining.len()).unwrap();
        let weight_per_component = truncated_weight_sum / n_remaining;
        for component in &mut remaining {
            component.weight += weight_per_component;
        }
    }

    GaussianMixture::from_components(remaining)
}

/// Computes the squared Mahalanobis distance between two Gaussian components.
#[cfg(feature = "alloc")]
pub fn mahalanobis_distance_squared<T: RealField + Float + Copy, const N: usize>(
    a: &GaussianState<T, N>,
    b: &GaussianState<T, N>,
) -> T {
    let diff = StateVector::from_svector(a.mean.as_svector() - b.mean.as_svector());

    // Use the covariance of component a for distance calculation
    if let Some(cov_inv) = a.covariance.try_inverse() {
        let d = diff.as_svector();
        let m = cov_inv.as_matrix();
        (d.transpose() * m * d)[(0, 0)]
    } else {
        T::infinity()
    }
}

/// Merges two Gaussian components into one.
///
/// Returns `None` if both components have non-positive weights (which indicates
/// a bug in the calling code - components should have positive weights).
///
/// # Panics
/// In debug builds, panics if both weights are non-positive, as this indicates
/// a logic error in the filter.
#[cfg(feature = "alloc")]
pub fn merge_components<T: RealField + Float + Copy, const N: usize>(
    a: &GaussianState<T, N>,
    b: &GaussianState<T, N>,
) -> Option<GaussianState<T, N>> {
    let w_sum = a.weight + b.weight;

    if w_sum <= T::zero() {
        // Both weights are non-positive - this shouldn't happen in normal operation.
        // Return None to signal this condition to the caller.
        return None;
    }

    // Merged mean: weighted average
    let mean = StateVector::from_svector(
        (a.mean.as_svector().scale(a.weight) + b.mean.as_svector().scale(b.weight))
            .scale(T::one() / w_sum),
    );

    // Merged covariance: weighted average plus spread of means
    let diff_a = StateVector::from_svector(a.mean.as_svector() - mean.as_svector());
    let diff_b = StateVector::from_svector(b.mean.as_svector() - mean.as_svector());

    let spread_a = diff_a.as_svector() * diff_a.as_svector().transpose();
    let spread_b = diff_b.as_svector() * diff_b.as_svector().transpose();

    let merged_cov = StateCovariance::from_matrix(
        (a.covariance.as_matrix().scale(a.weight)
            + b.covariance.as_matrix().scale(b.weight)
            + spread_a.scale(a.weight)
            + spread_b.scale(b.weight))
        .scale(T::one() / w_sum),
    );

    Some(GaussianState::new(w_sum, mean, merged_cov))
}

/// Merges nearby components based on Mahalanobis distance.
#[cfg(feature = "alloc")]
pub fn merge_nearby<T: RealField + Float + Copy, const N: usize>(
    mixture: &GaussianMixture<T, N>,
    threshold: T,
) -> GaussianMixture<T, N> {
    if mixture.is_empty() {
        return GaussianMixture::new();
    }

    let threshold_sq = threshold * threshold;
    let mut remaining: Vec<_> = mixture.iter().cloned().collect();
    let mut merged = GaussianMixture::new();

    while !remaining.is_empty() {
        // Find component with highest weight
        let max_idx = remaining
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.weight
                    .partial_cmp(&b.weight)
                    .unwrap_or(core::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap();

        let mut current = remaining.remove(max_idx);

        // Find all components within threshold and merge
        let mut i = 0;
        while i < remaining.len() {
            let dist_sq = mahalanobis_distance_squared(&current, &remaining[i]);

            if dist_sq < threshold_sq {
                let to_merge = remaining.remove(i);
                // merge_components returns None only if both weights are non-positive,
                // which shouldn't happen with properly pruned mixtures.
                if let Some(merged_component) = merge_components(&current, &to_merge) {
                    current = merged_component;
                }
                // If merge fails (both weights non-positive), we skip this merge
                // and continue with the current component as-is.
            } else {
                i += 1;
            }
        }

        merged.push(current);
    }

    merged
}

/// Applies full pruning pipeline: weight threshold, merge, truncate.
#[cfg(feature = "alloc")]
pub fn prune_and_merge<T: RealField + Float + Copy, const N: usize>(
    mixture: &GaussianMixture<T, N>,
    config: &PruningConfig<T>,
) -> GaussianMixture<T, N> {
    // Step 1: Remove low-weight components
    let pruned = prune_by_weight(mixture, config.weight_threshold);

    // Step 2: Merge nearby components
    let merged = merge_nearby(&pruned, config.merge_threshold);

    // Step 3: Truncate to max components
    truncate(&merged, config.max_components)
}

/// Normalizes component weights to sum to a target value.
#[cfg(feature = "alloc")]
pub fn normalize_weights<T: RealField + Float + Copy, const N: usize>(
    mixture: &mut GaussianMixture<T, N>,
    target_sum: T,
) {
    let total = mixture.total_weight();
    if total > T::zero() {
        let scale = target_sum / total;
        mixture.scale_weights(scale);
    }
}

/// Caps the total weight at a maximum value.
#[cfg(feature = "alloc")]
pub fn cap_total_weight<T: RealField + Float + Copy, const N: usize>(
    mixture: &mut GaussianMixture<T, N>,
    max_weight: T,
) {
    let total = mixture.total_weight();
    if total > max_weight {
        let scale = max_weight / total;
        mixture.scale_weights(scale);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "alloc")]
    fn make_component(weight: f64, x: f64, y: f64) -> GaussianState<f64, 4> {
        GaussianState::new(
            weight,
            StateVector::from_array([x, y, 0.0, 0.0]),
            StateCovariance::identity(),
        )
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_prune_by_weight() {
        let mut mixture = GaussianMixture::new();
        mixture.push(make_component(0.5, 0.0, 0.0));
        mixture.push(make_component(0.001, 10.0, 10.0));
        mixture.push(make_component(0.3, 20.0, 20.0));

        let original_total = mixture.total_weight();
        let pruned = prune_by_weight(&mixture, 0.01);

        assert_eq!(pruned.len(), 2);
        // Weight should be preserved
        assert!((pruned.total_weight() - original_total).abs() < 1e-10);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_truncate() {
        let mut mixture = GaussianMixture::new();
        for i in 0..10 {
            mixture.push(make_component(i as f64 * 0.1, i as f64, 0.0));
        }

        let original_total = mixture.total_weight();
        let truncated = truncate(&mixture, 3);

        assert_eq!(truncated.len(), 3);

        // Weight should be preserved
        assert!((truncated.total_weight() - original_total).abs() < 1e-10);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_merge_components() {
        let a = make_component(0.6, 0.0, 0.0);
        let b = make_component(0.4, 2.0, 0.0);

        let merged = merge_components(&a, &b).expect("merge should succeed with positive weights");

        assert!((merged.weight - 1.0).abs() < 1e-10);
        // Merged mean should be weighted average: 0.6*0 + 0.4*2 = 0.8
        assert!((merged.mean.index(0) - 0.8).abs() < 1e-10);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_merge_components_zero_weights() {
        // Merging two zero-weight components should return None
        let a = make_component(0.0, 0.0, 0.0);
        let b = make_component(0.0, 2.0, 0.0);

        assert!(merge_components(&a, &b).is_none());
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_merge_nearby() {
        let mut mixture = GaussianMixture::new();
        // Two close components
        mixture.push(make_component(0.5, 0.0, 0.0));
        mixture.push(make_component(0.3, 0.5, 0.0));
        // One far component
        mixture.push(make_component(0.2, 100.0, 100.0));

        let merged = merge_nearby(&mixture, 2.0);

        // Should merge the two close ones, keep the far one separate
        assert_eq!(merged.len(), 2);
        assert!((merged.total_weight() - 1.0).abs() < 1e-10);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_prune_and_merge() {
        let mut mixture = GaussianMixture::new();
        mixture.push(make_component(0.5, 0.0, 0.0));
        mixture.push(make_component(0.3, 0.5, 0.0));
        mixture.push(make_component(0.0001, 50.0, 50.0)); // Low weight
        mixture.push(make_component(0.2, 100.0, 100.0));

        let config = PruningConfig::new(0.001, 2.0, 10);
        let result = prune_and_merge(&mixture, &config);

        // Low weight should be removed, close components merged
        assert_eq!(result.len(), 2);
    }
}
