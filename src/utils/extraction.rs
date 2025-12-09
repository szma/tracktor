//! State extraction strategies for multi-target tracking
//!
//! Various methods for extracting target state estimates from
//! probability distributions.

use nalgebra::RealField;
use num_traits::Float;

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use crate::types::gaussian::{GaussianMixture, GaussianState};
use crate::types::spaces::{StateCovariance, StateVector};

/// A target state estimate with associated uncertainty.
#[derive(Debug, Clone)]
pub struct TargetEstimate<T: RealField, const N: usize> {
    /// Estimated state vector
    pub state: StateVector<T, N>,
    /// State covariance (uncertainty)
    pub covariance: StateCovariance<T, N>,
    /// Confidence/weight
    pub confidence: T,
}

impl<T: RealField + Copy, const N: usize> TargetEstimate<T, N> {
    /// Creates a new target estimate.
    pub fn new(state: StateVector<T, N>, covariance: StateCovariance<T, N>, confidence: T) -> Self {
        Self {
            state,
            covariance,
            confidence,
        }
    }

    /// Creates an estimate from a Gaussian component.
    pub fn from_gaussian(gaussian: &GaussianState<T, N>) -> Self {
        Self {
            state: gaussian.mean,
            covariance: gaussian.covariance,
            confidence: gaussian.weight,
        }
    }
}

/// Extraction method for converting mixtures to point estimates.
#[derive(Debug, Clone, Copy)]
pub enum ExtractionMethod {
    /// Extract components with weight above threshold
    WeightThreshold,
    /// Extract N highest-weighted components
    TopN,
    /// Extract based on expected target count (rounds total weight)
    ExpectedCount,
    /// Extract local maxima in the mixture
    LocalMaxima,
}

/// Configuration for state extraction.
#[derive(Debug, Clone)]
pub struct ExtractionConfig<T: RealField> {
    /// Extraction method
    pub method: ExtractionMethod,
    /// Weight threshold (for WeightThreshold and LocalMaxima methods)
    pub weight_threshold: T,
    /// Number of targets to extract (for TopN method)
    pub top_n: usize,
    /// Mahalanobis distance squared threshold for local maxima merging
    /// Components within this distance of a higher-weight component are not considered maxima
    pub local_maxima_dist_sq: T,
}

impl<T: RealField + Float> ExtractionConfig<T> {
    /// Creates a weight threshold configuration.
    pub fn weight_threshold(threshold: T) -> Self {
        Self {
            method: ExtractionMethod::WeightThreshold,
            weight_threshold: threshold,
            top_n: 0,
            local_maxima_dist_sq: T::from_f64(4.0).unwrap(),
        }
    }

    /// Creates a top-N configuration.
    pub fn top_n(n: usize) -> Self {
        Self {
            method: ExtractionMethod::TopN,
            weight_threshold: T::zero(),
            top_n: n,
            local_maxima_dist_sq: T::from_f64(4.0).unwrap(),
        }
    }

    /// Creates an expected count configuration.
    pub fn expected_count() -> Self {
        Self {
            method: ExtractionMethod::ExpectedCount,
            weight_threshold: T::from_f64(0.5).unwrap(),
            top_n: 0,
            local_maxima_dist_sq: T::from_f64(4.0).unwrap(),
        }
    }

    /// Creates a local maxima configuration.
    ///
    /// # Arguments
    /// - `min_weight`: Minimum weight for a component to be considered
    /// - `merge_dist_sq`: Squared Mahalanobis distance threshold. Components within
    ///   this distance of a higher-weight component are suppressed.
    ///   Default is 4.0 (corresponds to ~2 sigma for Gaussian distributions).
    pub fn local_maxima(min_weight: T, merge_dist_sq: T) -> Self {
        Self {
            method: ExtractionMethod::LocalMaxima,
            weight_threshold: min_weight,
            top_n: 0,
            local_maxima_dist_sq: merge_dist_sq,
        }
    }
}

/// Extracts target estimates from a Gaussian mixture.
#[cfg(feature = "alloc")]
pub fn extract_targets<T: RealField + Float + Copy, const N: usize>(
    mixture: &GaussianMixture<T, N>,
    config: &ExtractionConfig<T>,
) -> Vec<TargetEstimate<T, N>> {
    match config.method {
        ExtractionMethod::WeightThreshold => extract_by_threshold(mixture, config.weight_threshold),
        ExtractionMethod::TopN => extract_top_n(mixture, config.top_n),
        ExtractionMethod::ExpectedCount => {
            let n = num_traits::Float::round(mixture.total_weight())
                .to_usize()
                .unwrap_or(0);
            extract_top_n(mixture, n)
        }
        ExtractionMethod::LocalMaxima => extract_local_maxima_with_dist(
            mixture,
            config.weight_threshold,
            config.local_maxima_dist_sq,
        ),
    }
}

/// Extracts components with weight above threshold.
#[cfg(feature = "alloc")]
pub fn extract_by_threshold<T: RealField + Float + Copy, const N: usize>(
    mixture: &GaussianMixture<T, N>,
    threshold: T,
) -> Vec<TargetEstimate<T, N>> {
    mixture
        .iter()
        .filter(|c| c.weight >= threshold)
        .map(TargetEstimate::from_gaussian)
        .collect()
}

/// Extracts the N highest-weighted components.
#[cfg(feature = "alloc")]
pub fn extract_top_n<T: RealField + Float + Copy, const N: usize>(
    mixture: &GaussianMixture<T, N>,
    n: usize,
) -> Vec<TargetEstimate<T, N>> {
    if n == 0 {
        return Vec::new();
    }

    let mut indexed: Vec<_> = mixture.iter().enumerate().collect();
    indexed.sort_by(|(_, a), (_, b)| {
        b.weight
            .partial_cmp(&a.weight)
            .unwrap_or(core::cmp::Ordering::Equal)
    });

    indexed
        .into_iter()
        .take(n)
        .map(|(_, c)| TargetEstimate::from_gaussian(c))
        .collect()
}

/// Extracts local maxima (components that are not close to higher-weighted components).
///
/// Uses a default merge distance of 4.0 (squared Mahalanobis distance).
/// For configurable distance, use `extract_local_maxima_with_dist`.
#[cfg(feature = "alloc")]
pub fn extract_local_maxima<T: RealField + Float + Copy, const N: usize>(
    mixture: &GaussianMixture<T, N>,
    min_weight: T,
) -> Vec<TargetEstimate<T, N>> {
    extract_local_maxima_with_dist(mixture, min_weight, T::from_f64(4.0).unwrap())
}

/// Extracts local maxima with a configurable merge distance.
///
/// Components within `merge_dist_sq` (squared Mahalanobis distance) of a
/// higher-weighted component are suppressed.
#[cfg(feature = "alloc")]
pub fn extract_local_maxima_with_dist<T: RealField + Float + Copy, const N: usize>(
    mixture: &GaussianMixture<T, N>,
    min_weight: T,
    merge_dist_sq: T,
) -> Vec<TargetEstimate<T, N>> {
    use crate::utils::pruning::mahalanobis_distance_squared;

    // Sort by weight descending
    let mut sorted: Vec<_> = mixture
        .iter()
        .filter(|c| c.weight >= min_weight)
        .cloned()
        .collect();

    sorted.sort_by(|a, b| {
        b.weight
            .partial_cmp(&a.weight)
            .unwrap_or(core::cmp::Ordering::Equal)
    });

    let mut maxima = Vec::new();

    for component in sorted {
        // Check if this component is a local maximum
        let is_local_max = maxima.iter().all(|m: &TargetEstimate<T, N>| {
            let temp = GaussianState::new(T::one(), m.state, m.covariance);
            mahalanobis_distance_squared(&component, &temp) > merge_dist_sq
        });

        if is_local_max {
            maxima.push(TargetEstimate::from_gaussian(&component));
        }
    }

    maxima
}

/// Estimates the number of targets from a mixture.
#[cfg(feature = "alloc")]
pub fn estimate_cardinality<T: RealField + Float + Copy, const N: usize>(
    mixture: &GaussianMixture<T, N>,
) -> usize {
    num_traits::Float::round(mixture.total_weight())
        .to_usize()
        .unwrap_or(0)
}

/// Computes the mean state from a mixture (treating all components equally).
#[cfg(feature = "alloc")]
pub fn mixture_mean<T: RealField + Float + Copy, const N: usize>(
    mixture: &GaussianMixture<T, N>,
) -> Option<StateVector<T, N>> {
    if mixture.is_empty() {
        return None;
    }

    let total_weight = mixture.total_weight();
    if total_weight <= T::zero() {
        return None;
    }

    let mut sum = nalgebra::SVector::<T, N>::zeros();
    for component in mixture.iter() {
        sum += component.mean.as_svector().scale(component.weight);
    }

    Some(StateVector::from_svector(
        sum.scale(T::one() / total_weight),
    ))
}

/// Computes the spread (covariance) of the mixture.
#[cfg(feature = "alloc")]
pub fn mixture_covariance<T: RealField + Float + Copy, const N: usize>(
    mixture: &GaussianMixture<T, N>,
) -> Option<StateCovariance<T, N>> {
    let mean = mixture_mean(mixture)?;
    let total_weight = mixture.total_weight();

    if total_weight <= T::zero() {
        return None;
    }

    let mut cov_sum = nalgebra::SMatrix::<T, N, N>::zeros();

    for component in mixture.iter() {
        let diff = component.mean.as_svector() - mean.as_svector();
        let spread = diff * diff.transpose();

        // Weighted covariance + spread of means
        cov_sum += (component.covariance.as_matrix() + spread).scale(component.weight);
    }

    Some(StateCovariance::from_matrix(
        cov_sum.scale(T::one() / total_weight),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "alloc")]
    fn make_gaussian(weight: f64, x: f64, y: f64) -> GaussianState<f64, 4> {
        GaussianState::new(
            weight,
            StateVector::from_array([x, y, 0.0, 0.0]),
            StateCovariance::identity(),
        )
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_extract_by_threshold() {
        let mut mixture = GaussianMixture::new();
        mixture.push(make_gaussian(0.8, 0.0, 0.0));
        mixture.push(make_gaussian(0.3, 10.0, 10.0));
        mixture.push(make_gaussian(0.1, 20.0, 20.0));

        let targets = extract_by_threshold(&mixture, 0.5);

        assert_eq!(targets.len(), 1);
        assert!((targets[0].state.index(0) - 0.0).abs() < 1e-10);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_extract_top_n() {
        let mut mixture = GaussianMixture::new();
        mixture.push(make_gaussian(0.3, 0.0, 0.0));
        mixture.push(make_gaussian(0.8, 10.0, 10.0));
        mixture.push(make_gaussian(0.5, 20.0, 20.0));

        let targets = extract_top_n(&mixture, 2);

        assert_eq!(targets.len(), 2);
        // First should be highest weight (0.8)
        assert!((targets[0].confidence - 0.8).abs() < 1e-10);
        // Second should be 0.5
        assert!((targets[1].confidence - 0.5).abs() < 1e-10);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_estimate_cardinality() {
        let mut mixture = GaussianMixture::new();
        mixture.push(make_gaussian(0.8, 0.0, 0.0));
        mixture.push(make_gaussian(0.7, 10.0, 10.0));
        mixture.push(make_gaussian(0.6, 20.0, 20.0));

        // Total weight = 2.1, rounds to 2
        assert_eq!(estimate_cardinality(&mixture), 2);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_mixture_mean() {
        let mut mixture = GaussianMixture::new();
        mixture.push(make_gaussian(0.5, 0.0, 0.0));
        mixture.push(make_gaussian(0.5, 10.0, 0.0));

        let mean = mixture_mean(&mixture).unwrap();

        // Weighted mean: (0.5*0 + 0.5*10) / 1.0 = 5.0
        assert!((mean.index(0) - 5.0).abs() < 1e-10);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_extract_config() {
        let mut mixture = GaussianMixture::new();
        mixture.push(make_gaussian(0.9, 0.0, 0.0));
        mixture.push(make_gaussian(0.1, 10.0, 10.0));

        let config = ExtractionConfig::expected_count();
        let targets = extract_targets(&mixture, &config);

        // Total weight = 1.0, expect 1 target
        assert_eq!(targets.len(), 1);
    }
}
