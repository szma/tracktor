//! Cardinality estimation for LMB filters
//!
//! Implements the Elementary Symmetric Function (ESF) and MAP cardinality
//! estimation algorithms for LMB densities.

use nalgebra::RealField;

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

// ============================================================================
// Elementary Symmetric Function (ESF)
// ============================================================================

/// Computes the elementary symmetric function of a set of values.
///
/// The elementary symmetric function e_k(z) is the sum of all products
/// of k distinct elements from z. For LMB cardinality estimation,
/// z_i = r_i / (1 - r_i) where r_i is the existence probability.
///
/// Uses Mahler's recursive formula with a two-row buffer for efficiency:
/// ```text
/// e_0 = 1
/// e_k(z_1, ..., z_n) = e_k(z_1, ..., z_{n-1}) + z_n * e_{k-1}(z_1, ..., z_{n-1})
/// ```
///
/// # Arguments
///
/// * `z` - Input values (typically r/(1-r) for each track)
///
/// # Returns
///
/// Vector [e_0, e_1, ..., e_n] where e_k is the k-th elementary symmetric function.
#[cfg(feature = "alloc")]
pub fn elementary_symmetric_function<T: RealField + Copy>(z: &[T]) -> Vec<T> {
    if z.is_empty() {
        return vec![T::one()];
    }

    let n = z.len();

    // Use two-row buffer to avoid allocating full n x n matrix
    let mut f_prev = vec![T::zero(); n + 1];
    let mut f_curr = vec![T::zero(); n + 1];
    f_prev[0] = T::one();

    for (i, &z_i) in z.iter().enumerate() {
        // e_0 is always 1
        f_curr[0] = T::one();

        // Compute e_k for k = 1..=i+1
        for k in 1..=i + 1 {
            // e_k = e_k(without z_i) + z_i * e_{k-1}(without z_i)
            f_curr[k] = f_prev[k] + z_i * f_prev[k - 1];
        }

        // Swap buffers
        core::mem::swap(&mut f_prev, &mut f_curr);
    }

    // Result is in f_prev after final swap
    f_prev.truncate(n + 1);
    f_prev
}

/// Fixed-size elementary symmetric function for no_std environments.
///
/// # Type Parameters
///
/// * `T` - Scalar type
/// * `N` - Maximum number of elements (tracks)
pub fn elementary_symmetric_function_fixed<T: RealField + Copy, const N: usize>(z: &[T]) -> [T; N] {
    let mut result = [T::zero(); N];
    result[0] = T::one();

    if z.is_empty() || N == 0 {
        return result;
    }

    let mut f_prev = [T::zero(); N];
    let mut f_curr = [T::zero(); N];
    f_prev[0] = T::one();

    let max_k = z.len().min(N - 1);

    for (i, &z_i) in z.iter().enumerate() {
        f_curr[0] = T::one();

        let k_max = (i + 1).min(max_k);
        for k in 1..=k_max {
            f_curr[k] = f_prev[k] + z_i * f_prev[k - 1];
        }

        core::mem::swap(&mut f_prev, &mut f_curr);
    }

    for k in 0..N.min(z.len() + 1) {
        result[k] = f_prev[k];
    }
    result
}

// ============================================================================
// LMB Cardinality Distribution
// ============================================================================

/// Computes the cardinality distribution for an LMB density.
///
/// The cardinality distribution rho(n) gives the probability of having
/// exactly n targets:
/// ```text
/// rho(n) = prod_i(1 - r_i) * e_n(r_1/(1-r_1), ..., r_N/(1-r_N))
/// ```
///
/// # Arguments
///
/// * `existence_probs` - Existence probabilities r_i for each track
///
/// # Returns
///
/// Vector [rho(0), rho(1), ..., rho(N)] where rho(k) is P(exactly k targets exist).
#[cfg(feature = "alloc")]
pub fn lmb_cardinality_distribution<T: RealField + Copy>(existence_probs: &[T]) -> Vec<T> {
    if existence_probs.is_empty() {
        return vec![T::one()]; // rho(0) = 1 when no tracks
    }

    // Clamp existence probabilities to avoid division by zero
    let epsilon = T::from_f64(1e-10).unwrap();
    let one_minus_eps = T::one() - epsilon;

    // Compute z_i = r_i / (1 - r_i) and prod(1 - r_i)
    let mut z = Vec::with_capacity(existence_probs.len());
    let mut prod_1_minus_r = T::one();

    for &r in existence_probs {
        // Clamp r to (epsilon, 1 - epsilon)
        let r_clamped = if r < epsilon {
            epsilon
        } else if r > one_minus_eps {
            one_minus_eps
        } else {
            r
        };
        let one_minus_r = T::one() - r_clamped;
        z.push(r_clamped / one_minus_r);
        prod_1_minus_r *= one_minus_r;
    }

    // Compute ESF
    let esf = elementary_symmetric_function(&z);

    // Scale by prod(1 - r_i)
    esf.into_iter().map(|e| prod_1_minus_r * e).collect()
}

// ============================================================================
// MAP Cardinality Estimation
// ============================================================================

/// Result of MAP cardinality estimation.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct MapCardinalityResult<T> {
    /// The MAP estimate of the number of targets
    pub cardinality: usize,
    /// Indices of the tracks to extract (sorted by existence probability)
    pub track_indices: Vec<usize>,
    /// The cardinality distribution
    pub distribution: Vec<T>,
}

/// Computes the MAP (Maximum A Posteriori) cardinality estimate.
///
/// Finds the most likely number of targets and returns the indices
/// of the most likely tracks to include.
///
/// # Arguments
///
/// * `existence_probs` - Existence probabilities r_i for each track
///
/// # Returns
///
/// * `cardinality` - MAP estimate of target count
/// * `track_indices` - Indices of the n_map highest existence tracks
/// * `distribution` - Full cardinality distribution
#[cfg(feature = "alloc")]
pub fn map_cardinality_estimate<T: RealField + Copy>(
    existence_probs: &[T],
) -> MapCardinalityResult<T> {
    if existence_probs.is_empty() {
        return MapCardinalityResult {
            cardinality: 0,
            track_indices: Vec::new(),
            distribution: vec![T::one()],
        };
    }

    // Compute cardinality distribution
    let distribution = lmb_cardinality_distribution(existence_probs);

    // Find MAP cardinality (argmax of distribution)
    let n_map = distribution
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);

    // Get indices sorted by existence probability (descending)
    let mut indices: Vec<usize> = (0..existence_probs.len()).collect();
    indices.sort_by(|&a, &b| existence_probs[b].partial_cmp(&existence_probs[a]).unwrap());
    indices.truncate(n_map);

    MapCardinalityResult {
        cardinality: n_map,
        track_indices: indices,
        distribution,
    }
}

/// Simple cardinality estimate by rounding total existence mass.
///
/// This is a faster but less accurate alternative to MAP estimation.
/// Returns round(sum(r_i)).
pub fn simple_cardinality_estimate<T: RealField + Copy>(existence_probs: &[T]) -> usize {
    let total: T = existence_probs.iter().fold(T::zero(), |acc, &r| acc + r);
    // Use RealField's floor and add 0.5 for rounding
    let half = T::from_f64(0.5).unwrap();
    let rounded = (total + half).floor();
    rounded.to_subset().unwrap_or(0.0) as usize
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "alloc")]
    #[test]
    fn test_esf_empty() {
        let z: Vec<f64> = vec![];
        let esf = elementary_symmetric_function(&z);
        assert_eq!(esf.len(), 1);
        assert!((esf[0] - 1.0).abs() < 1e-10);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_esf_single() {
        let z = vec![2.0_f64];
        let esf = elementary_symmetric_function(&z);
        assert_eq!(esf.len(), 2);
        assert!((esf[0] - 1.0).abs() < 1e-10); // e_0 = 1
        assert!((esf[1] - 2.0).abs() < 1e-10); // e_1 = z_1 = 2
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_esf_two_elements() {
        let z = vec![2.0_f64, 3.0];
        let esf = elementary_symmetric_function(&z);
        assert_eq!(esf.len(), 3);
        assert!((esf[0] - 1.0).abs() < 1e-10); // e_0 = 1
        assert!((esf[1] - 5.0).abs() < 1e-10); // e_1 = 2 + 3 = 5
        assert!((esf[2] - 6.0).abs() < 1e-10); // e_2 = 2 * 3 = 6
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_esf_three_elements() {
        let z = vec![1.0_f64, 2.0, 3.0];
        let esf = elementary_symmetric_function(&z);
        assert_eq!(esf.len(), 4);
        assert!((esf[0] - 1.0).abs() < 1e-10); // e_0 = 1
        assert!((esf[1] - 6.0).abs() < 1e-10); // e_1 = 1+2+3 = 6
        assert!((esf[2] - 11.0).abs() < 1e-10); // e_2 = 1*2 + 1*3 + 2*3 = 11
        assert!((esf[3] - 6.0).abs() < 1e-10); // e_3 = 1*2*3 = 6
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_cardinality_distribution_sums_to_one() {
        let existence_probs = vec![0.9_f64, 0.8, 0.7, 0.2];
        let rho = lmb_cardinality_distribution(&existence_probs);

        let sum: f64 = rho.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "Cardinality distribution should sum to 1, got {}",
            sum
        );
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_cardinality_distribution_all_zeros() {
        let existence_probs = vec![0.0_f64, 0.0, 0.0];
        let rho = lmb_cardinality_distribution(&existence_probs);

        // With all r_i ≈ 0, rho(0) should be ≈ 1
        assert!(rho[0] > 0.99);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_cardinality_distribution_all_ones() {
        let existence_probs = vec![0.99_f64, 0.99, 0.99];
        let rho = lmb_cardinality_distribution(&existence_probs);

        // With all r_i ≈ 1, rho(3) should be highest
        let n_map = rho
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        assert_eq!(n_map, 3);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_map_cardinality_estimate() {
        let existence_probs = vec![0.9_f64, 0.85, 0.1, 0.05];
        let result = map_cardinality_estimate(&existence_probs);

        // Should estimate 2 targets (the two with high existence)
        assert_eq!(result.cardinality, 2);
        assert_eq!(result.track_indices.len(), 2);

        // Indices should be 0 and 1 (highest existence)
        assert!(result.track_indices.contains(&0));
        assert!(result.track_indices.contains(&1));
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_simple_cardinality() {
        let existence_probs = vec![0.9_f64, 0.8, 0.3];
        let n = simple_cardinality_estimate(&existence_probs);
        assert_eq!(n, 2); // round(0.9 + 0.8 + 0.3) = round(2.0) = 2
    }

    #[test]
    fn test_esf_fixed() {
        let z = [2.0_f64, 3.0];
        let esf: [f64; 4] = elementary_symmetric_function_fixed(&z);
        assert!((esf[0] - 1.0).abs() < 1e-10);
        assert!((esf[1] - 5.0).abs() < 1e-10);
        assert!((esf[2] - 6.0).abs() < 1e-10);
        assert!((esf[3] - 0.0).abs() < 1e-10); // Beyond input size
    }
}
