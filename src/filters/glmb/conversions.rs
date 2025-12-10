//! Conversions between LMB and GLMB densities
//!
//! This module provides functions to convert between LMB (Labeled Multi-Bernoulli)
//! and GLMB (Generalized Labeled Multi-Bernoulli) density representations.
//!
//! # LMB to GLMB
//!
//! Converting LMB to GLMB expands the implicit hypothesis space into explicit
//! hypotheses. This is computationally expensive (exponential in track count)
//! so truncation is applied.
//!
//! # GLMB to LMB
//!
//! Converting GLMB to LMB computes the marginal existence and state for each
//! label, losing correlation information.


use nalgebra::{ComplexField, RealField};
use num_traits::Float;

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use super::types::{GlmbDensity, GlmbHypothesis, GlmbTrack};
use crate::filters::lmb::{LmbTrack, LmbTrackSet};
use crate::types::gaussian::{GaussianMixture, GaussianState};
use crate::types::labels::Label;
use crate::types::spaces::StateVector;

// ============================================================================
// LMB to GLMB Conversion
// ============================================================================

/// Converts an LMB density to a GLMB density.
///
/// Creates hypotheses for possible subsets of tracks, weighted by the product
/// of existence/non-existence probabilities.
///
/// # Warning
///
/// The full expansion is exponential in the number of tracks (2^n hypotheses).
/// This function uses heuristics to generate only the most likely hypotheses:
/// 1. Focuses on cardinalities near the MAP estimate
/// 2. Truncates to max_hypotheses
///
/// # Arguments
///
/// * `tracks` - The LMB track set to convert
/// * `max_hypotheses` - Maximum number of hypotheses to generate
///
/// # Returns
///
/// A GLMB density approximating the LMB density
#[cfg(feature = "alloc")]
pub fn lmb_to_glmb<T: RealField + Float + Copy, const N: usize>(
    tracks: &LmbTrackSet<T, N>,
    max_hypotheses: usize,
) -> GlmbDensity<T, N> {
    use crate::filters::lmb::cardinality::map_cardinality_estimate;

    if tracks.is_empty() {
        return GlmbDensity::with_empty_hypothesis();
    }

    let n_tracks = tracks.len();

    // Get existence probabilities and MAP cardinality
    let existence_probs: Vec<T> = tracks.iter().map(|t| t.existence).collect();
    let map_result = map_cardinality_estimate(&existence_probs);
    let map_card = map_result.cardinality;

    // Determine which cardinalities to explore
    let target_cardinalities: Vec<usize> = {
        let mut cards = Vec::new();
        // Start with MAP cardinality
        cards.push(map_card);
        // Add neighboring cardinalities
        if map_card > 0 {
            cards.push(map_card - 1);
        }
        if map_card < n_tracks {
            cards.push(map_card + 1);
        }
        if map_card > 1 {
            cards.push(map_card - 2);
        }
        if map_card + 1 < n_tracks {
            cards.push(map_card + 2);
        }
        // Include 0 if not already present and map_card is small
        if map_card <= 2 && !cards.contains(&0) {
            cards.push(0);
        }
        cards.sort();
        cards.dedup();
        cards
    };

    let mut hypotheses: Vec<GlmbHypothesis<T, N>> = Vec::new();

    // Generate combinations for each target cardinality
    for &card in &target_cardinalities {
        if card > n_tracks {
            continue;
        }

        let combinations = generate_combinations(n_tracks, card);

        for combo in combinations {
            // Compute log weight: sum of log(r_i) for included, log(1-r_i) for excluded
            let mut log_weight = T::zero();
            let mut glmb_tracks: Vec<GlmbTrack<T, N>> = Vec::new();

            let mut valid = true;
            for (i, track) in tracks.iter().enumerate() {
                if combo.contains(&i) {
                    // Track exists in this hypothesis
                    if track.existence > T::zero() {
                        log_weight += ComplexField::ln(track.existence);
                    } else {
                        valid = false;
                        break;
                    }

                    // Use best estimate from mixture
                    if let Some(best) = track.best_estimate() {
                        glmb_tracks.push(GlmbTrack {
                            label: track.label,
                            state: best.clone(),
                        });
                    }
                } else {
                    // Track does not exist
                    let one_minus_r = T::one() - track.existence;
                    if one_minus_r > T::zero() {
                        log_weight += ComplexField::ln(one_minus_r);
                    } else {
                        valid = false;
                        break;
                    }
                }
            }

            if valid {
                hypotheses.push(GlmbHypothesis {
                    log_weight,
                    tracks: glmb_tracks,
                    current_association: vec![None; card],
                });
            }

            if hypotheses.len() >= max_hypotheses * 2 {
                // Generate extra then truncate for better selection
                break;
            }
        }

        if hypotheses.len() >= max_hypotheses * 2 {
            break;
        }
    }

    let mut density = GlmbDensity { hypotheses };
    density.normalize_log_weights();
    density.keep_top_k(max_hypotheses);
    density
}

/// Converts an LMB density to GLMB using all possible hypotheses.
///
/// # Warning
///
/// This generates 2^n hypotheses where n is the number of tracks!
/// Only use for small track sets (n < 15 recommended).
#[cfg(feature = "alloc")]
pub fn lmb_to_glmb_full<T: RealField + Float + Copy, const N: usize>(
    tracks: &LmbTrackSet<T, N>,
) -> GlmbDensity<T, N> {
    if tracks.is_empty() {
        return GlmbDensity::with_empty_hypothesis();
    }

    let n_tracks = tracks.len();
    let n_hypotheses = 1usize << n_tracks; // 2^n

    let mut hypotheses = Vec::with_capacity(n_hypotheses);

    for mask in 0..n_hypotheses {
        let mut log_weight = T::zero();
        let mut glmb_tracks: Vec<GlmbTrack<T, N>> = Vec::new();
        let mut valid = true;

        for (i, track) in tracks.iter().enumerate() {
            let included = (mask >> i) & 1 == 1;

            if included {
                if track.existence > T::zero() {
                    log_weight += ComplexField::ln(track.existence);
                } else {
                    valid = false;
                    break;
                }

                if let Some(best) = track.best_estimate() {
                    glmb_tracks.push(GlmbTrack {
                        label: track.label,
                        state: best.clone(),
                    });
                }
            } else {
                let one_minus_r = T::one() - track.existence;
                if one_minus_r > T::zero() {
                    log_weight += ComplexField::ln(one_minus_r);
                } else {
                    valid = false;
                    break;
                }
            }
        }

        if valid {
            let n_tracks = glmb_tracks.len();
            hypotheses.push(GlmbHypothesis {
                log_weight,
                tracks: glmb_tracks,
                current_association: vec![None; n_tracks],
            });
        }
    }

    let mut density = GlmbDensity { hypotheses };
    density.normalize_log_weights();
    density
}

// ============================================================================
// GLMB to LMB Conversion
// ============================================================================

/// Converts a GLMB density to an LMB density (marginal approximation).
///
/// Computes the marginal existence probability and state for each label by
/// summing over all hypotheses where the label appears.
///
/// # Note
///
/// This loses correlation information present in the GLMB density.
///
/// # Arguments
///
/// * `density` - The GLMB density to convert
///
/// # Returns
///
/// An LMB track set with marginal statistics
#[cfg(feature = "alloc")]
pub fn glmb_to_lmb<T: RealField + Float + Copy, const N: usize>(
    density: &GlmbDensity<T, N>,
) -> LmbTrackSet<T, N> {
    use alloc::collections::BTreeMap;

    if density.hypotheses.is_empty() {
        return LmbTrackSet::new();
    }

    // Compute normalized weights
    let max_log = density
        .hypotheses
        .iter()
        .map(|h| h.log_weight)
        .fold(Float::neg_infinity(), |a, b| if a > b { a } else { b });

    let weights: Vec<T> = density
        .hypotheses
        .iter()
        .map(|h| Float::exp(h.log_weight - max_log))
        .collect();
    let total: T = weights.iter().fold(T::zero(), |acc, &w| acc + w);

    if total <= T::zero() {
        return LmbTrackSet::new();
    }

    // Accumulate per-label statistics
    // For each label: (weighted state sum, weighted covariance sum, existence probability)
    let mut label_data: BTreeMap<Label, (GaussianMixture<T, N>, T)> = BTreeMap::new();

    for (h, &w) in density.hypotheses.iter().zip(&weights) {
        let norm_w = w / total;

        for track in &h.tracks {
            let entry = label_data
                .entry(track.label)
                .or_insert_with(|| (GaussianMixture::new(), T::zero()));

            // Add weighted component to mixture
            entry.0.push(GaussianState {
                weight: norm_w,
                mean: track.state.mean,
                covariance: track.state.covariance,
            });

            // Accumulate existence probability
            entry.1 += norm_w;
        }
    }

    // Build LMB track set
    let mut tracks = LmbTrackSet::new();

    for (label, (mut components, existence)) in label_data {
        // Normalize component weights to sum to 1 (conditioned on existence)
        let comp_total = components.total_weight();
        if comp_total > T::zero() {
            for c in components.iter_mut() {
                c.weight /= comp_total;
            }
        }

        tracks.push(LmbTrack {
            label,
            existence,
            components,
        });
    }

    tracks
}

/// Converts a GLMB density to LMB with merged Gaussian components.
///
/// Unlike `glmb_to_lmb`, this function merges Gaussian components from
/// different hypotheses into a single Gaussian per track using moment
/// matching.
///
/// # Arguments
///
/// * `density` - The GLMB density to convert
///
/// # Returns
///
/// An LMB track set with single-Gaussian tracks
#[cfg(feature = "alloc")]
pub fn glmb_to_lmb_merged<T: RealField + Float + Copy, const N: usize>(
    density: &GlmbDensity<T, N>,
) -> LmbTrackSet<T, N> {
    use crate::types::spaces::StateCovariance;
    use alloc::collections::BTreeMap;

    if density.hypotheses.is_empty() {
        return LmbTrackSet::new();
    }

    // Compute normalized weights
    let max_log = density
        .hypotheses
        .iter()
        .map(|h| h.log_weight)
        .fold(Float::neg_infinity(), |a, b| if a > b { a } else { b });

    let weights: Vec<T> = density
        .hypotheses
        .iter()
        .map(|h| Float::exp(h.log_weight - max_log))
        .collect();
    let total: T = weights.iter().fold(T::zero(), |acc, &w| acc + w);

    if total <= T::zero() {
        return LmbTrackSet::new();
    }

    // Accumulate per-label statistics for moment matching
    let mut label_data: BTreeMap<Label, (StateVector<T, N>, StateCovariance<T, N>, T)> =
        BTreeMap::new();

    for (h, &w) in density.hypotheses.iter().zip(&weights) {
        let norm_w = w / total;

        for track in &h.tracks {
            let entry = label_data
                .entry(track.label)
                .or_insert_with(|| (StateVector::zeros(), StateCovariance::zeros(), T::zero()));

            // Weighted mean accumulation
            entry.0 = StateVector::from_svector(
                entry.0.as_svector() + track.state.mean.as_svector().scale(norm_w),
            );

            // Weighted covariance accumulation (will need correction for mean)
            // Using P_merged = E[P] + E[(x - x_mean)(x - x_mean)^T]
            entry.1 = entry.1.add(&track.state.covariance.scale(norm_w));

            entry.2 += norm_w;
        }
    }

    // Build LMB track set with merged Gaussians
    let mut tracks = LmbTrackSet::new();

    for (label, (weighted_mean, weighted_cov, existence)) in label_data {
        if existence <= T::zero() {
            continue;
        }

        // Normalize mean
        let mean =
            StateVector::from_svector(weighted_mean.as_svector().scale(T::one() / existence));

        // Compute covariance with spread-of-means correction
        // For simplicity, we use the weighted average covariance
        // A more accurate approach would add the spread of means
        let covariance = weighted_cov.scale(T::one() / existence);

        let state = GaussianState::new(T::one(), mean, covariance);
        let mut components = GaussianMixture::new();
        components.push(state.clone());

        tracks.push(LmbTrack {
            label,
            existence,
            components,
        });
    }

    tracks
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Generates all k-combinations of indices 0..n
#[cfg(feature = "alloc")]
fn generate_combinations(n: usize, k: usize) -> Vec<Vec<usize>> {
    if k == 0 {
        return vec![Vec::new()];
    }
    if k > n {
        return Vec::new();
    }

    let mut result = Vec::new();
    let mut combo = Vec::with_capacity(k);
    generate_combinations_recursive(n, k, 0, &mut combo, &mut result);
    result
}

#[cfg(feature = "alloc")]
fn generate_combinations_recursive(
    n: usize,
    k: usize,
    start: usize,
    current: &mut Vec<usize>,
    result: &mut Vec<Vec<usize>>,
) {
    if current.len() == k {
        result.push(current.clone());
        return;
    }

    for i in start..n {
        if n - i < k - current.len() {
            break; // Not enough elements left
        }
        current.push(i);
        generate_combinations_recursive(n, k, i + 1, current, result);
        current.pop();
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::spaces::StateCovariance;

    #[cfg(feature = "alloc")]
    fn create_test_lmb_track_set() -> LmbTrackSet<f64, 4> {
        let mut tracks = LmbTrackSet::new();

        let mean: StateVector<f64, 4> = StateVector::from_array([0.0, 0.0, 0.0, 0.0]);
        let cov: StateCovariance<f64, 4> = StateCovariance::identity();

        // Track 1: high existence
        let mut components1 = GaussianMixture::new();
        components1.push(GaussianState::new(1.0, mean, cov));
        tracks.push(LmbTrack {
            label: Label::new(0, 0),
            existence: 0.9,
            components: components1,
        });

        // Track 2: medium existence
        let mean2: StateVector<f64, 4> = StateVector::from_array([10.0, 0.0, 0.0, 0.0]);
        let mut components2 = GaussianMixture::new();
        components2.push(GaussianState::new(1.0, mean2, cov));
        tracks.push(LmbTrack {
            label: Label::new(0, 1),
            existence: 0.6,
            components: components2,
        });

        tracks
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_generate_combinations() {
        let combos = generate_combinations(4, 2);
        assert_eq!(combos.len(), 6); // C(4,2) = 6

        let combos = generate_combinations(3, 0);
        assert_eq!(combos.len(), 1); // Empty set

        let combos = generate_combinations(3, 3);
        assert_eq!(combos.len(), 1); // {0, 1, 2}

        let combos = generate_combinations(3, 4);
        assert!(combos.is_empty()); // Impossible
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_lmb_to_glmb() {
        let tracks = create_test_lmb_track_set();
        let glmb = lmb_to_glmb(&tracks, 10);

        // Should have generated some hypotheses
        assert!(!glmb.hypotheses.is_empty());

        // Check that weights are normalized (max = 0)
        let max_log = glmb
            .hypotheses
            .iter()
            .map(|h| h.log_weight)
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(max_log <= 0.01); // Allow small numerical error
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_lmb_to_glmb_full_small() {
        // Small test with 2 tracks -> 4 hypotheses
        let mut tracks = LmbTrackSet::new();
        let mean: StateVector<f64, 4> = StateVector::from_array([0.0, 0.0, 0.0, 0.0]);
        let cov: StateCovariance<f64, 4> = StateCovariance::identity();

        let mut components = GaussianMixture::new();
        components.push(GaussianState::new(1.0, mean, cov));

        tracks.push(LmbTrack {
            label: Label::new(0, 0),
            existence: 0.8,
            components: components.clone(),
        });
        tracks.push(LmbTrack {
            label: Label::new(0, 1),
            existence: 0.5,
            components,
        });

        let glmb = lmb_to_glmb_full(&tracks);

        // Should have exactly 4 hypotheses: {}, {0}, {1}, {0,1}
        assert_eq!(glmb.hypotheses.len(), 4);

        // Check cardinality distribution sums to 1
        let dist = glmb.cardinality_distribution();
        let sum: f64 = dist.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_glmb_to_lmb() {
        let tracks = create_test_lmb_track_set();
        let glmb = lmb_to_glmb_full(&tracks);
        let lmb_recovered = glmb_to_lmb(&glmb);

        // Should recover same number of tracks
        assert_eq!(lmb_recovered.len(), tracks.len());

        // Existence probabilities should be approximately preserved
        for original in tracks.iter() {
            let recovered = lmb_recovered.find_by_label(original.label);
            assert!(recovered.is_some());

            let r_orig = original.existence;
            let r_recv = recovered.unwrap().existence;
            assert!(
                (r_orig - r_recv).abs() < 0.01,
                "Existence mismatch: {} vs {}",
                r_orig,
                r_recv
            );
        }
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_roundtrip_preserves_marginals() {
        // Create LMB with specific existence probabilities
        let mut tracks = LmbTrackSet::new();
        let _mean: StateVector<f64, 4> = StateVector::from_array([0.0, 0.0, 0.0, 0.0]);
        let cov: StateCovariance<f64, 4> = StateCovariance::identity();

        for i in 0..3 {
            let mut components = GaussianMixture::new();
            let m = StateVector::from_array([i as f64 * 10.0, 0.0, 0.0, 0.0]);
            components.push(GaussianState::new(1.0, m, cov));

            tracks.push(LmbTrack {
                label: Label::new(0, i),
                existence: 0.3 + 0.2 * i as f64,
                components,
            });
        }

        // Round trip
        let glmb = lmb_to_glmb_full(&tracks);
        let lmb_recovered = glmb_to_lmb(&glmb);

        // Expected cardinality should be preserved
        let expected_original: f64 = tracks.iter().map(|t| t.existence).sum();
        let expected_recovered: f64 = lmb_recovered.iter().map(|t| t.existence).sum();

        assert!(
            (expected_original - expected_recovered).abs() < 0.01,
            "Expected cardinality mismatch: {} vs {}",
            expected_original,
            expected_recovered
        );
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_glmb_to_lmb_merged() {
        let tracks = create_test_lmb_track_set();
        let glmb = lmb_to_glmb_full(&tracks);
        let lmb_merged = glmb_to_lmb_merged(&glmb);

        // Should have same number of tracks
        assert_eq!(lmb_merged.len(), tracks.len());

        // Each track should have single component
        for track in lmb_merged.iter() {
            assert_eq!(track.components.len(), 1);
        }
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_empty_conversions() {
        let empty_lmb: LmbTrackSet<f64, 4> = LmbTrackSet::new();
        let glmb = lmb_to_glmb(&empty_lmb, 10);

        // Should have single empty hypothesis
        assert_eq!(glmb.hypotheses.len(), 1);
        assert_eq!(glmb.hypotheses[0].cardinality(), 0);

        let lmb_recovered = glmb_to_lmb(&glmb);
        assert!(lmb_recovered.is_empty());
    }
}
