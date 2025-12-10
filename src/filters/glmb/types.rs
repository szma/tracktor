//! GLMB-specific types
//!
//! Core data structures for delta-GLMB filter implementation.
//!
//! The GLMB (Generalized Labeled Multi-Bernoulli) density represents a
//! multi-target distribution as a mixture of hypotheses, where each hypothesis
//! specifies which tracks exist and their states.

use core::marker::PhantomData;
use nalgebra::RealField;
use num_traits::Float;

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use crate::types::gaussian::GaussianState;
use crate::types::labels::Label;
use crate::types::spaces::{StateCovariance, StateVector};

// ============================================================================
// GLMB Track
// ============================================================================

/// A track in a GLMB hypothesis.
///
/// Unlike [`BernoulliTrack`](crate::types::labels::BernoulliTrack), this does
/// NOT have an explicit existence probability because existence is implicit:
/// if a track is in the hypothesis, it exists with probability 1 (conditioned
/// on the hypothesis).
///
/// # Type Parameters
///
/// - `T`: Scalar type (e.g., `f64`)
/// - `N`: State dimension
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq)]
pub struct GlmbTrack<T: RealField, const N: usize> {
    /// Unique label identifying this track
    pub label: Label,
    /// Gaussian state estimate
    pub state: GaussianState<T, N>,
}

#[cfg(feature = "alloc")]
impl<T: RealField + Copy, const N: usize> GlmbTrack<T, N> {
    /// Creates a new GLMB track.
    #[inline]
    pub fn new(label: Label, state: GaussianState<T, N>) -> Self {
        Self { label, state }
    }

    /// Creates a GLMB track with unit weight.
    #[inline]
    pub fn with_state(
        label: Label,
        mean: StateVector<T, N>,
        covariance: StateCovariance<T, N>,
    ) -> Self {
        Self {
            label,
            state: GaussianState::new(T::one(), mean, covariance),
        }
    }

    /// Creates from a BernoulliTrack (drops existence probability).
    pub fn from_bernoulli(track: &crate::types::labels::BernoulliTrack<T, N>) -> Self {
        Self {
            label: track.label,
            state: track.state.clone(),
        }
    }
}

// ============================================================================
// GLMB Hypothesis
// ============================================================================

/// A single hypothesis in the GLMB filter.
///
/// Each hypothesis represents a specific:
/// - Label set I (which tracks exist) - encoded by `tracks` membership
/// - Association history ξ - tracked in `current_association`
/// - Joint weight w^(I,ξ) - stored as `log_weight`
///
/// In delta-GLMB, each hypothesis has a unique (I, ξ) pair.
///
/// # Type Parameters
///
/// - `T`: Scalar type (e.g., `f64`)
/// - `N`: State dimension
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct GlmbHypothesis<T: RealField, const N: usize> {
    /// Log weight for numerical stability (log of hypothesis probability)
    pub log_weight: T,
    /// Tracks in this hypothesis (all exist by definition)
    pub tracks: Vec<GlmbTrack<T, N>>,
    /// Current timestep association: tracks[i] -> Some(meas_idx) or None (miss)
    /// Length matches tracks.len()
    pub current_association: Vec<Option<usize>>,
}

#[cfg(feature = "alloc")]
impl<T: RealField + Float + Copy, const N: usize> GlmbHypothesis<T, N> {
    /// Creates a new hypothesis.
    pub fn new(
        log_weight: T,
        tracks: Vec<GlmbTrack<T, N>>,
        current_association: Vec<Option<usize>>,
    ) -> Self {
        debug_assert_eq!(tracks.len(), current_association.len());
        Self {
            log_weight,
            tracks,
            current_association,
        }
    }

    /// Creates an empty hypothesis (no tracks).
    #[inline]
    pub fn empty(log_weight: T) -> Self {
        Self {
            log_weight,
            tracks: Vec::new(),
            current_association: Vec::new(),
        }
    }

    /// Returns the cardinality (number of tracks).
    #[inline]
    pub fn cardinality(&self) -> usize {
        self.tracks.len()
    }

    /// Returns the weight (exponentiated log weight).
    #[inline]
    pub fn weight(&self) -> T {
        Float::exp(self.log_weight)
    }

    /// Returns the label set for this hypothesis.
    pub fn label_set(&self) -> Vec<Label> {
        self.tracks.iter().map(|t| t.label).collect()
    }

    /// Finds a track by label.
    pub fn find_track(&self, label: Label) -> Option<&GlmbTrack<T, N>> {
        self.tracks.iter().find(|t| t.label == label)
    }

    /// Finds a track by label (mutable).
    pub fn find_track_mut(&mut self, label: Label) -> Option<&mut GlmbTrack<T, N>> {
        self.tracks.iter_mut().find(|t| t.label == label)
    }

    /// Returns true if this hypothesis contains a track with the given label.
    pub fn contains_label(&self, label: Label) -> bool {
        self.tracks.iter().any(|t| t.label == label)
    }
}

// ============================================================================
// GLMB Density
// ============================================================================

/// A GLMB density represented as a mixture of hypotheses.
///
/// The density is:
/// ```text
/// π(X) = Σ_c w^(c) · δ_{I^(c)}(L(X)) · [p^(c)]^X
/// ```
/// where c indexes hypotheses, w^(c) is the hypothesis weight, I^(c) is the
/// label set, and p^(c) is the spatial distribution for tracks in hypothesis c.
///
/// # Type Parameters
///
/// - `T`: Scalar type (e.g., `f64`)
/// - `N`: State dimension
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct GlmbDensity<T: RealField, const N: usize> {
    /// Collection of hypotheses
    pub hypotheses: Vec<GlmbHypothesis<T, N>>,
}

#[cfg(feature = "alloc")]
impl<T: RealField + Float + Copy, const N: usize> GlmbDensity<T, N> {
    /// Creates an empty GLMB density.
    #[inline]
    pub fn new() -> Self {
        Self {
            hypotheses: Vec::new(),
        }
    }

    /// Creates a GLMB density with a single empty hypothesis.
    #[inline]
    pub fn with_empty_hypothesis() -> Self {
        Self {
            hypotheses: vec![GlmbHypothesis::empty(T::zero())],
        }
    }

    /// Creates a GLMB density with the given capacity for hypotheses.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            hypotheses: Vec::with_capacity(capacity),
        }
    }

    /// Returns the number of hypotheses.
    #[inline]
    pub fn num_hypotheses(&self) -> usize {
        self.hypotheses.len()
    }

    /// Returns true if the density has no hypotheses.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.hypotheses.is_empty()
    }

    /// Adds a hypothesis to the density.
    #[inline]
    pub fn push(&mut self, hypothesis: GlmbHypothesis<T, N>) {
        self.hypotheses.push(hypothesis);
    }

    /// Normalizes log weights so the maximum is 0.
    ///
    /// This prevents numerical underflow in the weights while
    /// preserving their relative values.
    pub fn normalize_log_weights(&mut self) {
        if self.hypotheses.is_empty() {
            return;
        }

        let max_log = self
            .hypotheses
            .iter()
            .map(|h| h.log_weight)
            .fold(Float::neg_infinity(), |a, b| if a > b { a } else { b });

        if max_log.is_finite() {
            for h in &mut self.hypotheses {
                h.log_weight = h.log_weight - max_log;
            }
        }
    }

    /// Prunes hypotheses with log weight below the threshold.
    pub fn prune_by_weight(&mut self, log_threshold: T) {
        self.hypotheses.retain(|h| h.log_weight >= log_threshold);
    }

    /// Keeps top k hypotheses by weight.
    pub fn keep_top_k(&mut self, k: usize) {
        if self.hypotheses.len() <= k {
            return;
        }
        self.hypotheses
            .sort_by(|a, b| b.log_weight.partial_cmp(&a.log_weight).unwrap());
        self.hypotheses.truncate(k);
    }

    /// Groups hypotheses by cardinality.
    ///
    /// Returns a vector of (cardinality, hypotheses) pairs, sorted by cardinality.
    pub fn group_by_cardinality(&self) -> Vec<(usize, Vec<&GlmbHypothesis<T, N>>)> {
        use alloc::collections::BTreeMap;

        let mut groups: BTreeMap<usize, Vec<&GlmbHypothesis<T, N>>> = BTreeMap::new();
        for h in &self.hypotheses {
            groups.entry(h.cardinality()).or_default().push(h);
        }
        groups.into_iter().collect()
    }

    /// Keeps top k hypotheses per cardinality value.
    ///
    /// This preserves the cardinality distribution while limiting the total
    /// number of hypotheses.
    pub fn keep_top_k_per_cardinality(&mut self, k: usize) {
        use alloc::collections::BTreeMap;

        let mut groups: BTreeMap<usize, Vec<GlmbHypothesis<T, N>>> = BTreeMap::new();
        for h in self.hypotheses.drain(..) {
            groups.entry(h.cardinality()).or_default().push(h);
        }

        for group in groups.values_mut() {
            group.sort_by(|a, b| b.log_weight.partial_cmp(&a.log_weight).unwrap());
            group.truncate(k);
        }

        self.hypotheses = groups.into_values().flatten().collect();
    }

    /// Computes the cardinality distribution.
    ///
    /// Returns a vector where index n contains P(exactly n targets exist).
    pub fn cardinality_distribution(&self) -> Vec<T> {
        if self.hypotheses.is_empty() {
            return vec![T::one()];
        }

        // Normalize weights
        let max_log = self
            .hypotheses
            .iter()
            .map(|h| h.log_weight)
            .fold(Float::neg_infinity(), |a, b| if a > b { a } else { b });

        let weights: Vec<T> = self
            .hypotheses
            .iter()
            .map(|h| Float::exp(h.log_weight - max_log))
            .collect();
        let total: T = weights.iter().fold(T::zero(), |acc, &w| acc + w);

        if total <= T::zero() {
            return vec![T::one()];
        }

        // Find max cardinality
        let max_card = self
            .hypotheses
            .iter()
            .map(|h| h.cardinality())
            .max()
            .unwrap_or(0);

        // Accumulate
        let mut dist = vec![T::zero(); max_card + 1];
        for (h, &w) in self.hypotheses.iter().zip(&weights) {
            dist[h.cardinality()] = dist[h.cardinality()] + w / total;
        }

        dist
    }

    /// Returns the MAP (most likely) cardinality.
    pub fn map_cardinality(&self) -> usize {
        let dist = self.cardinality_distribution();
        dist.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Computes marginal existence probability for each label.
    ///
    /// Returns a vector of (label, existence_probability) pairs.
    pub fn marginal_existence(&self) -> Vec<(Label, T)> {
        use alloc::collections::BTreeMap;

        if self.hypotheses.is_empty() {
            return Vec::new();
        }

        // Normalize weights
        let max_log = self
            .hypotheses
            .iter()
            .map(|h| h.log_weight)
            .fold(Float::neg_infinity(), |a, b| if a > b { a } else { b });

        let weights: Vec<T> = self
            .hypotheses
            .iter()
            .map(|h| Float::exp(h.log_weight - max_log))
            .collect();
        let total: T = weights.iter().fold(T::zero(), |acc, &w| acc + w);

        if total <= T::zero() {
            return Vec::new();
        }

        let mut existence: BTreeMap<Label, T> = BTreeMap::new();
        for (h, &w) in self.hypotheses.iter().zip(&weights) {
            let norm_w = w / total;
            for track in &h.tracks {
                *existence.entry(track.label).or_insert(T::zero()) += norm_w;
            }
        }

        existence.into_iter().collect()
    }

    /// Computes marginal state for each label (weighted average).
    ///
    /// Returns a vector of (label, mean_state, existence_probability) tuples.
    pub fn marginal_states(&self) -> Vec<(Label, StateVector<T, N>, T)> {
        use alloc::collections::BTreeMap;

        if self.hypotheses.is_empty() {
            return Vec::new();
        }

        // Normalize weights
        let max_log = self
            .hypotheses
            .iter()
            .map(|h| h.log_weight)
            .fold(Float::neg_infinity(), |a, b| if a > b { a } else { b });

        let weights: Vec<T> = self
            .hypotheses
            .iter()
            .map(|h| Float::exp(h.log_weight - max_log))
            .collect();
        let total: T = weights.iter().fold(T::zero(), |acc, &w| acc + w);

        if total <= T::zero() {
            return Vec::new();
        }

        let mut accum: BTreeMap<Label, (StateVector<T, N>, T)> = BTreeMap::new();

        for (h, &w) in self.hypotheses.iter().zip(&weights) {
            let norm_w = w / total;
            for track in &h.tracks {
                let entry = accum
                    .entry(track.label)
                    .or_insert((StateVector::zeros(), T::zero()));
                entry.0 = StateVector::from_svector(
                    entry.0.as_svector() + track.state.mean.as_svector().scale(norm_w),
                );
                entry.1 = entry.1 + norm_w;
            }
        }

        accum
            .into_iter()
            .map(|(label, (state_sum, existence))| {
                let normalized =
                    StateVector::from_svector(state_sum.as_svector().scale(T::one() / existence));
                (label, normalized, existence)
            })
            .collect()
    }

    /// Returns the best (highest weight) hypothesis.
    pub fn best_hypothesis(&self) -> Option<&GlmbHypothesis<T, N>> {
        self.hypotheses
            .iter()
            .max_by(|a, b| a.log_weight.partial_cmp(&b.log_weight).unwrap())
    }

    /// Clears all hypotheses.
    #[inline]
    pub fn clear(&mut self) {
        self.hypotheses.clear();
    }
}

#[cfg(feature = "alloc")]
impl<T: RealField + Float + Copy, const N: usize> Default for GlmbDensity<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Truncation Configuration
// ============================================================================

/// Configuration for GLMB hypothesis truncation.
///
/// Truncation is essential for computational tractability, as the number of
/// hypotheses can grow exponentially with measurements.
#[derive(Debug, Clone)]
pub struct GlmbTruncationConfig<T: RealField> {
    /// Log weight threshold for pruning (hypotheses below this are removed)
    pub log_weight_threshold: T,
    /// Maximum total number of hypotheses
    pub max_hypotheses: usize,
    /// Maximum hypotheses per cardinality value (None = no limit)
    pub max_per_cardinality: Option<usize>,
}

impl<T: RealField + Float> GlmbTruncationConfig<T> {
    /// Creates the default truncation configuration.
    ///
    /// Default values:
    /// - `log_weight_threshold`: -20.0
    /// - `max_hypotheses`: 1000
    /// - `max_per_cardinality`: Some(100)
    pub fn default_config() -> Self {
        Self {
            log_weight_threshold: T::from_f64(-20.0).unwrap(),
            max_hypotheses: 1000,
            max_per_cardinality: Some(100),
        }
    }

    /// Creates a custom truncation configuration.
    pub fn new(
        log_weight_threshold: T,
        max_hypotheses: usize,
        max_per_cardinality: Option<usize>,
    ) -> Self {
        Self {
            log_weight_threshold,
            max_hypotheses,
            max_per_cardinality,
        }
    }

    /// Creates a lightweight configuration for testing or low-latency applications.
    pub fn lightweight() -> Self {
        Self {
            log_weight_threshold: T::from_f64(-10.0).unwrap(),
            max_hypotheses: 100,
            max_per_cardinality: Some(20),
        }
    }

    /// Creates a high-fidelity configuration for maximum accuracy.
    pub fn high_fidelity() -> Self {
        Self {
            log_weight_threshold: T::from_f64(-30.0).unwrap(),
            max_hypotheses: 5000,
            max_per_cardinality: Some(500),
        }
    }
}

impl<T: RealField + Float> Default for GlmbTruncationConfig<T> {
    fn default() -> Self {
        Self::default_config()
    }
}

// ============================================================================
// Fixed-Size GLMB Track (no_std support)
// ============================================================================

/// Fixed-capacity GLMB track for no_std environments without allocation.
///
/// # Type Parameters
///
/// - `T`: Scalar type
/// - `N`: State dimension
pub struct FixedGlmbTrack<T: RealField, const N: usize> {
    /// Unique label identifying this track
    pub label: Label,
    /// Gaussian state estimate
    pub state: GaussianState<T, N>,
    _marker: PhantomData<T>,
}

impl<T: RealField + Copy, const N: usize> FixedGlmbTrack<T, N> {
    /// Creates a new fixed GLMB track.
    #[inline]
    pub fn new(label: Label, state: GaussianState<T, N>) -> Self {
        Self {
            label,
            state,
            _marker: PhantomData,
        }
    }
}

impl<T: RealField + Copy, const N: usize> Clone for FixedGlmbTrack<T, N> {
    fn clone(&self) -> Self {
        Self {
            label: self.label,
            state: self.state.clone(),
            _marker: PhantomData,
        }
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
    #[test]
    fn test_glmb_track_creation() {
        let label = Label::new(0, 0);
        let mean: StateVector<f64, 4> = StateVector::from_array([1.0, 2.0, 0.5, 0.5]);
        let cov: StateCovariance<f64, 4> = StateCovariance::identity();
        let state = GaussianState::new(1.0, mean, cov);

        let track = GlmbTrack::new(label, state);

        assert_eq!(track.label, Label::new(0, 0));
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_glmb_hypothesis_creation() {
        let label = Label::new(0, 0);
        let mean: StateVector<f64, 4> = StateVector::from_array([1.0, 2.0, 0.0, 0.0]);
        let cov: StateCovariance<f64, 4> = StateCovariance::identity();
        let state = GaussianState::new(1.0, mean, cov);
        let track = GlmbTrack::new(label, state);

        let hypothesis = GlmbHypothesis::new(0.0, vec![track], vec![Some(0)]);

        assert_eq!(hypothesis.cardinality(), 1);
        assert!((hypothesis.weight() - 1.0).abs() < 1e-10);
        assert!(hypothesis.contains_label(Label::new(0, 0)));
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_glmb_hypothesis_empty() {
        let hypothesis: GlmbHypothesis<f64, 4> = GlmbHypothesis::empty(-1.0);

        assert_eq!(hypothesis.cardinality(), 0);
        assert!(hypothesis.tracks.is_empty());
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_glmb_density_normalization() {
        let mut density = GlmbDensity::<f64, 4>::new();
        density.push(GlmbHypothesis::empty(-100.0));
        density.push(GlmbHypothesis::empty(-99.0));

        density.normalize_log_weights();

        let max_log = density
            .hypotheses
            .iter()
            .map(|h| h.log_weight)
            .fold(f64::NEG_INFINITY, f64::max);
        assert!((max_log - 0.0).abs() < 1e-10);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_glmb_density_cardinality_distribution() {
        let mut density = GlmbDensity::<f64, 4>::new();

        // Hypothesis with 0 tracks (weight 0.3)
        let h0 = GlmbHypothesis::empty((0.3_f64).ln());

        // Hypothesis with 1 track (weight 0.7)
        let label = Label::new(0, 0);
        let mean: StateVector<f64, 4> = StateVector::from_array([0.0, 0.0, 0.0, 0.0]);
        let cov: StateCovariance<f64, 4> = StateCovariance::identity();
        let track = GlmbTrack::new(label, GaussianState::new(1.0, mean, cov));
        let h1 = GlmbHypothesis::new((0.7_f64).ln(), vec![track], vec![None]);

        density.push(h0);
        density.push(h1);

        let dist = density.cardinality_distribution();

        assert_eq!(dist.len(), 2);
        // Distribution should sum to 1
        let sum: f64 = dist.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_glmb_density_marginal_existence() {
        let mut density = GlmbDensity::<f64, 4>::new();

        let label0 = Label::new(0, 0);
        let label1 = Label::new(0, 1);
        let mean: StateVector<f64, 4> = StateVector::from_array([0.0, 0.0, 0.0, 0.0]);
        let cov: StateCovariance<f64, 4> = StateCovariance::identity();

        // Hypothesis 1: only label0 (weight 0.4)
        let track0 = GlmbTrack::new(label0, GaussianState::new(1.0, mean, cov));
        let h1 = GlmbHypothesis::new(0.4_f64.ln(), vec![track0], vec![None]);

        // Hypothesis 2: both labels (weight 0.6)
        let track0 = GlmbTrack::new(label0, GaussianState::new(1.0, mean, cov));
        let track1 = GlmbTrack::new(label1, GaussianState::new(1.0, mean, cov));
        let h2 = GlmbHypothesis::new(0.6_f64.ln(), vec![track0, track1], vec![None, None]);

        density.push(h1);
        density.push(h2);

        let marginals = density.marginal_existence();

        // label0 exists in both hypotheses: 0.4 + 0.6 = 1.0
        // label1 exists only in h2: 0.6
        let label0_existence = marginals
            .iter()
            .find(|(l, _)| *l == label0)
            .map(|(_, r)| *r);
        let label1_existence = marginals
            .iter()
            .find(|(l, _)| *l == label1)
            .map(|(_, r)| *r);

        assert!((label0_existence.unwrap() - 1.0).abs() < 1e-10);
        assert!((label1_existence.unwrap() - 0.6).abs() < 1e-10);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_truncation_config() {
        let config: GlmbTruncationConfig<f64> = GlmbTruncationConfig::default_config();
        assert_eq!(config.max_hypotheses, 1000);
        assert_eq!(config.max_per_cardinality, Some(100));

        let lightweight = GlmbTruncationConfig::<f64>::lightweight();
        assert!(lightweight.max_hypotheses < config.max_hypotheses);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_keep_top_k() {
        let mut density = GlmbDensity::<f64, 4>::new();

        for i in 0..10 {
            density.push(GlmbHypothesis::empty(-(i as f64)));
        }

        density.keep_top_k(3);

        assert_eq!(density.num_hypotheses(), 3);
        // Best hypotheses should be kept (log_weight 0, -1, -2)
        assert!(density.hypotheses[0].log_weight >= -0.01);
    }
}
