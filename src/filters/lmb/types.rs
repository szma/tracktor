//! LMB-specific types
//!
//! Core data structures for LMB and LMBM filters, extending tracktor's
//! type-safe framework with labeled Bernoulli components.

use core::marker::PhantomData;
use nalgebra::RealField;
use num_traits::Float;

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use crate::types::gaussian::{GaussianMixture, GaussianState};
use crate::types::labels::{BernoulliTrack, Label};
use crate::types::spaces::{StateCovariance, StateVector};

// ============================================================================
// LMB Track (with Gaussian Mixture)
// ============================================================================

/// A labeled Bernoulli track with Gaussian mixture representation.
///
/// Unlike [`BernoulliTrack`] which has a single Gaussian component, `LmbTrack`
/// maintains a mixture of Gaussians to represent association uncertainty.
/// The mixture weights sum to 1 (conditioned on existence).
///
/// # Type Parameters
///
/// - `T`: Scalar type (e.g., `f64`)
/// - `N`: State dimension
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct LmbTrack<T: RealField, const N: usize> {
    /// Unique label identifying this track
    pub label: Label,
    /// Existence probability r in [0, 1]
    pub existence: T,
    /// Gaussian mixture components (weights sum to 1 conditioned on existence)
    pub components: GaussianMixture<T, N>,
}

#[cfg(feature = "alloc")]
impl<T: RealField + Copy, const N: usize> LmbTrack<T, N> {
    /// Creates a new LMB track with a single component.
    #[inline]
    pub fn new(label: Label, existence: T, state: GaussianState<T, N>) -> Self {
        let mut components = GaussianMixture::with_capacity(1);
        components.push(state);
        Self {
            label,
            existence,
            components,
        }
    }

    /// Creates a new LMB track with multiple components.
    #[inline]
    pub fn with_components(label: Label, existence: T, components: GaussianMixture<T, N>) -> Self {
        Self {
            label,
            existence,
            components,
        }
    }

    /// Creates an LMB track from a simple Bernoulli track.
    #[inline]
    pub fn from_bernoulli(track: BernoulliTrack<T, N>) -> Self {
        Self::new(track.label, track.existence, track.state)
    }

    /// Returns true if this track is likely to exist (existence > 0.5).
    #[inline]
    pub fn is_likely_existing(&self) -> bool {
        self.existence > T::from_f64(0.5).unwrap()
    }

    /// Returns the number of Gaussian components.
    #[inline]
    pub fn num_components(&self) -> usize {
        self.components.len()
    }

    /// Returns the mean of the highest-weight component.
    pub fn best_estimate(&self) -> Option<&GaussianState<T, N>> {
        self.components
            .iter()
            .max_by(|a, b| a.weight.partial_cmp(&b.weight).unwrap())
    }

    /// Returns the weighted mean state across all components.
    pub fn weighted_mean(&self) -> StateVector<T, N> {
        let total_weight = self.components.total_weight();
        if total_weight <= T::zero() {
            return self.components.components[0].mean;
        }

        let mut mean = StateVector::zeros();
        for component in self.components.iter() {
            let scaled = component
                .mean
                .as_svector()
                .scale(component.weight / total_weight);
            mean = StateVector::from_svector(mean.as_svector() + scaled);
        }
        mean
    }

    /// Normalizes component weights to sum to 1.
    pub fn normalize_weights(&mut self) {
        let total = self.components.total_weight();
        if total > T::zero() {
            for component in self.components.iter_mut() {
                component.weight /= total;
            }
        }
    }

    /// Converts to a simple Bernoulli track using the best estimate.
    pub fn to_bernoulli(&self) -> Option<BernoulliTrack<T, N>> {
        self.best_estimate().map(|state| BernoulliTrack {
            label: self.label,
            existence: self.existence,
            state: state.clone(),
        })
    }
}

// ============================================================================
// LMB Track Set
// ============================================================================

/// A collection of LMB tracks representing the multi-target state.
///
/// The LMB density is a union of independent Bernoulli components,
/// each with its own existence probability and spatial distribution.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, Default)]
pub struct LmbTrackSet<T: RealField, const N: usize> {
    /// The tracks in this set
    pub tracks: Vec<LmbTrack<T, N>>,
}

#[cfg(feature = "alloc")]
impl<T: RealField + Copy, const N: usize> LmbTrackSet<T, N> {
    /// Creates an empty track set.
    #[inline]
    pub fn new() -> Self {
        Self { tracks: Vec::new() }
    }

    /// Creates a track set with the given capacity.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            tracks: Vec::with_capacity(capacity),
        }
    }

    /// Returns the number of tracks.
    #[inline]
    pub fn len(&self) -> usize {
        self.tracks.len()
    }

    /// Returns true if the track set is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.tracks.is_empty()
    }

    /// Adds a track to the set.
    #[inline]
    pub fn push(&mut self, track: LmbTrack<T, N>) {
        self.tracks.push(track);
    }

    /// Iterates over the tracks.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &LmbTrack<T, N>> {
        self.tracks.iter()
    }

    /// Mutably iterates over the tracks.
    #[inline]
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut LmbTrack<T, N>> {
        self.tracks.iter_mut()
    }

    /// Returns the expected number of targets (sum of existence probabilities).
    pub fn expected_cardinality(&self) -> T {
        self.tracks
            .iter()
            .fold(T::zero(), |acc, t| acc + t.existence)
    }

    /// Returns tracks with existence probability above the threshold.
    pub fn filter_by_existence(&self, threshold: T) -> Vec<&LmbTrack<T, N>> {
        self.tracks
            .iter()
            .filter(|t| t.existence >= threshold)
            .collect()
    }

    /// Removes tracks with existence probability below the threshold.
    pub fn prune_by_existence(&mut self, threshold: T) {
        self.tracks.retain(|t| t.existence >= threshold);
    }

    /// Finds a track by its label.
    pub fn find_by_label(&self, label: Label) -> Option<&LmbTrack<T, N>> {
        self.tracks.iter().find(|t| t.label == label)
    }

    /// Finds a track by its label (mutable).
    pub fn find_by_label_mut(&mut self, label: Label) -> Option<&mut LmbTrack<T, N>> {
        self.tracks.iter_mut().find(|t| t.label == label)
    }

    /// Returns existence probabilities for all tracks.
    pub fn existence_probabilities(&self) -> Vec<T> {
        self.tracks.iter().map(|t| t.existence).collect()
    }

    /// Clears all tracks.
    #[inline]
    pub fn clear(&mut self) {
        self.tracks.clear();
    }
}

// ============================================================================
// Fixed-Size LMB Track (no_std)
// ============================================================================

/// Fixed-capacity LMB track for no_std environments.
///
/// Uses a fixed-size array for Gaussian components instead of dynamic allocation.
///
/// # Type Parameters
///
/// - `T`: Scalar type
/// - `N`: State dimension
/// - `MAX_COMPONENTS`: Maximum number of Gaussian components
pub struct FixedLmbTrack<T: RealField, const N: usize, const MAX_COMPONENTS: usize> {
    /// Unique label identifying this track
    pub label: Label,
    /// Existence probability r in [0, 1]
    pub existence: T,
    /// Storage for Gaussian components
    components: [core::mem::MaybeUninit<GaussianState<T, N>>; MAX_COMPONENTS],
    /// Number of valid components
    num_components: usize,
    _marker: PhantomData<T>,
}

impl<T: RealField + Copy, const N: usize, const MAX_COMPONENTS: usize>
    FixedLmbTrack<T, N, MAX_COMPONENTS>
{
    /// Creates a new fixed-size LMB track with a single component.
    pub fn new(label: Label, existence: T, state: GaussianState<T, N>) -> Self {
        let mut track = Self {
            label,
            existence,
            components: unsafe { core::mem::MaybeUninit::uninit().assume_init() },
            num_components: 0,
            _marker: PhantomData,
        };
        track.components[0].write(state);
        track.num_components = 1;
        track
    }

    /// Returns the number of components.
    #[inline]
    pub fn num_components(&self) -> usize {
        self.num_components
    }

    /// Returns the maximum capacity.
    #[inline]
    pub fn capacity(&self) -> usize {
        MAX_COMPONENTS
    }

    /// Attempts to add a component. Returns Err if at capacity.
    pub fn try_push(&mut self, component: GaussianState<T, N>) -> Result<(), crate::TracktorError> {
        if self.num_components >= MAX_COMPONENTS {
            return Err(crate::TracktorError::MaxComponentsExceeded);
        }
        self.components[self.num_components].write(component);
        self.num_components += 1;
        Ok(())
    }

    /// Iterates over the components.
    pub fn iter(&self) -> impl Iterator<Item = &GaussianState<T, N>> {
        self.components[..self.num_components]
            .iter()
            .map(|c| unsafe { c.assume_init_ref() })
    }

    /// Returns a slice of the components.
    pub fn as_slice(&self) -> &[GaussianState<T, N>] {
        unsafe {
            core::slice::from_raw_parts(
                self.components.as_ptr() as *const GaussianState<T, N>,
                self.num_components,
            )
        }
    }

    /// Clears all components.
    pub fn clear(&mut self) {
        for i in 0..self.num_components {
            unsafe { self.components[i].assume_init_drop() };
        }
        self.num_components = 0;
    }
}

impl<T: RealField, const N: usize, const MAX_COMPONENTS: usize> Drop
    for FixedLmbTrack<T, N, MAX_COMPONENTS>
{
    fn drop(&mut self) {
        for i in 0..self.num_components {
            unsafe { self.components[i].assume_init_drop() };
        }
    }
}

impl<T: RealField + Copy, const N: usize, const MAX_COMPONENTS: usize> Clone
    for FixedLmbTrack<T, N, MAX_COMPONENTS>
{
    fn clone(&self) -> Self {
        let mut new = Self {
            label: self.label,
            existence: self.existence,
            components: unsafe { core::mem::MaybeUninit::uninit().assume_init() },
            num_components: 0,
            _marker: PhantomData,
        };
        for component in self.iter() {
            new.try_push(component.clone())
                .expect("Clone into same-capacity FixedLmbTrack should never fail");
        }
        new
    }
}

// ============================================================================
// LMBM Types (Multi-Hypothesis)
// ============================================================================

/// A single hypothesis in the LMBM filter.
///
/// Each hypothesis represents a possible data association event and
/// contains single-component tracks (hard assignments).
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct LmbmHypothesis<T: RealField, const N: usize> {
    /// Log weight for numerical stability (log of hypothesis probability)
    pub log_weight: T,
    /// Tracks in this hypothesis (single Gaussian per track)
    pub tracks: Vec<BernoulliTrack<T, N>>,
}

#[cfg(feature = "alloc")]
impl<T: RealField + Float + Copy, const N: usize> LmbmHypothesis<T, N> {
    /// Creates a new hypothesis with the given log weight and tracks.
    #[inline]
    pub fn new(log_weight: T, tracks: Vec<BernoulliTrack<T, N>>) -> Self {
        Self { log_weight, tracks }
    }

    /// Creates an empty hypothesis (no tracks) with the given log weight.
    #[inline]
    pub fn empty(log_weight: T) -> Self {
        Self {
            log_weight,
            tracks: Vec::new(),
        }
    }

    /// Returns the number of tracks in this hypothesis.
    #[inline]
    pub fn num_tracks(&self) -> usize {
        self.tracks.len()
    }

    /// Returns the weight (exponentiated log weight).
    #[inline]
    pub fn weight(&self) -> T {
        Float::exp(self.log_weight)
    }

    /// Returns the expected cardinality for this hypothesis.
    pub fn expected_cardinality(&self) -> T {
        self.tracks
            .iter()
            .fold(T::zero(), |acc, t| acc + t.existence)
    }
}

/// LMBM filter state - a mixture of hypotheses.
///
/// The LMBM filter maintains multiple hypotheses, each representing
/// a different data association history. Hypothesis weights are
/// maintained in log space for numerical stability.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct LmbmState<T: RealField, const N: usize> {
    /// Collection of hypotheses
    pub hypotheses: Vec<LmbmHypothesis<T, N>>,
}

#[cfg(feature = "alloc")]
impl<T: RealField + Float + Copy, const N: usize> LmbmState<T, N> {
    /// Creates an empty LMBM state.
    #[inline]
    pub fn new() -> Self {
        Self {
            hypotheses: Vec::new(),
        }
    }

    /// Creates an LMBM state with a single empty hypothesis.
    #[inline]
    pub fn with_empty_hypothesis() -> Self {
        Self {
            hypotheses: vec![LmbmHypothesis::empty(T::zero())],
        }
    }

    /// Returns the number of hypotheses.
    #[inline]
    pub fn num_hypotheses(&self) -> usize {
        self.hypotheses.len()
    }

    /// Normalizes hypothesis log weights so the maximum is 0.
    ///
    /// This prevents numerical underflow in the weights while
    /// preserving their relative values.
    pub fn normalize_log_weights(&mut self) {
        if self.hypotheses.is_empty() {
            return;
        }

        // Find maximum log weight
        let max_log_weight = self
            .hypotheses
            .iter()
            .map(|h| h.log_weight)
            .fold(Float::neg_infinity(), |a, b| if a > b { a } else { b });

        // Subtract maximum (equivalent to dividing weights by max)
        for hypothesis in &mut self.hypotheses {
            hypothesis.log_weight -= max_log_weight;
        }
    }

    /// Prunes hypotheses with log weight below the threshold.
    pub fn prune_by_weight(&mut self, log_threshold: T) {
        self.hypotheses.retain(|h| h.log_weight >= log_threshold);
    }

    /// Keeps only the top k hypotheses by weight.
    pub fn keep_top_k(&mut self, k: usize) {
        if self.hypotheses.len() <= k {
            return;
        }

        self.hypotheses
            .sort_by(|a, b| b.log_weight.partial_cmp(&a.log_weight).unwrap());
        self.hypotheses.truncate(k);
    }

    /// Computes the marginal existence probability for each track label.
    ///
    /// Returns a list of (label, marginal_existence) pairs.
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
        let total_weight: T = weights.iter().fold(T::zero(), |acc, &w| acc + w);

        // Accumulate weighted existence per label
        let mut existence_map: BTreeMap<Label, T> = BTreeMap::new();

        for (hypothesis, &weight) in self.hypotheses.iter().zip(&weights) {
            let normalized_weight = weight / total_weight;
            for track in &hypothesis.tracks {
                *existence_map.entry(track.label).or_insert(T::zero()) +=
                    normalized_weight * track.existence;
            }
        }

        existence_map.into_iter().collect()
    }
}

#[cfg(feature = "alloc")]
impl<T: RealField + Float + Copy, const N: usize> Default for LmbmState<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Posterior Grid (precomputed Kalman posteriors)
// ============================================================================

/// Precomputed Kalman filter posteriors for all track-measurement pairs.
///
/// This avoids redundant computation during the update step when
/// the same posterior is needed for multiple association weights.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct PosteriorGrid<T: RealField, const N: usize> {
    /// Posteriors indexed by [track_idx][measurement_idx]
    /// Each entry is (updated_mean, updated_covariance, likelihood)
    pub posteriors: Vec<Vec<Option<(StateVector<T, N>, StateCovariance<T, N>, T)>>>,
}

#[cfg(feature = "alloc")]
impl<T: RealField + Copy, const N: usize> PosteriorGrid<T, N> {
    /// Creates an empty posterior grid.
    pub fn new(num_tracks: usize, num_measurements: usize) -> Self {
        Self {
            posteriors: vec![vec![None; num_measurements]; num_tracks],
        }
    }

    /// Gets the posterior for a track-measurement pair.
    #[inline]
    pub fn get(
        &self,
        track_idx: usize,
        meas_idx: usize,
    ) -> Option<&(StateVector<T, N>, StateCovariance<T, N>, T)> {
        self.posteriors
            .get(track_idx)
            .and_then(|row| row.get(meas_idx))
            .and_then(|p| p.as_ref())
    }

    /// Sets the posterior for a track-measurement pair.
    #[inline]
    pub fn set(
        &mut self,
        track_idx: usize,
        meas_idx: usize,
        posterior: (StateVector<T, N>, StateCovariance<T, N>, T),
    ) {
        if track_idx < self.posteriors.len() && meas_idx < self.posteriors[track_idx].len() {
            self.posteriors[track_idx][meas_idx] = Some(posterior);
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
    fn test_lmb_track_creation() {
        let label = Label::new(0, 0);
        let mean: StateVector<f64, 4> = StateVector::from_array([1.0, 2.0, 0.5, 0.5]);
        let cov: StateCovariance<f64, 4> = StateCovariance::identity();
        let state = GaussianState::new(1.0, mean, cov);

        let track = LmbTrack::new(label, 0.8, state);

        assert!(track.is_likely_existing());
        assert_eq!(track.num_components(), 1);
        assert!((track.existence - 0.8).abs() < 1e-10);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_lmb_track_set() {
        let mut track_set = LmbTrackSet::new();

        let label1 = Label::new(0, 0);
        let label2 = Label::new(0, 1);
        let mean: StateVector<f64, 4> = StateVector::from_array([0.0, 0.0, 0.0, 0.0]);
        let cov: StateCovariance<f64, 4> = StateCovariance::identity();
        let state = GaussianState::new(1.0, mean, cov);

        track_set.push(LmbTrack::new(label1, 0.9, state.clone()));
        track_set.push(LmbTrack::new(label2, 0.3, state));

        assert_eq!(track_set.len(), 2);
        assert!((track_set.expected_cardinality() - 1.2).abs() < 1e-10);

        // Prune low existence
        track_set.prune_by_existence(0.5);
        assert_eq!(track_set.len(), 1);
    }

    #[test]
    fn test_fixed_lmb_track() {
        let label = Label::new(0, 0);
        let mean: StateVector<f64, 4> = StateVector::from_array([1.0, 2.0, 0.0, 0.0]);
        let cov: StateCovariance<f64, 4> = StateCovariance::identity();
        let state = GaussianState::new(1.0, mean, cov);

        let track: FixedLmbTrack<f64, 4, 10> = FixedLmbTrack::new(label, 0.8, state);

        assert_eq!(track.num_components(), 1);
        assert_eq!(track.capacity(), 10);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_lmbm_hypothesis() {
        let label = Label::new(0, 0);
        let mean: StateVector<f64, 4> = StateVector::from_array([0.0, 0.0, 0.0, 0.0]);
        let cov: StateCovariance<f64, 4> = StateCovariance::identity();
        let state = GaussianState::new(1.0, mean, cov);
        let track = BernoulliTrack::new(label, 0.9, state);

        let hypothesis = LmbmHypothesis::new(0.0, vec![track]); // log_weight = 0 -> weight = 1

        assert_eq!(hypothesis.num_tracks(), 1);
        assert!((hypothesis.weight() - 1.0).abs() < 1e-10);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_lmbm_state_normalization() {
        let mut state = LmbmState::<f64, 4>::new();
        state.hypotheses.push(LmbmHypothesis::empty(-100.0)); // Very small weight
        state.hypotheses.push(LmbmHypothesis::empty(-99.0));

        state.normalize_log_weights();

        // After normalization, max log weight should be 0
        let max_log = state
            .hypotheses
            .iter()
            .map(|h| h.log_weight)
            .fold(f64::NEG_INFINITY, f64::max);
        assert!((max_log - 0.0).abs() < 1e-10);
    }
}
