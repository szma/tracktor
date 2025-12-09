//! Track labels for labeled multi-target filters (GLMB/LMB)
//!
//! Labels uniquely identify targets across time, enabling track continuity
//! and trajectory estimation.

use super::gaussian::GaussianState;
use nalgebra::RealField;

// ============================================================================
// Track Label
// ============================================================================

/// A unique identifier for a track.
///
/// The tuple `(birth_time, index)` guarantees uniqueness within a filter instance.
/// - `birth_time`: The time step when the target was born
/// - `index`: A unique index among targets born at the same time
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Label {
    /// Time step when the target was born
    pub birth_time: u32,
    /// Unique index among targets born at the same time
    pub index: u32,
}

impl Label {
    /// Creates a new label.
    #[inline]
    pub const fn new(birth_time: u32, index: u32) -> Self {
        Self { birth_time, index }
    }
}

impl core::fmt::Display for Label {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "({}, {})", self.birth_time, self.index)
    }
}

// ============================================================================
// Label Generator
// ============================================================================

/// Generates unique labels for new tracks.
#[derive(Debug, Clone)]
pub struct LabelGenerator {
    current_time: u32,
    next_index: u32,
}

impl LabelGenerator {
    /// Creates a new label generator starting at time 0.
    #[inline]
    pub const fn new() -> Self {
        Self {
            current_time: 0,
            next_index: 0,
        }
    }

    /// Creates a label generator starting at a specific time.
    #[inline]
    pub const fn at_time(time: u32) -> Self {
        Self {
            current_time: time,
            next_index: 0,
        }
    }

    /// Advances to the next time step.
    #[inline]
    pub fn advance_time(&mut self) {
        self.current_time += 1;
        self.next_index = 0;
    }

    /// Sets the current time step.
    #[inline]
    pub fn set_time(&mut self, time: u32) {
        self.current_time = time;
        self.next_index = 0;
    }

    /// Returns the current time step.
    #[inline]
    pub fn current_time(&self) -> u32 {
        self.current_time
    }

    /// Generates a new unique label at the current time.
    #[inline]
    pub fn next_label(&mut self) -> Label {
        let label = Label::new(self.current_time, self.next_index);
        self.next_index += 1;
        label
    }

    /// Generates `n` new unique labels at the current time.
    #[cfg(feature = "alloc")]
    pub fn next_labels(&mut self, n: usize) -> alloc::vec::Vec<Label> {
        (0..n).map(|_| self.next_label()).collect()
    }
}

impl Default for LabelGenerator {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Bernoulli Track
// ============================================================================

/// A labeled Bernoulli track with existence probability.
///
/// Used in LMB and GLMB filters where each track has an associated
/// probability of existence.
#[derive(Debug, Clone, PartialEq)]
pub struct BernoulliTrack<T: RealField, const N: usize> {
    /// Unique label identifying this track
    pub label: Label,
    /// Probability that this track exists
    pub existence: T,
    /// Gaussian state estimate (conditioned on existence)
    pub state: GaussianState<T, N>,
}

impl<T: RealField + Copy, const N: usize> BernoulliTrack<T, N> {
    /// Creates a new Bernoulli track.
    #[inline]
    pub fn new(label: Label, existence: T, state: GaussianState<T, N>) -> Self {
        Self {
            label,
            existence,
            state,
        }
    }

    /// Returns true if this track is likely to exist (existence > 0.5).
    #[inline]
    pub fn is_likely_existing(&self) -> bool {
        self.existence > T::from_f64(0.5).unwrap()
    }

    /// Returns the expected weight (existence probability × state weight).
    #[inline]
    pub fn expected_weight(&self) -> T {
        self.existence * self.state.weight
    }
}

// ============================================================================
// Labeled Gaussian Component
// ============================================================================

/// A Gaussian component with an attached label.
///
/// Used in GLMB filters where each hypothesis assigns labels to components.
#[derive(Debug, Clone, PartialEq)]
pub struct LabeledGaussian<T: RealField, const N: usize> {
    /// Unique label
    pub label: Label,
    /// Gaussian state
    pub state: GaussianState<T, N>,
}

impl<T: RealField + Copy, const N: usize> LabeledGaussian<T, N> {
    /// Creates a new labeled Gaussian.
    #[inline]
    pub fn new(label: Label, state: GaussianState<T, N>) -> Self {
        Self { label, state }
    }
}

// ============================================================================
// Hypothesis (for GLMB)
// ============================================================================

/// A GLMB hypothesis representing a possible set of existing targets.
///
/// Each hypothesis assigns labels to tracks and has an associated weight.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct Hypothesis {
    /// Weight of this hypothesis
    pub weight: f64,
    /// Set of labels in this hypothesis (indices into a track list)
    pub labels: alloc::vec::Vec<Label>,
}

#[cfg(feature = "alloc")]
impl Hypothesis {
    /// Creates a new hypothesis.
    #[inline]
    pub fn new(weight: f64, labels: alloc::vec::Vec<Label>) -> Self {
        Self { weight, labels }
    }

    /// Creates an empty hypothesis (no targets).
    #[inline]
    pub fn empty(weight: f64) -> Self {
        Self {
            weight,
            labels: alloc::vec::Vec::new(),
        }
    }

    /// Returns the number of targets in this hypothesis.
    #[inline]
    pub fn cardinality(&self) -> usize {
        self.labels.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::spaces::{StateCovariance, StateVector};

    #[test]
    fn test_label_generation() {
        let mut gen = LabelGenerator::new();

        let l1 = gen.next_label();
        let l2 = gen.next_label();

        assert_eq!(l1, Label::new(0, 0));
        assert_eq!(l2, Label::new(0, 1));

        gen.advance_time();
        let l3 = gen.next_label();

        assert_eq!(l3, Label::new(1, 0));
    }

    #[test]
    fn test_bernoulli_track() {
        let label = Label::new(0, 0);
        let mean: StateVector<f64, 4> = StateVector::from_array([0.0, 0.0, 1.0, 0.0]);
        let cov: StateCovariance<f64, 4> = StateCovariance::identity();
        let state = GaussianState::new(1.0, mean, cov);

        let track = BernoulliTrack::new(label, 0.8, state);

        assert!(track.is_likely_existing());
        assert!((track.existence - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_label_display() {
        let label = Label::new(5, 3);
        assert_eq!(format!("{}", label), "(5, 3)");
    }
}
