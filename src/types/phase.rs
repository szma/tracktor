// ============================================================================
// Filter Phase Markers
// ============================================================================

/// Marker type indicating a predicted filter state.
#[derive(Debug, Clone, Copy)]
pub struct Predicted;

/// Marker type indicating an updated filter state.
#[derive(Debug, Clone, Copy)]
pub struct Updated;
///
/// Statistics from a filter update step, reporting any numerical issues.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, Default)]
pub struct UpdateStats {
    /// Number of components where Kalman gain computation failed (singular innovation covariance)
    pub singular_covariance_count: usize,
    /// Number of components where likelihood computation returned zero or failed
    pub zero_likelihood_count: usize,
    /// Whether LBP converged (None if LBP not used, Some(true) if converged, Some(false) if hit max iterations)
    pub lbp_converged: Option<bool>,
    /// Number of LBP iterations run (None if LBP not used)
    pub lbp_iterations: Option<usize>,
}

#[cfg(feature = "alloc")]
impl UpdateStats {
    /// Returns true if any numerical issues were encountered.
    pub fn has_issues(&self) -> bool {
        self.singular_covariance_count > 0
            || self.zero_likelihood_count > 0
            || self.lbp_converged == Some(false)
    }
}
