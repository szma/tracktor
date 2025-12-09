//! Clutter models for false alarm generation
//!
//! Describes the statistical properties of clutter (false alarms)
//! in the surveillance region.

use nalgebra::RealField;
use num_traits::Float;

use crate::types::spaces::Measurement;

/// Trait for clutter models.
///
/// Clutter represents false alarms - measurements that do not originate
/// from any target of interest.
pub trait ClutterModel<T: RealField, const M: usize> {
    /// Returns the expected number of clutter measurements per scan.
    ///
    /// This is often denoted as λ (lambda) in the literature.
    fn clutter_rate(&self) -> T;

    /// Returns the spatial density of clutter at a given measurement.
    ///
    /// For uniform clutter over volume V, this is 1/V.
    fn clutter_density(&self, measurement: &Measurement<T, M>) -> T;

    /// Returns the clutter intensity (rate × density).
    ///
    /// This is the expected number of clutter measurements per unit volume.
    fn clutter_intensity(&self, measurement: &Measurement<T, M>) -> T {
        self.clutter_rate() * self.clutter_density(measurement)
    }
}

// ============================================================================
// Uniform Clutter
// ============================================================================

/// Uniform clutter model over a rectangular surveillance region.
///
/// Clutter is uniformly distributed over the surveillance volume.
#[derive(Debug, Clone)]
pub struct UniformClutter<T: RealField, const M: usize> {
    /// Expected number of clutter measurements per scan
    clutter_rate: T,
    /// Volume of the surveillance region
    volume: T,
}

impl<T: RealField + Float + Copy, const M: usize> UniformClutter<T, M> {
    /// Creates a uniform clutter model.
    ///
    /// # Arguments
    /// - `clutter_rate`: Expected number of false alarms per scan (must be >= 0)
    /// - `volume`: Volume/area of the surveillance region (must be > 0)
    ///
    /// # Panics
    /// Panics if `volume <= 0` or `clutter_rate < 0`.
    pub fn new(clutter_rate: T, volume: T) -> Self {
        assert!(volume > T::zero(), "Clutter volume must be positive");
        assert!(clutter_rate >= T::zero(), "Clutter rate must be non-negative");
        Self { clutter_rate, volume }
    }

    /// Creates a uniform clutter model from region bounds.
    ///
    /// # Arguments
    /// - `clutter_rate`: Expected number of false alarms per scan (must be >= 0)
    /// - `bounds`: Array of (min, max) pairs for each dimension (max > min for all)
    ///
    /// # Panics
    /// Panics if any dimension has `max <= min` or `clutter_rate < 0`.
    pub fn from_bounds(clutter_rate: T, bounds: [(T, T); M]) -> Self {
        assert!(clutter_rate >= T::zero(), "Clutter rate must be non-negative");
        for (i, (min, max)) in bounds.iter().enumerate() {
            assert!(*max > *min, "Bound {} must have max > min", i);
        }
        let volume = bounds.iter().fold(T::one(), |acc, (min, max)| {
            acc * (*max - *min)
        });
        Self { clutter_rate, volume }
    }
}

impl<T: RealField + Float + Copy, const M: usize> ClutterModel<T, M> for UniformClutter<T, M> {
    fn clutter_rate(&self) -> T {
        self.clutter_rate
    }

    fn clutter_density(&self, _measurement: &Measurement<T, M>) -> T {
        T::one() / self.volume
    }
}

/// Uniform clutter in a 2D rectangular region.
#[derive(Debug, Clone)]
pub struct UniformClutter2D<T: RealField> {
    /// Expected number of clutter measurements per scan
    clutter_rate: T,
    /// X bounds (min, max)
    x_bounds: (T, T),
    /// Y bounds (min, max)
    y_bounds: (T, T),
    /// Precomputed area
    area: T,
}

impl<T: RealField + Float + Copy> UniformClutter2D<T> {
    /// Creates uniform 2D clutter.
    ///
    /// # Arguments
    /// - `clutter_rate`: Expected number of false alarms per scan (must be >= 0)
    /// - `x_bounds`: (min, max) bounds for x dimension (max > min)
    /// - `y_bounds`: (min, max) bounds for y dimension (max > min)
    ///
    /// # Panics
    /// Panics if bounds are invalid or clutter_rate is negative.
    pub fn new(clutter_rate: T, x_bounds: (T, T), y_bounds: (T, T)) -> Self {
        assert!(clutter_rate >= T::zero(), "Clutter rate must be non-negative");
        assert!(x_bounds.1 > x_bounds.0, "x_bounds must have max > min");
        assert!(y_bounds.1 > y_bounds.0, "y_bounds must have max > min");
        let area = (x_bounds.1 - x_bounds.0) * (y_bounds.1 - y_bounds.0);
        Self { clutter_rate, x_bounds, y_bounds, area }
    }

    /// Checks if a measurement is within the surveillance region.
    pub fn contains(&self, measurement: &Measurement<T, 2>) -> bool {
        let x = *measurement.index(0);
        let y = *measurement.index(1);

        x >= self.x_bounds.0 && x <= self.x_bounds.1 &&
        y >= self.y_bounds.0 && y <= self.y_bounds.1
    }
}

impl<T: RealField + Float + Copy> ClutterModel<T, 2> for UniformClutter2D<T> {
    fn clutter_rate(&self) -> T {
        self.clutter_rate
    }

    fn clutter_density(&self, _measurement: &Measurement<T, 2>) -> T {
        T::one() / self.area
    }
}

/// Uniform clutter in a 3D rectangular region.
#[derive(Debug, Clone)]
pub struct UniformClutter3D<T: RealField> {
    /// Expected number of clutter measurements per scan
    clutter_rate: T,
    /// X bounds (min, max)
    x_bounds: (T, T),
    /// Y bounds (min, max)
    y_bounds: (T, T),
    /// Z bounds (min, max)
    z_bounds: (T, T),
    /// Precomputed volume
    volume: T,
}

impl<T: RealField + Float + Copy> UniformClutter3D<T> {
    /// Creates uniform 3D clutter.
    ///
    /// # Arguments
    /// - `clutter_rate`: Expected number of false alarms per scan (must be >= 0)
    /// - `x_bounds`: (min, max) bounds for x dimension (max > min)
    /// - `y_bounds`: (min, max) bounds for y dimension (max > min)
    /// - `z_bounds`: (min, max) bounds for z dimension (max > min)
    ///
    /// # Panics
    /// Panics if bounds are invalid or clutter_rate is negative.
    pub fn new(clutter_rate: T, x_bounds: (T, T), y_bounds: (T, T), z_bounds: (T, T)) -> Self {
        assert!(clutter_rate >= T::zero(), "Clutter rate must be non-negative");
        assert!(x_bounds.1 > x_bounds.0, "x_bounds must have max > min");
        assert!(y_bounds.1 > y_bounds.0, "y_bounds must have max > min");
        assert!(z_bounds.1 > z_bounds.0, "z_bounds must have max > min");
        let volume = (x_bounds.1 - x_bounds.0) *
                     (y_bounds.1 - y_bounds.0) *
                     (z_bounds.1 - z_bounds.0);
        Self { clutter_rate, x_bounds, y_bounds, z_bounds, volume }
    }
}

impl<T: RealField + Float + Copy> ClutterModel<T, 3> for UniformClutter3D<T> {
    fn clutter_rate(&self) -> T {
        self.clutter_rate
    }

    fn clutter_density(&self, _measurement: &Measurement<T, 3>) -> T {
        T::one() / self.volume
    }
}

// ============================================================================
// Non-uniform Clutter
// ============================================================================

/// Gaussian-shaped clutter density (for testing or specific scenarios).
///
/// Clutter is more likely in certain regions of the measurement space.
#[derive(Debug, Clone)]
pub struct GaussianClutter<T: RealField, const M: usize> {
    /// Expected number of clutter measurements per scan
    clutter_rate: T,
    /// Mean of the clutter distribution
    mean: Measurement<T, M>,
    /// Covariance of the clutter distribution
    covariance: crate::types::spaces::MeasurementCovariance<T, M>,
}

impl<T: RealField + Float + Copy, const M: usize> GaussianClutter<T, M> {
    /// Creates a Gaussian clutter model.
    ///
    /// # Arguments
    /// - `clutter_rate`: Expected number of false alarms per scan (must be >= 0)
    /// - `mean`: Mean location of clutter distribution
    /// - `covariance`: Covariance of clutter distribution (must be positive definite)
    ///
    /// # Panics
    /// Panics if `clutter_rate < 0` or covariance is not positive definite.
    pub fn new(
        clutter_rate: T,
        mean: Measurement<T, M>,
        covariance: crate::types::spaces::MeasurementCovariance<T, M>,
    ) -> Self {
        assert!(clutter_rate >= T::zero(), "Clutter rate must be non-negative");
        assert!(
            covariance.determinant_cholesky().is_some(),
            "Clutter covariance must be positive definite"
        );
        Self { clutter_rate, mean, covariance }
    }
}

impl<T: RealField + Float + Copy, const M: usize> ClutterModel<T, M> for GaussianClutter<T, M> {
    fn clutter_rate(&self) -> T {
        self.clutter_rate
    }

    fn clutter_density(&self, measurement: &Measurement<T, M>) -> T {
        use crate::types::spaces::ComputeInnovation;

        // Compute Gaussian density
        // Note: We validated in constructor that covariance is positive definite,
        // so this should not fail. If it does, return zero (very low density).
        let diff = measurement.innovation(self.mean);
        crate::types::gaussian::gaussian_likelihood(&diff, &crate::types::spaces::Covariance::from_matrix(*self.covariance.as_matrix()))
            .unwrap_or(T::zero())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_clutter_2d() {
        let clutter = UniformClutter2D::new(10.0_f64, (0.0, 100.0), (0.0, 100.0));

        assert!((clutter.clutter_rate() - 10.0).abs() < 1e-10);

        let measurement = Measurement::from_array([50.0, 50.0]);
        let density = clutter.clutter_density(&measurement);

        // Area = 100 * 100 = 10000, density = 1/10000
        assert!((density - 0.0001).abs() < 1e-10);
    }

    #[test]
    fn test_uniform_clutter_contains() {
        let clutter = UniformClutter2D::new(10.0_f64, (0.0, 100.0), (0.0, 100.0));

        let inside = Measurement::from_array([50.0, 50.0]);
        let outside = Measurement::from_array([150.0, 50.0]);

        assert!(clutter.contains(&inside));
        assert!(!clutter.contains(&outside));
    }

    #[test]
    fn test_clutter_intensity() {
        let clutter = UniformClutter2D::new(10.0_f64, (0.0, 100.0), (0.0, 100.0));
        let measurement = Measurement::from_array([50.0, 50.0]);

        let intensity = clutter.clutter_intensity(&measurement);
        // intensity = rate * density = 10 * 0.0001 = 0.001
        assert!((intensity - 0.001).abs() < 1e-10);
    }
}
