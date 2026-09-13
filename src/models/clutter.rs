//! Clutter models for false alarm generation
//!
//! Describes the statistical properties of clutter (false alarms)
//! in the surveillance region.

use nalgebra::RealField;
use num_traits::Float;

use crate::types::spaces::Measurement;

/// Returns a safe denominator for target-to-clutter likelihood ratios.
///
/// Finite-support clutter has exactly zero intensity outside its region, but a
/// target may still generate such a measurement. Using one shared positive
/// floor preserves that target-originated hypothesis while avoiding division
/// by zero in ratio-based association algorithms.
#[cfg(feature = "alloc")]
pub(crate) fn effective_clutter_intensity<T: RealField + Float>(intensity: T) -> T {
    let floor = T::from_f64(1e-300).unwrap();
    if intensity > floor { intensity } else { floor }
}

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
    /// Finite support when constructed from explicit bounds.
    bounds: Option<[(T, T); M]>,
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
        assert!(
            clutter_rate >= T::zero(),
            "Clutter rate must be non-negative"
        );
        Self {
            clutter_rate,
            volume,
            bounds: None,
        }
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
        assert!(
            clutter_rate >= T::zero(),
            "Clutter rate must be non-negative"
        );
        for (i, (min, max)) in bounds.iter().enumerate() {
            assert!(*max > *min, "Bound {} must have max > min", i);
        }
        let volume = bounds
            .iter()
            .fold(T::one(), |acc, (min, max)| acc * (*max - *min));
        Self {
            clutter_rate,
            volume,
            bounds: Some(bounds),
        }
    }

    /// Checks whether a measurement lies in the finite support.
    ///
    /// Models created with [`Self::new`] describe an abstract surveillance
    /// volume without coordinate bounds and therefore contain every coordinate.
    pub fn contains(&self, measurement: &Measurement<T, M>) -> bool {
        self.bounds.as_ref().is_none_or(|bounds| {
            bounds.iter().enumerate().all(|(index, &(min, max))| {
                let value = *measurement.index(index);
                value >= min && value <= max
            })
        })
    }
}

impl<T: RealField + Float + Copy, const M: usize> ClutterModel<T, M> for UniformClutter<T, M> {
    fn clutter_rate(&self) -> T {
        self.clutter_rate
    }

    fn clutter_density(&self, measurement: &Measurement<T, M>) -> T {
        if self.contains(measurement) {
            T::one() / self.volume
        } else {
            T::zero()
        }
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
        assert!(
            clutter_rate >= T::zero(),
            "Clutter rate must be non-negative"
        );
        assert!(x_bounds.1 > x_bounds.0, "x_bounds must have max > min");
        assert!(y_bounds.1 > y_bounds.0, "y_bounds must have max > min");
        let area = (x_bounds.1 - x_bounds.0) * (y_bounds.1 - y_bounds.0);
        Self {
            clutter_rate,
            x_bounds,
            y_bounds,
            area,
        }
    }

    /// Checks if a measurement is within the surveillance region.
    pub fn contains(&self, measurement: &Measurement<T, 2>) -> bool {
        let x = *measurement.index(0);
        let y = *measurement.index(1);

        x >= self.x_bounds.0 && x <= self.x_bounds.1 && y >= self.y_bounds.0 && y <= self.y_bounds.1
    }
}

impl<T: RealField + Float + Copy> ClutterModel<T, 2> for UniformClutter2D<T> {
    fn clutter_rate(&self) -> T {
        self.clutter_rate
    }

    fn clutter_density(&self, measurement: &Measurement<T, 2>) -> T {
        if self.contains(measurement) {
            T::one() / self.area
        } else {
            T::zero()
        }
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
        assert!(
            clutter_rate >= T::zero(),
            "Clutter rate must be non-negative"
        );
        assert!(x_bounds.1 > x_bounds.0, "x_bounds must have max > min");
        assert!(y_bounds.1 > y_bounds.0, "y_bounds must have max > min");
        assert!(z_bounds.1 > z_bounds.0, "z_bounds must have max > min");
        let volume =
            (x_bounds.1 - x_bounds.0) * (y_bounds.1 - y_bounds.0) * (z_bounds.1 - z_bounds.0);
        Self {
            clutter_rate,
            x_bounds,
            y_bounds,
            z_bounds,
            volume,
        }
    }

    /// Checks if a measurement is within the surveillance region.
    pub fn contains(&self, measurement: &Measurement<T, 3>) -> bool {
        let x = *measurement.index(0);
        let y = *measurement.index(1);
        let z = *measurement.index(2);

        x >= self.x_bounds.0
            && x <= self.x_bounds.1
            && y >= self.y_bounds.0
            && y <= self.y_bounds.1
            && z >= self.z_bounds.0
            && z <= self.z_bounds.1
    }
}

impl<T: RealField + Float + Copy> ClutterModel<T, 3> for UniformClutter3D<T> {
    fn clutter_rate(&self) -> T {
        self.clutter_rate
    }

    fn clutter_density(&self, measurement: &Measurement<T, 3>) -> T {
        if self.contains(measurement) {
            T::one() / self.volume
        } else {
            T::zero()
        }
    }
}

// ============================================================================
// Polar/Range-Bearing Clutter
// ============================================================================

/// Uniform clutter in range-bearing (polar) measurement space.
///
/// This clutter model is appropriate for radar-like sensors that measure
/// in (range, bearing) coordinates rather than Cartesian (x, y).
///
/// The surveillance region is defined by:
/// - Range bounds: [r_min, r_max]
/// - Bearing bounds: [θ_min, θ_max] in radians
///
/// The density is uniform with respect to the sensor's `[range, bearing]`
/// measurement coordinates. Its measurement-space volume is therefore
/// `(r_max - r_min) * (θ_max - θ_min)`.
#[derive(Debug, Clone)]
pub struct UniformClutterRangeBearing<T: RealField> {
    /// Expected number of clutter measurements per scan
    clutter_rate: T,
    /// Range bounds (min, max) in distance units
    range_bounds: (T, T),
    /// Bearing bounds (min, max) in radians, typically [-π, π] or [0, 2π]
    bearing_bounds: (T, T),
    /// Precomputed range-bearing measurement-space volume
    measurement_volume: T,
}

impl<T: RealField + Float + Copy> UniformClutterRangeBearing<T> {
    /// Creates uniform clutter in range-bearing space.
    ///
    /// # Arguments
    /// - `clutter_rate`: Expected number of false alarms per scan (must be >= 0)
    /// - `range_bounds`: (min, max) bounds for range (max > min >= 0)
    /// - `bearing_bounds`: (min, max) bounds for bearing in radians (max > min)
    ///
    /// # Panics
    /// Panics if bounds are invalid or clutter_rate is negative.
    ///
    /// # Example
    /// ```no_run
    /// use tracktor::prelude::*;
    ///
    /// // Clutter in a radar field of view: range 0-1000m, bearing ±60°
    /// let clutter = UniformClutterRangeBearing::new(
    ///     10.0,                    // 10 expected false alarms per scan
    ///     (0.0, 1000.0),           // range: 0 to 1000 meters
    ///     (-1.047, 1.047),         // bearing: -60° to +60° (in radians)
    /// );
    /// ```
    pub fn new(clutter_rate: T, range_bounds: (T, T), bearing_bounds: (T, T)) -> Self {
        assert!(
            clutter_rate >= T::zero(),
            "Clutter rate must be non-negative"
        );
        assert!(
            range_bounds.0 >= T::zero(),
            "Minimum range must be non-negative"
        );
        assert!(
            range_bounds.1 > range_bounds.0,
            "range_bounds must have max > min"
        );
        assert!(
            bearing_bounds.1 > bearing_bounds.0,
            "bearing_bounds must have max > min"
        );

        let measurement_volume =
            (range_bounds.1 - range_bounds.0) * (bearing_bounds.1 - bearing_bounds.0);

        Self {
            clutter_rate,
            range_bounds,
            bearing_bounds,
            measurement_volume,
        }
    }

    /// Creates uniform clutter for a full 360° field of view.
    ///
    /// # Arguments
    /// - `clutter_rate`: Expected number of false alarms per scan
    /// - `max_range`: Maximum detection range (min range is 0)
    pub fn full_circle(clutter_rate: T, max_range: T) -> Self {
        let pi = T::from_f64(core::f64::consts::PI).unwrap();
        Self::new(clutter_rate, (T::zero(), max_range), (-pi, pi))
    }

    /// Checks if a measurement is within the surveillance region.
    pub fn contains(&self, measurement: &Measurement<T, 2>) -> bool {
        let range = *measurement.index(0);
        let bearing = *measurement.index(1);

        range >= self.range_bounds.0
            && range <= self.range_bounds.1
            && bearing >= self.bearing_bounds.0
            && bearing <= self.bearing_bounds.1
    }
}

impl<T: RealField + Float + Copy> ClutterModel<T, 2> for UniformClutterRangeBearing<T> {
    fn clutter_rate(&self) -> T {
        self.clutter_rate
    }

    fn clutter_density(&self, measurement: &Measurement<T, 2>) -> T {
        if self.contains(measurement) {
            T::one() / self.measurement_volume
        } else {
            T::zero()
        }
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
        assert!(
            clutter_rate >= T::zero(),
            "Clutter rate must be non-negative"
        );
        assert!(
            covariance.determinant_cholesky().is_some(),
            "Clutter covariance must be positive definite"
        );
        Self {
            clutter_rate,
            mean,
            covariance,
        }
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
        crate::types::gaussian::gaussian_likelihood(
            &diff,
            &crate::types::spaces::Covariance::from_matrix(*self.covariance.as_matrix()),
        )
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
    fn test_uniform_clutter_3d_contains() {
        let clutter = UniformClutter3D::new(10.0_f64, (0.0, 100.0), (-50.0, 50.0), (10.0, 20.0));

        let inside = Measurement::from_array([50.0, 0.0, 15.0]);
        let on_boundary = Measurement::from_array([0.0, 50.0, 20.0]);
        let outside_x = Measurement::from_array([101.0, 0.0, 15.0]);
        let outside_y = Measurement::from_array([50.0, -51.0, 15.0]);
        let outside_z = Measurement::from_array([50.0, 0.0, 9.0]);

        assert!(clutter.contains(&inside));
        assert!(clutter.contains(&on_boundary));
        assert!(!clutter.contains(&outside_x));
        assert!(!clutter.contains(&outside_y));
        assert!(!clutter.contains(&outside_z));
    }

    #[test]
    fn test_clutter_intensity() {
        let clutter = UniformClutter2D::new(10.0_f64, (0.0, 100.0), (0.0, 100.0));
        let measurement = Measurement::from_array([50.0, 50.0]);

        let intensity = clutter.clutter_intensity(&measurement);
        // intensity = rate * density = 10 * 0.0001 = 0.001
        assert!((intensity - 0.001).abs() < 1e-10);
    }

    #[test]
    fn test_uniform_clutter_range_bearing() {
        use core::f64::consts::PI;

        // Full 360° coverage, range 0-100
        let clutter = UniformClutterRangeBearing::new(10.0_f64, (0.0, 100.0), (-PI, PI));

        assert!((clutter.clutter_rate() - 10.0).abs() < 1e-10);

        // Uniform in [range, bearing] measurement coordinates.
        let expected_volume = 100.0 * 2.0 * PI;
        let density = clutter.clutter_density(&Measurement::from_array([50.0, 0.0]));
        let expected_density = 1.0 / expected_volume;

        assert!(
            (density - expected_density).abs() < 1e-10,
            "Expected density {}, got {}",
            expected_density,
            density
        );
    }

    #[test]
    fn test_range_bearing_clutter_contains() {
        use core::f64::consts::PI;

        let clutter =
            UniformClutterRangeBearing::new(10.0_f64, (10.0, 100.0), (-PI / 2.0, PI / 2.0));

        // Inside: range 50, bearing 0
        let inside = Measurement::from_array([50.0, 0.0]);
        assert!(clutter.contains(&inside));

        // Outside: range too small
        let too_close = Measurement::from_array([5.0, 0.0]);
        assert!(!clutter.contains(&too_close));

        // Outside: range too large
        let too_far = Measurement::from_array([150.0, 0.0]);
        assert!(!clutter.contains(&too_far));

        // Outside: bearing outside bounds
        let wrong_bearing = Measurement::from_array([50.0, 2.0]); // ~115°, outside ±90°
        assert!(!clutter.contains(&wrong_bearing));
    }

    #[test]
    fn test_uniform_clutter_density_is_zero_outside_support() {
        let rectangular = UniformClutter2D::new(1.0_f64, (0.0, 1.0), (0.0, 1.0));
        assert_eq!(
            rectangular.clutter_density(&Measurement::from_array([2.0, 0.5])),
            0.0
        );

        let generic = UniformClutter::<f64, 2>::from_bounds(1.0, [(0.0, 1.0), (0.0, 2.0)]);
        assert_eq!(
            generic.clutter_density(&Measurement::from_array([0.5, 3.0])),
            0.0
        );

        let polar = UniformClutterRangeBearing::new(1.0, (0.0, 10.0), (-1.0, 1.0));
        assert_eq!(
            polar.clutter_density(&Measurement::from_array([11.0, 0.0])),
            0.0
        );
    }

    #[test]
    fn test_range_bearing_clutter_full_circle() {
        let clutter = UniformClutterRangeBearing::full_circle(5.0_f64, 200.0);

        assert!((clutter.clutter_rate() - 5.0).abs() < 1e-10);

        // Should contain any point within range
        let inside = Measurement::from_array([100.0, 3.0]); // ~172°
        assert!(clutter.contains(&inside));
    }
}
