//! Birth models for spontaneous target appearance
//!
//! Describes how new targets appear in the surveillance region.

use nalgebra::RealField;
use num_traits::Float;

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use crate::types::spaces::{StateVector, StateCovariance, Measurement};
use crate::types::gaussian::GaussianState;

/// Trait for birth models.
///
/// Birth models describe the spontaneous appearance of new targets
/// that were not present in the previous time step.
pub trait BirthModel<T: RealField, const N: usize> {
    /// Returns the birth intensity (Gaussian mixture components).
    ///
    /// Returns a slice to avoid cloning on every predict step.
    #[cfg(feature = "alloc")]
    fn birth_components(&self) -> &[GaussianState<T, N>];

    /// Returns the total expected number of births per time step.
    fn total_birth_mass(&self) -> T;
}

// ============================================================================
// Fixed Birth Model
// ============================================================================

/// Birth model with fixed birth locations.
///
/// New targets can appear at predefined locations with specified intensities.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct FixedBirthModel<T: RealField, const N: usize> {
    /// Birth components (fixed locations)
    components: Vec<GaussianState<T, N>>,
}

#[cfg(feature = "alloc")]
impl<T: RealField + Copy, const N: usize> FixedBirthModel<T, N> {
    /// Creates an empty birth model.
    pub fn new() -> Self {
        Self { components: Vec::new() }
    }

    /// Creates a birth model from a vector of components.
    pub fn from_components(components: Vec<GaussianState<T, N>>) -> Self {
        Self { components }
    }

    /// Adds a birth component.
    pub fn add_component(&mut self, component: GaussianState<T, N>) {
        self.components.push(component);
    }

    /// Adds a birth location with specified weight and covariance.
    ///
    /// # Arguments
    /// - `weight`: Birth weight/intensity (must be >= 0)
    /// - `mean`: Mean state of the birth component
    /// - `covariance`: Covariance of the birth component (must be positive definite)
    ///
    /// # Panics
    /// Panics if `weight < 0` or covariance is not positive definite.
    pub fn add_birth_location(&mut self, weight: T, mean: StateVector<T, N>, covariance: StateCovariance<T, N>) {
        assert!(weight >= T::zero(), "Birth weight must be non-negative");
        assert!(
            covariance.determinant_cholesky().is_some(),
            "Birth covariance must be positive definite"
        );
        self.components.push(GaussianState::new(weight, mean, covariance));
    }
}

#[cfg(feature = "alloc")]
impl<T: RealField + Copy, const N: usize> Default for FixedBirthModel<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "alloc")]
impl<T: RealField + Copy, const N: usize> BirthModel<T, N> for FixedBirthModel<T, N> {
    fn birth_components(&self) -> &[GaussianState<T, N>] {
        &self.components
    }

    fn total_birth_mass(&self) -> T {
        self.components.iter().fold(T::zero(), |acc, c| acc + c.weight)
    }
}

// ============================================================================
// Adaptive Birth Model
// ============================================================================

/// Measurement-driven adaptive birth model.
///
/// Creates birth components from unassociated measurements.
/// This is useful when target entry points are unknown.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct AdaptiveBirthModel<T: RealField, const N: usize, const M: usize> {
    /// Weight assigned to each birth component
    birth_weight: T,
    /// Initial velocity covariance (for CV models)
    velocity_covariance: T,
    /// Base birth components (always included)
    base_components: Vec<GaussianState<T, N>>,
    /// Function to expand measurement to state (e.g., add zero velocity)
    _marker: core::marker::PhantomData<[T; M]>,
}

#[cfg(feature = "alloc")]
impl<T: RealField + Float + Copy> AdaptiveBirthModel<T, 4, 2> {
    /// Creates an adaptive birth model for 2D constant velocity tracking.
    ///
    /// State: [x, y, vx, vy], Measurement: [x, y]
    ///
    /// # Arguments
    /// - `birth_weight`: Weight for each birth component (must be >= 0)
    /// - `velocity_covariance`: Initial velocity variance (must be > 0)
    ///
    /// # Panics
    /// Panics if `birth_weight < 0` or `velocity_covariance <= 0`.
    pub fn new_cv2d(birth_weight: T, velocity_covariance: T) -> Self {
        assert!(birth_weight >= T::zero(), "Birth weight must be non-negative");
        assert!(velocity_covariance > T::zero(), "Velocity covariance must be positive");
        Self {
            birth_weight,
            velocity_covariance,
            base_components: Vec::new(),
            _marker: core::marker::PhantomData,
        }
    }

    /// Adds a base birth component.
    pub fn add_base_component(&mut self, component: GaussianState<T, 4>) {
        self.base_components.push(component);
    }

    /// Creates birth components from measurements.
    pub fn birth_from_measurements(&self, measurements: &[Measurement<T, 2>]) -> Vec<GaussianState<T, 4>> {
        let mut components = self.base_components.clone();

        for meas in measurements {
            let mean = StateVector::from_array([
                *meas.index(0),
                *meas.index(1),
                T::zero(),
                T::zero(),
            ]);

            // Covariance: position from measurement, large velocity uncertainty
            let zero = T::zero();
            let pos_cov = T::one(); // Small position uncertainty
            let vel_cov = self.velocity_covariance;

            let covariance = StateCovariance::from_matrix(nalgebra::matrix![
                pos_cov, zero, zero, zero;
                zero, pos_cov, zero, zero;
                zero, zero, vel_cov, zero;
                zero, zero, zero, vel_cov
            ]);

            components.push(GaussianState::new(self.birth_weight, mean, covariance));
        }

        components
    }
}

#[cfg(feature = "alloc")]
impl<T: RealField + Float + Copy> BirthModel<T, 4> for AdaptiveBirthModel<T, 4, 2> {
    fn birth_components(&self) -> &[GaussianState<T, 4>] {
        &self.base_components
    }

    fn total_birth_mass(&self) -> T {
        self.base_components.iter().fold(T::zero(), |acc, c| acc + c.weight)
    }
}

// ============================================================================
// Uniform Birth Model
// ============================================================================

/// Uniform birth model over a rectangular region.
///
/// Creates a grid of birth components covering the surveillance region.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct UniformBirthModel2D<T: RealField> {
    /// Total birth intensity (distributed among components)
    total_intensity: T,
    /// Grid of birth components
    components: Vec<GaussianState<T, 4>>,
}

#[cfg(feature = "alloc")]
impl<T: RealField + Float + Copy> UniformBirthModel2D<T> {
    /// Creates a uniform birth model on a grid.
    ///
    /// # Arguments
    /// - `total_intensity`: Total expected births per time step (must be >= 0)
    /// - `x_bounds`: X range (min, max), max must be > min
    /// - `y_bounds`: Y range (min, max), max must be > min
    /// - `grid_size`: Number of grid points per dimension (must be >= 2)
    /// - `position_std`: Position uncertainty standard deviation (must be > 0)
    /// - `velocity_std`: Velocity uncertainty standard deviation (must be > 0)
    ///
    /// # Panics
    /// Panics if any parameter constraint is violated.
    pub fn new(
        total_intensity: T,
        x_bounds: (T, T),
        y_bounds: (T, T),
        grid_size: usize,
        position_std: T,
        velocity_std: T,
    ) -> Self {
        assert!(total_intensity >= T::zero(), "Total intensity must be non-negative");
        assert!(x_bounds.1 > x_bounds.0, "x_bounds must have max > min");
        assert!(y_bounds.1 > y_bounds.0, "y_bounds must have max > min");
        assert!(grid_size >= 2, "Grid size must be at least 2");
        assert!(position_std > T::zero(), "Position std must be positive");
        assert!(velocity_std > T::zero(), "Velocity std must be positive");
        let mut components = Vec::with_capacity(grid_size * grid_size);
        let n = T::from_usize(grid_size).unwrap();
        let n_sq = T::from_usize(grid_size * grid_size).unwrap();
        let weight_per_component = total_intensity / n_sq;

        let dx = (x_bounds.1 - x_bounds.0) / (n - T::one());
        let dy = (y_bounds.1 - y_bounds.0) / (n - T::one());

        let pos_var = position_std * position_std;
        let vel_var = velocity_std * velocity_std;
        let zero = T::zero();

        for i in 0..grid_size {
            for j in 0..grid_size {
                let x = x_bounds.0 + T::from_usize(i).unwrap() * dx;
                let y = y_bounds.0 + T::from_usize(j).unwrap() * dy;

                let mean = StateVector::from_array([x, y, zero, zero]);
                let covariance = StateCovariance::from_matrix(nalgebra::matrix![
                    pos_var, zero, zero, zero;
                    zero, pos_var, zero, zero;
                    zero, zero, vel_var, zero;
                    zero, zero, zero, vel_var
                ]);

                components.push(GaussianState::new(weight_per_component, mean, covariance));
            }
        }

        Self { total_intensity, components }
    }
}

#[cfg(feature = "alloc")]
impl<T: RealField + Copy> BirthModel<T, 4> for UniformBirthModel2D<T> {
    fn birth_components(&self) -> &[GaussianState<T, 4>] {
        &self.components
    }

    fn total_birth_mass(&self) -> T {
        self.total_intensity
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "alloc")]
    #[test]
    fn test_fixed_birth_model() {
        let mut birth = FixedBirthModel::<f64, 4>::new();

        let mean = StateVector::from_array([0.0, 0.0, 0.0, 0.0]);
        let cov = StateCovariance::identity();
        birth.add_birth_location(0.1, mean, cov);

        let components = birth.birth_components();
        assert_eq!(components.len(), 1);
        assert!((birth.total_birth_mass() - 0.1).abs() < 1e-10);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_adaptive_birth_from_measurements() {
        let birth = AdaptiveBirthModel::<f64, 4, 2>::new_cv2d(0.01, 100.0);

        let measurements = [
            Measurement::from_array([10.0, 20.0]),
            Measurement::from_array([30.0, 40.0]),
        ];

        let components = birth.birth_from_measurements(&measurements);
        assert_eq!(components.len(), 2);

        // Check that birth locations match measurements
        assert!((components[0].mean.index(0) - 10.0).abs() < 1e-10);
        assert!((components[0].mean.index(1) - 20.0).abs() < 1e-10);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_uniform_birth_model() {
        let birth = UniformBirthModel2D::new(
            0.5_f64,      // total intensity
            (0.0, 100.0), // x bounds
            (0.0, 100.0), // y bounds
            3,            // grid size (3x3 = 9 components)
            10.0,         // position std
            5.0,          // velocity std
        );

        let components = birth.birth_components();
        assert_eq!(components.len(), 9);
        assert!((birth.total_birth_mass() - 0.5).abs() < 1e-10);
    }
}
