//! Birth models for spontaneous target appearance
//!
//! Describes how new targets appear in the surveillance region.

use nalgebra::RealField;
use num_traits::Float;

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use crate::types::gaussian::GaussianState;
use crate::types::spaces::{Measurement, StateCovariance, StateVector};

/// Trait for birth models.
///
/// Birth models describe the spontaneous appearance of new targets
/// that were not present in the previous time step.
pub trait BirthModel<T: RealField, const N: usize> {
    /// Returns the birth intensity (Gaussian mixture components) as a slice.
    ///
    /// # Deprecation
    /// This method is deprecated in favor of [`birth_components_vec`] which
    /// supports dynamic birth component generation. New implementations
    /// should implement `birth_components_vec` instead and can leave this
    /// method with the default empty implementation.
    #[cfg(feature = "alloc")]
    #[deprecated(
        since = "0.4.0",
        note = "Use birth_components_vec() instead, which supports dynamic generation"
    )]
    fn birth_components(&self) -> &[GaussianState<T, N>] {
        &[]
    }

    /// Returns the birth intensity (Gaussian mixture components) as an owned Vec.
    ///
    /// This is the preferred method for new implementations, especially for
    /// adaptive birth models that generate components dynamically.
    ///
    /// The default implementation calls the deprecated `birth_components()`
    /// for backwards compatibility with existing implementations.
    #[cfg(feature = "alloc")]
    fn birth_components_vec(&self) -> Vec<GaussianState<T, N>> {
        #[allow(deprecated)]
        self.birth_components().to_vec()
    }

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
        Self {
            components: Vec::new(),
        }
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
    pub fn add_birth_location(
        &mut self,
        weight: T,
        mean: StateVector<T, N>,
        covariance: StateCovariance<T, N>,
    ) {
        assert!(weight >= T::zero(), "Birth weight must be non-negative");
        assert!(
            covariance.determinant_cholesky().is_some(),
            "Birth covariance must be positive definite"
        );
        self.components
            .push(GaussianState::new(weight, mean, covariance));
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
    #[allow(deprecated)]
    fn birth_components(&self) -> &[GaussianState<T, N>] {
        &self.components
    }

    fn birth_components_vec(&self) -> Vec<GaussianState<T, N>> {
        self.components.clone()
    }

    fn total_birth_mass(&self) -> T {
        self.components
            .iter()
            .fold(T::zero(), |acc, c| acc + c.weight)
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
        assert!(
            birth_weight >= T::zero(),
            "Birth weight must be non-negative"
        );
        assert!(
            velocity_covariance > T::zero(),
            "Velocity covariance must be positive"
        );
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
    pub fn birth_from_measurements(
        &self,
        measurements: &[Measurement<T, 2>],
    ) -> Vec<GaussianState<T, 4>> {
        let mut components = self.base_components.clone();

        for meas in measurements {
            let mean =
                StateVector::from_array([*meas.index(0), *meas.index(1), T::zero(), T::zero()]);

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
    #[allow(deprecated)]
    fn birth_components(&self) -> &[GaussianState<T, 4>] {
        &self.base_components
    }

    fn birth_components_vec(&self) -> Vec<GaussianState<T, 4>> {
        self.base_components.clone()
    }

    fn total_birth_mass(&self) -> T {
        self.base_components
            .iter()
            .fold(T::zero(), |acc, c| acc + c.weight)
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
        assert!(
            total_intensity >= T::zero(),
            "Total intensity must be non-negative"
        );
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

        Self {
            total_intensity,
            components,
        }
    }
}

#[cfg(feature = "alloc")]
impl<T: RealField + Copy> BirthModel<T, 4> for UniformBirthModel2D<T> {
    #[allow(deprecated)]
    fn birth_components(&self) -> &[GaussianState<T, 4>] {
        &self.components
    }

    fn birth_components_vec(&self) -> Vec<GaussianState<T, 4>> {
        self.components.clone()
    }

    fn total_birth_mass(&self) -> T {
        self.total_intensity
    }
}

// ============================================================================
// Measurement State Expander Trait
// ============================================================================

/// Trait for expanding measurements into full state vectors.
///
/// This trait defines how M-dimensional sensor measurements are expanded
/// into N-dimensional state vectors with appropriate covariance. This is
/// used by adaptive birth models to create birth components from measurements.
///
/// # Type Parameters
/// - `T`: Scalar type (typically f32 or f64)
/// - `N`: State dimension
/// - `M`: Measurement dimension
///
/// # Example
///
/// ```
/// use tracktor::models::MeasurementStateExpander;
/// use tracktor::types::spaces::{Measurement, StateVector, StateCovariance};
///
/// // Expander for 2D constant velocity: [x, y] -> [x, y, vx, vy]
/// struct MyExpander {
///     position_cov: f64,
///     velocity_cov: f64,
///     birth_weight: f64,
/// }
///
/// impl MeasurementStateExpander<f64, 4, 2> for MyExpander {
///     fn expand_measurement(&self, m: &Measurement<f64, 2>)
///         -> (StateVector<f64, 4>, StateCovariance<f64, 4>)
///     {
///         let mean = StateVector::from_array([*m.index(0), *m.index(1), 0.0, 0.0]);
///         let cov = StateCovariance::from_diagonal(&nalgebra::vector![
///             self.position_cov, self.position_cov,
///             self.velocity_cov, self.velocity_cov
///         ]);
///         (mean, cov)
///     }
///
///     fn birth_weight(&self) -> f64 { self.birth_weight }
/// }
/// ```
pub trait MeasurementStateExpander<T: RealField, const N: usize, const M: usize> {
    /// Expands a measurement into a full state vector with covariance.
    ///
    /// # Arguments
    /// - `measurement`: The M-dimensional sensor measurement
    ///
    /// # Returns
    /// A tuple of (state_mean, state_covariance) representing the birth component
    fn expand_measurement(
        &self,
        measurement: &Measurement<T, M>,
    ) -> (StateVector<T, N>, StateCovariance<T, N>);

    /// Returns the weight to assign to each birth component.
    ///
    /// This is the PHD weight (expected number of targets) for each
    /// measurement-derived birth component.
    fn birth_weight(&self) -> T;

    /// Returns the existence probability for LMB birth tracks.
    ///
    /// By default, this returns the same value as `birth_weight()`.
    /// Override this if you want different values for PHD vs LMB filters.
    fn birth_existence(&self) -> T {
        self.birth_weight()
    }
}

// ============================================================================
// Constant Velocity 2D Expander
// ============================================================================

/// Expander for 2D constant velocity model.
///
/// Expands position measurements [x, y] into CV state [x, y, vx, vy]
/// with zero initial velocity and configurable covariances.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct ConstantVelocity2DExpander<T: RealField> {
    /// Position covariance (variance in x and y)
    pub position_covariance: T,
    /// Velocity covariance (variance in vx and vy)
    pub velocity_covariance: T,
    /// Weight for each birth component
    pub birth_weight: T,
}

#[cfg(feature = "alloc")]
impl<T: RealField + Float + Copy> ConstantVelocity2DExpander<T> {
    /// Creates a new expander with the given parameters.
    ///
    /// # Arguments
    /// - `position_covariance`: Variance for position components (should match sensor noise)
    /// - `velocity_covariance`: Variance for velocity components (typically large)
    /// - `birth_weight`: Weight/existence probability for birth components
    ///
    /// # Panics
    /// Panics if any parameter is non-positive.
    pub fn new(position_covariance: T, velocity_covariance: T, birth_weight: T) -> Self {
        assert!(
            position_covariance > T::zero(),
            "Position covariance must be positive"
        );
        assert!(
            velocity_covariance > T::zero(),
            "Velocity covariance must be positive"
        );
        assert!(birth_weight > T::zero(), "Birth weight must be positive");
        Self {
            position_covariance,
            velocity_covariance,
            birth_weight,
        }
    }
}

#[cfg(feature = "alloc")]
impl<T: RealField + Float + Copy> MeasurementStateExpander<T, 4, 2>
    for ConstantVelocity2DExpander<T>
{
    fn expand_measurement(
        &self,
        measurement: &Measurement<T, 2>,
    ) -> (StateVector<T, 4>, StateCovariance<T, 4>) {
        let zero = T::zero();

        // State: [x, y, 0, 0] - position from measurement, zero velocity
        let mean =
            StateVector::from_array([*measurement.index(0), *measurement.index(1), zero, zero]);

        // Covariance: position from sensor noise, large velocity uncertainty
        let covariance = StateCovariance::from_matrix(nalgebra::matrix![
            self.position_covariance, zero, zero, zero;
            zero, self.position_covariance, zero, zero;
            zero, zero, self.velocity_covariance, zero;
            zero, zero, zero, self.velocity_covariance
        ]);

        (mean, covariance)
    }

    fn birth_weight(&self) -> T {
        self.birth_weight
    }
}

// ============================================================================
// Constant Velocity 3D Expander
// ============================================================================

/// Expander for 3D constant velocity model.
///
/// Expands position measurements [x, y, z] into CV state [x, y, z, vx, vy, vz]
/// with zero initial velocity and configurable covariances.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct ConstantVelocity3DExpander<T: RealField> {
    /// Position covariance (variance in x, y, z)
    pub position_covariance: T,
    /// Velocity covariance (variance in vx, vy, vz)
    pub velocity_covariance: T,
    /// Weight for each birth component
    pub birth_weight: T,
}

#[cfg(feature = "alloc")]
impl<T: RealField + Float + Copy> ConstantVelocity3DExpander<T> {
    /// Creates a new expander with the given parameters.
    pub fn new(position_covariance: T, velocity_covariance: T, birth_weight: T) -> Self {
        assert!(
            position_covariance > T::zero(),
            "Position covariance must be positive"
        );
        assert!(
            velocity_covariance > T::zero(),
            "Velocity covariance must be positive"
        );
        assert!(birth_weight > T::zero(), "Birth weight must be positive");
        Self {
            position_covariance,
            velocity_covariance,
            birth_weight,
        }
    }
}

#[cfg(feature = "alloc")]
impl<T: RealField + Float + Copy> MeasurementStateExpander<T, 6, 3>
    for ConstantVelocity3DExpander<T>
{
    fn expand_measurement(
        &self,
        measurement: &Measurement<T, 3>,
    ) -> (StateVector<T, 6>, StateCovariance<T, 6>) {
        let zero = T::zero();

        let mean = StateVector::from_array([
            *measurement.index(0),
            *measurement.index(1),
            *measurement.index(2),
            zero,
            zero,
            zero,
        ]);

        let covariance = StateCovariance::from_matrix(nalgebra::matrix![
            self.position_covariance, zero, zero, zero, zero, zero;
            zero, self.position_covariance, zero, zero, zero, zero;
            zero, zero, self.position_covariance, zero, zero, zero;
            zero, zero, zero, self.velocity_covariance, zero, zero;
            zero, zero, zero, zero, self.velocity_covariance, zero;
            zero, zero, zero, zero, zero, self.velocity_covariance
        ]);

        (mean, covariance)
    }

    fn birth_weight(&self) -> T {
        self.birth_weight
    }
}

// ============================================================================
// Range-Bearing to CV2D Expander
// ============================================================================

/// Expander for range-bearing measurements to 2D constant velocity state.
///
/// Converts polar measurements [range, bearing] to Cartesian CV state
/// [x, y, vx, vy] with proper covariance transformation via Jacobian.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct RangeBearingToCv2DExpander<T: RealField> {
    /// Sensor x position
    pub sensor_x: T,
    /// Sensor y position
    pub sensor_y: T,
    /// Range measurement variance
    pub range_variance: T,
    /// Bearing measurement variance (radians squared)
    pub bearing_variance: T,
    /// Velocity covariance
    pub velocity_covariance: T,
    /// Weight for each birth component
    pub birth_weight: T,
}

#[cfg(feature = "alloc")]
impl<T: RealField + Float + Copy> RangeBearingToCv2DExpander<T> {
    /// Creates a new expander for a sensor at the origin.
    pub fn new(
        range_variance: T,
        bearing_variance: T,
        velocity_covariance: T,
        birth_weight: T,
    ) -> Self {
        Self::at_position(
            T::zero(),
            T::zero(),
            range_variance,
            bearing_variance,
            velocity_covariance,
            birth_weight,
        )
    }

    /// Creates a new expander for a sensor at a specific position.
    pub fn at_position(
        sensor_x: T,
        sensor_y: T,
        range_variance: T,
        bearing_variance: T,
        velocity_covariance: T,
        birth_weight: T,
    ) -> Self {
        assert!(
            range_variance > T::zero(),
            "Range variance must be positive"
        );
        assert!(
            bearing_variance > T::zero(),
            "Bearing variance must be positive"
        );
        assert!(
            velocity_covariance > T::zero(),
            "Velocity covariance must be positive"
        );
        assert!(birth_weight > T::zero(), "Birth weight must be positive");
        Self {
            sensor_x,
            sensor_y,
            range_variance,
            bearing_variance,
            velocity_covariance,
            birth_weight,
        }
    }
}

#[cfg(feature = "alloc")]
impl<T: RealField + Float + Copy> MeasurementStateExpander<T, 4, 2>
    for RangeBearingToCv2DExpander<T>
{
    fn expand_measurement(
        &self,
        measurement: &Measurement<T, 2>,
    ) -> (StateVector<T, 4>, StateCovariance<T, 4>) {
        let range = *measurement.index(0);
        let bearing = *measurement.index(1);
        let zero = T::zero();

        // Convert polar to Cartesian
        let cos_b = Float::cos(bearing);
        let sin_b = Float::sin(bearing);
        let x = self.sensor_x + range * cos_b;
        let y = self.sensor_y + range * sin_b;

        let mean = StateVector::from_array([x, y, zero, zero]);

        // Transform covariance from polar to Cartesian via Jacobian
        // Jacobian: J = [[cos(b), -r*sin(b)], [sin(b), r*cos(b)]]
        let j = nalgebra::matrix![
            cos_b, -range * sin_b;
            sin_b, range * cos_b
        ];

        let polar_cov = nalgebra::matrix![
            self.range_variance, zero;
            zero, self.bearing_variance
        ];

        // Transform: P_cart = J * P_polar * J^T
        let cart_pos_cov = j * polar_cov * j.transpose();

        let covariance = StateCovariance::from_matrix(nalgebra::matrix![
            cart_pos_cov[(0, 0)], cart_pos_cov[(0, 1)], zero, zero;
            cart_pos_cov[(1, 0)], cart_pos_cov[(1, 1)], zero, zero;
            zero, zero, self.velocity_covariance, zero;
            zero, zero, zero, self.velocity_covariance
        ]);

        (mean, covariance)
    }

    fn birth_weight(&self) -> T {
        self.birth_weight
    }
}

// ============================================================================
// New Generic Adaptive Birth Model
// ============================================================================

/// Generic measurement-driven adaptive birth model.
///
/// Creates birth components dynamically from measurements using a
/// user-provided [`MeasurementStateExpander`]. This model supports both
/// PHD filters (via [`BirthModel`]) and LMB/GLMB filters (via
/// [`LabeledBirthModel`](crate::filters::lmb::LabeledBirthModel)).
///
/// # Type Parameters
/// - `T`: Scalar type
/// - `E`: Expander type implementing [`MeasurementStateExpander`]
/// - `N`: State dimension
/// - `M`: Measurement dimension
///
/// # Example
///
/// ```
/// use tracktor::models::{
///     MeasurementDrivenBirthModel, ConstantVelocity2DExpander, BirthModel,
/// };
/// use tracktor::types::spaces::Measurement;
///
/// // Create expander
/// let expander = ConstantVelocity2DExpander::new(10.0, 100.0, 0.05);
///
/// // Create birth model
/// let mut birth = MeasurementDrivenBirthModel::new(expander);
///
/// // Set measurements for next predict step
/// let measurements = vec![
///     Measurement::from_array([10.0, 20.0]),
///     Measurement::from_array([30.0, 40.0]),
/// ];
/// birth.set_measurements(measurements);
///
/// // Get birth components (for PHD filter)
/// let components = birth.birth_components_vec();
/// assert_eq!(components.len(), 2);
/// ```
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct MeasurementDrivenBirthModel<T, E, const N: usize, const M: usize>
where
    T: RealField,
    E: MeasurementStateExpander<T, N, M>,
{
    /// Expander to convert measurements to states
    expander: E,
    /// Base birth components (always included)
    base_components: Vec<GaussianState<T, N>>,
    /// Measurements to generate adaptive birth components from
    adaptive_measurements: Vec<Measurement<T, M>>,
}

#[cfg(feature = "alloc")]
impl<T, E, const N: usize, const M: usize> MeasurementDrivenBirthModel<T, E, N, M>
where
    T: RealField + Copy,
    E: MeasurementStateExpander<T, N, M>,
{
    /// Creates a new measurement-driven birth model with the given expander.
    pub fn new(expander: E) -> Self {
        Self {
            expander,
            base_components: Vec::new(),
            adaptive_measurements: Vec::new(),
        }
    }

    /// Creates a new birth model with base components.
    pub fn with_base_components(expander: E, base_components: Vec<GaussianState<T, N>>) -> Self {
        Self {
            expander,
            base_components,
            adaptive_measurements: Vec::new(),
        }
    }

    /// Adds a fixed base birth component that is always included.
    pub fn add_base_component(&mut self, component: GaussianState<T, N>) {
        self.base_components.push(component);
    }

    /// Sets the measurements to use for adaptive birth generation.
    ///
    /// Call this before the filter's predict step with measurements
    /// (typically unassociated measurements from the previous update).
    pub fn set_measurements(&mut self, measurements: Vec<Measurement<T, M>>) {
        self.adaptive_measurements = measurements;
    }

    /// Sets measurements from a slice (cloning).
    pub fn set_measurements_from_slice(&mut self, measurements: &[Measurement<T, M>]) {
        self.adaptive_measurements = measurements.to_vec();
    }

    /// Clears the adaptive measurements.
    pub fn clear_measurements(&mut self) {
        self.adaptive_measurements.clear();
    }

    /// Returns a reference to the expander.
    pub fn expander(&self) -> &E {
        &self.expander
    }

    /// Returns a mutable reference to the expander.
    pub fn expander_mut(&mut self) -> &mut E {
        &mut self.expander
    }

    /// Returns the number of adaptive measurements currently set.
    pub fn num_adaptive_measurements(&self) -> usize {
        self.adaptive_measurements.len()
    }

    /// Returns the number of base components.
    pub fn num_base_components(&self) -> usize {
        self.base_components.len()
    }
}

#[cfg(feature = "alloc")]
impl<T, E, const N: usize, const M: usize> BirthModel<T, N>
    for MeasurementDrivenBirthModel<T, E, N, M>
where
    T: RealField + Float + Copy,
    E: MeasurementStateExpander<T, N, M>,
{
    fn birth_components_vec(&self) -> Vec<GaussianState<T, N>> {
        let mut components = self.base_components.clone();

        for measurement in &self.adaptive_measurements {
            let (mean, covariance) = self.expander.expand_measurement(measurement);
            components.push(GaussianState::new(
                self.expander.birth_weight(),
                mean,
                covariance,
            ));
        }

        components
    }

    fn total_birth_mass(&self) -> T {
        let base_mass = self
            .base_components
            .iter()
            .fold(T::zero(), |acc, c| acc + c.weight);

        let adaptive_mass =
            T::from_usize(self.adaptive_measurements.len()).unwrap() * self.expander.birth_weight();

        base_mass + adaptive_mass
    }
}

#[cfg(feature = "alloc")]
impl<T, E, const N: usize, const M: usize> crate::filters::lmb::LabeledBirthModel<T, N>
    for MeasurementDrivenBirthModel<T, E, N, M>
where
    T: RealField + Float + Copy,
    E: MeasurementStateExpander<T, N, M>,
{
    fn birth_tracks(
        &self,
        label_gen: &mut crate::types::labels::LabelGenerator,
    ) -> Vec<crate::types::labels::BernoulliTrack<T, N>> {
        use crate::types::labels::BernoulliTrack;

        let mut tracks = Vec::new();

        // Add base components as birth tracks
        for component in &self.base_components {
            let label = label_gen.next_label();
            tracks.push(BernoulliTrack::new(
                label,
                self.expander.birth_existence(),
                component.clone(),
            ));
        }

        // Add adaptive components from measurements
        for measurement in &self.adaptive_measurements {
            let (mean, covariance) = self.expander.expand_measurement(measurement);
            let label = label_gen.next_label();
            let state = GaussianState::new(T::one(), mean, covariance);
            tracks.push(BernoulliTrack::new(
                label,
                self.expander.birth_existence(),
                state,
            ));
        }

        tracks
    }

    fn expected_birth_count(&self) -> T {
        let n_base = T::from_usize(self.base_components.len()).unwrap();
        let n_adaptive = T::from_usize(self.adaptive_measurements.len()).unwrap();
        (n_base + n_adaptive) * self.expander.birth_existence()
    }
}

// ============================================================================
// Type Aliases
// ============================================================================

/// Type alias for measurement-driven birth model with 2D constant velocity.
#[cfg(feature = "alloc")]
pub type MeasurementDrivenBirthModelCv2D<T> =
    MeasurementDrivenBirthModel<T, ConstantVelocity2DExpander<T>, 4, 2>;

/// Type alias for measurement-driven birth model with 3D constant velocity.
#[cfg(feature = "alloc")]
pub type MeasurementDrivenBirthModelCv3D<T> =
    MeasurementDrivenBirthModel<T, ConstantVelocity3DExpander<T>, 6, 3>;

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

        let components = birth.birth_components_vec();
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

        let components = birth.birth_components_vec();
        assert_eq!(components.len(), 9);
        assert!((birth.total_birth_mass() - 0.5).abs() < 1e-10);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_cv2d_expander() {
        let expander = ConstantVelocity2DExpander::new(10.0, 100.0, 0.05);

        let measurement = Measurement::from_array([15.0, 25.0]);
        let (mean, cov) = expander.expand_measurement(&measurement);

        // Check mean: [x, y, 0, 0]
        assert!((mean.index(0) - 15.0).abs() < 1e-10);
        assert!((mean.index(1) - 25.0).abs() < 1e-10);
        assert!(mean.index(2).abs() < 1e-10);
        assert!(mean.index(3).abs() < 1e-10);

        // Check covariance diagonal
        assert!((cov.as_matrix()[(0, 0)] - 10.0).abs() < 1e-10);
        assert!((cov.as_matrix()[(1, 1)] - 10.0).abs() < 1e-10);
        assert!((cov.as_matrix()[(2, 2)] - 100.0).abs() < 1e-10);
        assert!((cov.as_matrix()[(3, 3)] - 100.0).abs() < 1e-10);

        // Check birth weight
        assert!((expander.birth_weight() - 0.05).abs() < 1e-10);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_cv3d_expander() {
        let expander = ConstantVelocity3DExpander::new(5.0, 50.0, 0.03);

        let measurement = Measurement::from_array([1.0, 2.0, 3.0]);
        let (mean, cov) = expander.expand_measurement(&measurement);

        // Check mean: [x, y, z, 0, 0, 0]
        assert!((mean.index(0) - 1.0).abs() < 1e-10);
        assert!((mean.index(1) - 2.0).abs() < 1e-10);
        assert!((mean.index(2) - 3.0).abs() < 1e-10);
        assert!(mean.index(3).abs() < 1e-10);
        assert!(mean.index(4).abs() < 1e-10);
        assert!(mean.index(5).abs() < 1e-10);

        // Check covariance diagonal
        assert!((cov.as_matrix()[(0, 0)] - 5.0).abs() < 1e-10);
        assert!((cov.as_matrix()[(3, 3)] - 50.0).abs() < 1e-10);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_range_bearing_expander() {
        let expander = RangeBearingToCv2DExpander::new(1.0, 0.01, 100.0, 0.05);

        // Target at range 10, bearing 0 (along positive x-axis)
        let measurement = Measurement::from_array([10.0, 0.0]);
        let (mean, _cov) = expander.expand_measurement(&measurement);

        // Should be at (10, 0)
        assert!((mean.index(0) - 10.0).abs() < 1e-10);
        assert!(mean.index(1).abs() < 1e-10);
        assert!(mean.index(2).abs() < 1e-10);
        assert!(mean.index(3).abs() < 1e-10);

        // Target at range 10, bearing pi/2 (along positive y-axis)
        let measurement_y = Measurement::from_array([10.0, core::f64::consts::FRAC_PI_2]);
        let (mean_y, _) = expander.expand_measurement(&measurement_y);

        assert!(mean_y.index(0).abs() < 1e-10);
        assert!((mean_y.index(1) - 10.0).abs() < 1e-10);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_measurement_driven_birth_model() {
        let expander = ConstantVelocity2DExpander::new(10.0, 100.0, 0.05);
        let mut birth = MeasurementDrivenBirthModel::new(expander);

        // Initially empty
        assert_eq!(birth.birth_components_vec().len(), 0);
        assert!((birth.total_birth_mass() - 0.0).abs() < 1e-10);

        // Set measurements
        let measurements = vec![
            Measurement::from_array([10.0, 20.0]),
            Measurement::from_array([30.0, 40.0]),
        ];
        birth.set_measurements(measurements);

        // Should have 2 components now
        let components = birth.birth_components_vec();
        assert_eq!(components.len(), 2);
        assert!((birth.total_birth_mass() - 0.10).abs() < 1e-10);

        // Check first component position
        assert!((components[0].mean.index(0) - 10.0).abs() < 1e-10);
        assert!((components[0].mean.index(1) - 20.0).abs() < 1e-10);

        // Clear measurements
        birth.clear_measurements();
        assert_eq!(birth.birth_components_vec().len(), 0);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_measurement_driven_birth_with_base_components() {
        let expander = ConstantVelocity2DExpander::new(10.0, 100.0, 0.05);
        let mut birth = MeasurementDrivenBirthModel::new(expander);

        // Add base component
        let base_mean = StateVector::from_array([50.0, 50.0, 0.0, 0.0]);
        let base_cov = StateCovariance::identity();
        birth.add_base_component(GaussianState::new(0.02, base_mean, base_cov));

        // Set one measurement
        birth.set_measurements(vec![Measurement::from_array([10.0, 20.0])]);

        // Should have 2 components: 1 base + 1 adaptive
        let components = birth.birth_components_vec();
        assert_eq!(components.len(), 2);

        // Total mass: 0.02 (base) + 0.05 (adaptive)
        assert!((birth.total_birth_mass() - 0.07).abs() < 1e-10);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_measurement_driven_birth_labeled() {
        use crate::filters::lmb::LabeledBirthModel;
        use crate::types::labels::LabelGenerator;

        let expander = ConstantVelocity2DExpander::new(10.0, 100.0, 0.05);
        let mut birth = MeasurementDrivenBirthModel::new(expander);

        birth.set_measurements(vec![
            Measurement::from_array([10.0, 20.0]),
            Measurement::from_array([30.0, 40.0]),
        ]);

        let mut label_gen = LabelGenerator::new();
        let tracks = birth.birth_tracks(&mut label_gen);

        assert_eq!(tracks.len(), 2);
        assert_eq!(tracks[0].label.birth_time, 0);
        assert_eq!(tracks[0].label.index, 0);
        assert_eq!(tracks[1].label.birth_time, 0);
        assert_eq!(tracks[1].label.index, 1);

        // Check existence probability
        assert!((tracks[0].existence - 0.05).abs() < 1e-10);
    }
}
