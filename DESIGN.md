# MTT-RS: Multi-Target Tracking Library for Rust

## Design Document v0.1

---

## 1. Motivation

### 1.1 Problem Statement

Multi-Target Tracking (MTT) is a critical component in robotics, autonomous vehicles, radar systems, and surveillance applications. While mature implementations exist in Python (Stone Soup), MATLAB, and C++, the Rust ecosystem lacks a comprehensive library for Random Finite Set (RFS) based tracking algorithms.

### 1.2 Why Rust?

- **Performance**: Zero-cost abstractions enable real-time tracking applications
- **Memory Safety**: Eliminates common bugs in complex probabilistic algorithms
- **Embedded Support**: `no_std` compatibility enables deployment on resource-constrained platforms
- **Type System**: Enables compile-time verification of mathematical correctness

### 1.3 Goals

1. Provide production-ready implementations of PHD, CPHD, LMB, and GLMB filters
2. Leverage Rust's type system to prevent common usage errors at compile time
3. Support `no_std` environments for embedded and real-time applications
4. Maintain high performance through const generics and zero-cost abstractions

---

## 2. Design Principles

### 2.1 Make Invalid States Unrepresentable

The library uses Rust's type system to enforce mathematical correctness. Operations that are mathematically undefined should not compile.

**Example**: A state vector cannot be added to a measurement vector because they belong to different vector spaces.

### 2.2 Zero-Cost Type Safety

All type-level distinctions use marker types and `#[repr(transparent)]` newtypes. The compiled binary is identical to an untyped implementation, but the source code is provably correct.

### 2.3 Compile-Time Dimensionality

State and measurement dimensions are encoded as const generic parameters. Dimension mismatches are caught at compile time, not runtime.

### 2.4 Explicit State Machines

Filter phases (predicted vs. updated) are encoded in the type system. The compiler enforces correct operation ordering.

---

## 3. Architecture Overview

```
mtt-rs/
├── core/
│   ├── spaces.rs       # Vector space markers and typed vectors
│   ├── transforms.rs   # Typed matrices for space transformations
│   ├── gaussian.rs     # Gaussian components and mixtures
│   └── labels.rs       # Track labels for GLMB/LMB
├── models/
│   ├── transition.rs   # Motion model traits
│   ├── observation.rs  # Sensor model traits
│   ├── clutter.rs      # Clutter model traits
│   └── birth.rs        # Birth model traits
├── filters/
│   ├── phd.rs          # GM-PHD filter
│   ├── cphd.rs         # GM-CPHD filter
│   ├── lmb.rs          # Labeled Multi-Bernoulli filter
│   └── glmb.rs         # Generalized Labeled Multi-Bernoulli filter
├── assignment/
│   ├── hungarian.rs    # Hungarian algorithm
│   └── murty.rs        # Murty's k-best assignments
└── utils/
    ├── pruning.rs      # Component pruning and merging
    └── extraction.rs   # State extraction strategies
```

---

## 4. Core Type System

### 4.1 Vector Space Markers

Empty marker types distinguish between mathematical spaces:

```rust
pub struct StateSpace;
pub struct MeasurementSpace;
pub struct InnovationSpace;
```

These markers have no runtime representation but enable compile-time discrimination.

### 4.2 Typed Vectors

Vectors are parameterized by their scalar type, dimension, and space:

```rust
#[repr(transparent)]
pub struct Vector<T: RealField, const N: usize, Space> {
    inner: SVector<T, N>,
    _marker: PhantomData<Space>,
}

pub type StateVector<T, const N: usize> = Vector<T, N, StateSpace>;
pub type Measurement<T, const M: usize> = Vector<T, M, MeasurementSpace>;
pub type Innovation<T, const M: usize> = Vector<T, M, InnovationSpace>;
```

**Invariants enforced by the type system**:
- Vectors in different spaces cannot be added or subtracted
- Innovation vectors can only be created by subtracting two measurements
- Dimension mismatches are compile-time errors

### 4.3 Typed Transformations

Matrices encode their source and target spaces:

```rust
#[repr(transparent)]
pub struct Transform<T: RealField, const ROWS: usize, const COLS: usize, To, From> {
    inner: SMatrix<T, ROWS, COLS>,
    _marker: PhantomData<(To, From)>,
}

pub type TransitionMatrix<T, const N: usize> = 
    Transform<T, N, N, StateSpace, StateSpace>;

pub type ObservationMatrix<T, const M: usize, const N: usize> = 
    Transform<T, M, N, MeasurementSpace, StateSpace>;

pub type KalmanGain<T, const N: usize, const M: usize> = 
    Transform<T, N, M, StateSpace, MeasurementSpace>;
```

**Enforced properties**:
- `ObservationMatrix * StateVector → Measurement` (not `StateVector`)
- `KalmanGain * Innovation → StateVector` (requires `Innovation`, not `Measurement`)
- `TransitionMatrix * StateVector → StateVector`

### 4.4 Typed Covariance Matrices

Covariance matrices are bound to their vector space:

```rust
#[repr(transparent)]
pub struct Covariance<T: RealField, const N: usize, Space> {
    inner: SMatrix<T, N, N>,
    _marker: PhantomData<Space>,
}

pub type StateCovariance<T, const N: usize> = Covariance<T, N, StateSpace>;
pub type MeasurementCovariance<T, const M: usize> = Covariance<T, M, MeasurementSpace>;
```

### 4.5 Filter Phase Encoding

The filter's operational phase is encoded at the type level:

```rust
pub struct Predicted;
pub struct Updated;

pub struct FilterState<T: RealField, const N: usize, Phase> {
    components: Vec<GaussianState<T, N>>,
    _phase: PhantomData<Phase>,
}
```

Method availability depends on the phase:

```rust
impl<T, const N: usize> FilterState<T, N, Updated> {
    pub fn predict(self, dt: T) -> FilterState<T, N, Predicted> { ... }
}

impl<T, const N: usize> FilterState<T, N, Predicted> {
    pub fn update(self, measurements: &[Measurement<T, M>]) -> FilterState<T, N, Updated> { ... }
}
```

**Enforced invariant**: `predict` and `update` must alternate. Calling `predict` twice without an intervening `update` is a compile-time error.

---

## 5. Model Traits

### 5.1 Transition Model

Describes target dynamics:

```rust
pub trait TransitionModel<T: RealField, const N: usize> {
    fn transition_matrix(&self, dt: T) -> TransitionMatrix<T, N>;
    fn process_noise(&self, dt: T) -> StateCovariance<T, N>;
    fn survival_probability(&self, state: &StateVector<T, N>) -> T;
}
```

### 5.2 Observation Model

Describes the sensor:

```rust
pub trait ObservationModel<T: RealField, const N: usize, const M: usize> {
    fn observation_matrix(&self) -> ObservationMatrix<T, M, N>;
    fn measurement_noise(&self) -> MeasurementCovariance<T, M>;
    fn detection_probability(&self, state: &StateVector<T, N>) -> T;
}
```

### 5.3 Clutter Model

Separated from the observation model for modularity:

```rust
pub trait ClutterModel<T: RealField, const M: usize> {
    fn clutter_rate(&self) -> T;
    fn clutter_density(&self, measurement: &Measurement<T, M>) -> T;
}
```

**Rationale for separation**: Clutter characteristics are often independent of the sensor model and may vary across different operational scenarios.

### 5.4 Birth Model

Describes spontaneous target appearance:

```rust
pub trait BirthModel<T: RealField, const N: usize> {
    type Components: IntoIterator<Item = GaussianState<T, N>>;
    
    fn birth_components(&self) -> Self::Components;
    fn total_birth_mass(&self) -> T;
}
```

---

## 6. Label System (GLMB/LMB)

### 6.1 Track Labels

Labels uniquely identify targets across time:

```rust
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Label {
    pub birth_time: u32,
    pub index: u32,
}
```

The tuple `(birth_time, index)` is guaranteed unique within a filter instance.

### 6.2 Bernoulli Tracks

For labeled filters, each track carries existence probability:

```rust
pub struct BernoulliTrack<T: RealField, const N: usize> {
    pub label: Label,
    pub existence: T,
    pub state: GaussianState<T, N>,
}
```

---

## 7. Assignment Problem

### 7.1 Cost Matrix

```rust
pub struct CostMatrix<T: RealField, const TRACKS: usize, const MEASUREMENTS: usize> {
    data: SMatrix<T, TRACKS, MEASUREMENTS>,
}
```

Const generic dimensions enable stack allocation for small problems.

### 7.2 Assignment Solver

```rust
pub struct Assignment<const TRACKS: usize> {
    pub mapping: [Option<usize>; TRACKS],
    pub cost: f64,
}

pub trait AssignmentSolver<const T: usize, const M: usize> {
    fn solve_optimal(cost: &CostMatrix<f64, T, M>) -> Assignment<T>;
    fn solve_k_best(cost: &CostMatrix<f64, T, M>, k: usize) -> Vec<Assignment<T>>;
}
```

### 7.3 Implementation Strategy

1. **Hungarian Algorithm**: O(n³) optimal assignment
2. **Murty's Algorithm**: Partitioning scheme for k-best assignments

Murty's algorithm is prioritized over Gibbs sampling for deterministic behavior in safety-critical applications.

---

## 8. Memory Management

### 8.1 no_std Compatibility

The library supports three tiers:

| Tier | Feature Flag | Allocator | Use Case |
|------|--------------|-----------|----------|
| 1 | `no_std` | None | Fixed-size filters, deeply embedded |
| 2 | `alloc` | Global | Dynamic components, no OS |
| 3 | `std` (default) | System | Full functionality |

### 8.2 Component Storage

For `no_std` without `alloc`, maximum component counts are const generic:

```rust
pub struct FixedGaussianMixture<T: RealField, const N: usize, const MAX: usize> {
    components: [MaybeUninit<GaussianState<T, N>>; MAX],
    len: usize,
}
```

---

## 9. Error Handling

### 9.1 Compile-Time Errors (Preferred)

The type system catches:
- Dimension mismatches
- Vector space violations
- Invalid operation sequences

### 9.2 Runtime Errors

Fallible operations return `Result`:

```rust
pub enum MttError {
    SingularMatrix,
    NumericalInstability,
    MaxComponentsExceeded,
    AssignmentFailed,
}
```

### 9.3 Panics

Panics are reserved for internal invariant violations (bugs), never for user input.

---

## 10. Example Usage

```rust
use mtt_rs::prelude::*;

// Define models
let transition = ConstantVelocity2D::new(process_noise);
let observation = PositionSensor2D::new(measurement_noise);
let clutter = UniformClutter::new(surveillance_region, clutter_rate);
let birth = AdaptiveBirth::new(birth_intensity);

// Create filter (type-safe!)
let filter = GmPhdFilter::new(transition, observation, clutter, birth);

// Process measurements
let predicted = filter.predict(dt);           // Returns FilterState<_, _, Predicted>
let updated = predicted.update(&measurements); // Returns FilterState<_, _, Updated>

// This would NOT compile:
// let wrong = predicted.predict(dt);  // predict() not available on Predicted
// let wrong = updated.update(&m);     // update() not available on Updated

// Extract states
let targets: Vec<(StateVector<f64, 4>, f64)> = updated.extract_states(threshold);
```

---

## 11. Future Extensions

### 11.1 Planned Features

- **Particle Filter variants**: SMC-PHD for highly non-linear systems
- **Extended/Unscented variants**: EK-PHD, UK-PHD for non-linear models
- **Adaptive birth**: Measurement-driven birth models
- **Track management**: Confirmed/tentative track logic

### 11.2 Potential Optimizations

- SIMD acceleration for Gaussian evaluations
- Parallel hypothesis evaluation for GLMB
- GPU offloading via `wgpu` (feature-gated)

---

## 12. References

1. Mahler, R. (2007). *Statistical Multisource-Multitarget Information Fusion*
2. Vo, B.-N., & Ma, W.-K. (2006). "The Gaussian Mixture Probability Hypothesis Density Filter"
3. Vo, B.-T., & Vo, B.-N. (2013). "Labeled Random Finite Sets and Multi-Object Conjugate Priors"
4. Reuter, S., et al. (2014). "The Labeled Multi-Bernoulli Filter"

---

## Appendix A: Type Safety Proof Sketches

### A.1 Vector Space Separation

**Claim**: `StateVector<T, N>` and `Measurement<T, M>` cannot be added.

**Proof**: The `Add` trait is implemented as:
```rust
impl<T, const N: usize, S> Add for Vector<T, N, S>
```

For `Add` to apply, both operands must have the same `S`. Since `StateSpace ≠ MeasurementSpace`, no implementation exists, and the compiler rejects the operation.

### A.2 Phase Ordering

**Claim**: `predict` cannot be called twice consecutively.

**Proof**: 
- `predict` consumes `FilterState<_, _, Updated>` and returns `FilterState<_, _, Predicted>`
- `predict` is only implemented for `FilterState<_, _, Updated>`
- After calling `predict`, the state is `Predicted`, which has no `predict` method

---

*Document version: 0.3*  
*Last updated: 2025-12*
