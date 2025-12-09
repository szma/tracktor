# Tracktor

A type-safe Rust library for **Multi-Target Tracking (MTT)** using Random Finite Set (RFS) based algorithms.

## Why Tracktor?

### Compile-Time Correctness

Tracktor leverages Rust's type system to catch errors at compile time, not runtime:

```rust
// Vector spaces are type-distinct - you can't accidentally mix them
let state: StateVector<f64, 4> = /* ... */;
let measurement: Measurement<f64, 2> = /* ... */;

// This won't compile - type system prevents invalid operations
// let wrong = state + measurement;  // Error!

// Matrices encode their transformations
let H: ObservationMatrix<f64, 2, 4> = /* ... */;  // Maps 4D state → 2D measurement
let measurement = H.observe(&state);               // Correct by construction
```

### State Machine Enforced Filter Phases

The predict-update cycle is enforced at the type level:

```rust
// Filter states are typed - can't update before predict
let filter: PhdFilterState<f64, 4, Updated> = /* ... */;

let predicted = filter.predict(&model);  // Returns PhdFilterState<_, _, Predicted>
let updated = predicted.update(&obs, &measurements);  // Returns PhdFilterState<_, _, Updated>

// This won't compile - type system enforces correct ordering
// let wrong = filter.update(&obs, &measurements);  // Error! Can't update an Updated state
```

### Const Generic Dimensions

State and measurement dimensions are compile-time constants:

```rust
// 4D state (x, y, vx, vy), 2D measurements (x, y)
type State = StateVector<f64, 4>;
type Meas = Measurement<f64, 2>;

// Dimension mismatches are caught at compile time, not runtime
```

### Embedded-Ready

Full `no_std` support with optional `alloc` - deploy on resource-constrained platforms and real-time systems without compromise.

## Features

- **GM-PHD Filter**: Complete Gaussian Mixture Probability Hypothesis Density filter implementation based on Vo & Ma (2006)
- **Pluggable Models**: Trait-based transition, observation, clutter, and birth models
- **Numerical Stability**: Joseph form covariance updates with singular matrix detection
- **Mixture Management**: Intelligent pruning and merging to maintain tractable component counts
- **State Extraction**: Multiple strategies (threshold, top-N, local maxima) for target estimation
- **Assignment Solver**: Hungarian algorithm for optimal track-to-measurement association

## Quick Start

```rust
use tracktor::{
    filters::phd::PhdFilterState,
    models::{ConstantVelocity2D, PositionSensor2D, UniformClutter2D, FixedBirthModel},
    types::{GaussianMixture, GaussianState, Measurement, StateVector},
    utils::{ExtractionConfig, prune_and_merge},
};

// Define models
let dt = 1.0;
let transition = ConstantVelocity2D::new(dt, 0.95, 1.0);
let observation = PositionSensor2D::new(0.98, 10.0);
let clutter = UniformClutter2D::new(10.0, 0.0, 100.0, 0.0, 100.0);

// Initialize filter with known targets
let mut mixture = GaussianMixture::new();
mixture.push(GaussianState::new(
    0.8,
    StateVector::from_vec(vec![25.0, 25.0, 1.0, 0.5]),
    /* covariance */
));

let filter = PhdFilterState::new(mixture);

// Predict-update cycle
let predicted = filter.predict(&transition);
let measurements = vec![Measurement::from_vec(vec![26.1, 25.4])];
let updated = predicted.update(&observation, &clutter, &measurements);

// Prune, merge, and extract targets
let pruned = prune_and_merge(updated.mixture(), 1e-5, 4.0, 100);
let targets = ExtractionConfig::default().extract(&pruned);

println!("Expected targets: {:.2}", pruned.expected_target_count());
for target in targets {
    println!("Target at ({:.1}, {:.1}) with confidence {:.2}",
        target.state[0], target.state[1], target.confidence);
}
```

## Models

### Transition Models
- `ConstantVelocity2D` - Nearly constant velocity with white noise acceleration

### Observation Models
- `PositionSensor2D` - Observes position from position-velocity state
- `PositionSensor2DAsym` - Asymmetric noise in x/y directions

### Clutter Models
- `UniformClutter2D` - Uniform Poisson clutter in rectangular region

### Birth Models
- `FixedBirthModel` - Predefined birth locations with configurable weights

## State Extraction

Multiple strategies for extracting target estimates from the mixture:

```rust
let config = ExtractionConfig::default()
    .with_weight_threshold(0.5)
    .with_max_targets(10);

let targets = config.extract(&mixture);
```

## References

Based on the seminal work:
- B.-N. Vo and W.-K. Ma, "The Gaussian Mixture Probability Hypothesis Density Filter," *IEEE Transactions on Signal Processing*, vol. 54, no. 11, pp. 4091-4104, Nov. 2006.

## License

Licensed under either of Apache License, Version 2.0 or MIT license at your option.
