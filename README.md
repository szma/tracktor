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
let H: ObservationMatrix<f64, 2, 4> = /* ... */;  // Maps 4D state -> 2D measurement
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

### Multi-Target Filters

| Filter | Track Identity | Multi-Hypothesis | Multi-Sensor | Description |
|--------|---------------|------------------|--------------|-------------|
| **GM-PHD** | No | No | No | Gaussian Mixture PHD - fast, scalable |
| **LMB** | Yes | Implicit | Yes | Labeled Multi-Bernoulli with track continuity |
| **LMBM** | Yes | Explicit | Yes | LMB Mixture for ambiguous scenarios |
| **GLMB** | Yes | Explicit | Yes | Full joint hypothesis, most accurate |

### Single-Target Filters

| Filter | Dynamics | Jacobian Required | Description |
|--------|----------|-------------------|-------------|
| **Kalman** | Linear | No | Standard discrete-time Kalman filter |
| **EKF** | Nonlinear | Yes | Extended Kalman with Jacobian linearization |
| **UKF** | Nonlinear | No | Unscented transform, no Jacobians needed |

### Core Capabilities

- **Pluggable Models**: Trait-based transition, observation, clutter, and birth models
- **Numerical Stability**: Joseph form covariance updates with singular matrix detection
- **Mixture Management**: Intelligent pruning and merging to maintain tractable component counts
- **State Extraction**: Multiple strategies (threshold, top-N, expected count, local maxima)
- **Data Association**: Loopy Belief Propagation (LBP) and Hungarian algorithm
- **Multi-Sensor Fusion**: AA-LMB, GA-LMB, PU-LMB, IC-LMB fusion strategies
- **Embedded-Ready**: Full `no_std` support with optional `alloc`

## Quick Start

```rust
use tracktor::prelude::*;

fn main() {
    // Define models
    let dt = 1.0;
    let transition = ConstantVelocity2D::new(1.0, 0.95);  // (noise_diff_coeff, p_survival)
    let observation = PositionSensor2D::new(10.0, 0.98);  // (noise_variance, p_detection)
    let clutter = UniformClutter2D::new(10.0, (0.0, 100.0), (0.0, 100.0));
    let birth = FixedBirthModel::<f64, 4>::new();  // Empty birth model

    // Initialize filter with known targets
    let mut mixture = GaussianMixture::new();
    mixture.push(GaussianState::new(
        0.8,
        StateVector::from_array([25.0, 25.0, 1.0, 0.5]),
        StateCovariance::identity().scale(10.0),
    ));

    let filter = PhdFilterState::from_mixture(mixture);

    // Predict-update cycle
    let predicted = filter.predict(&transition, &birth, dt);
    let measurements = [Measurement::from_array([26.1, 25.4])];
    let updated = predicted.update(&measurements, &observation, &clutter);

    // Prune, merge, and extract targets
    let config = PruningConfig::default_config();
    let pruned = prune_and_merge(&updated.mixture, &config);
    let targets = extract_targets(&pruned, &ExtractionConfig::weight_threshold(0.5));

    println!("Expected targets: {:.2}", pruned.total_weight());
    for target in targets {
        println!("Target at ({:.1}, {:.1}) with confidence {:.2}",
            *target.state.index(0), *target.state.index(1), target.confidence);
    }
}
```

## Models

### Transition Models

| Model | State Dim | Description |
|-------|-----------|-------------|
| `ConstantVelocity2D` | 4 | [x, y, vx, vy] with white noise acceleration |
| `ConstantVelocity3D` | 6 | [x, y, z, vx, vy, vz] for 3D tracking |
| `CoordinatedTurn2D` | 5 | [x, y, vx, vy, omega] nonlinear turn model (use with EKF/UKF) |

### Observation Models

| Model | State -> Meas | Description |
|-------|---------------|-------------|
| `PositionSensor2D` | 4D -> 2D | Observes [x, y] from position-velocity state |
| `PositionSensor2DAsym` | 4D -> 2D | Asymmetric noise in x/y directions |
| `PositionSensor3D` | 6D -> 3D | Observes [x, y, z] from 6D state |
| `RangeBearingSensor` | 4D -> 2D | Nonlinear range-bearing (use with EKF/UKF) |
| `RangeBearingSensor5D` | 5D -> 2D | Range-bearing for coordinated turn model |

### Clutter Models

| Model | Description |
|-------|-------------|
| `UniformClutter` | Generic uniform Poisson clutter over rectangular region |
| `UniformClutter2D` | 2D rectangular surveillance region |
| `UniformClutter3D` | 3D rectangular surveillance region |
| `UniformClutterRangeBearing` | Polar coordinates (range-bearing space) |
| `GaussianClutter` | Gaussian-shaped clutter density |

### Birth Models

| Model | Description |
|-------|-------------|
| `FixedBirthModel` | Predefined birth locations with configurable weights |
| `UniformBirthModel2D` | Grid of birth components over rectangular region |
| `AdaptiveBirthModel` | Creates birth components from unassociated measurements |
| `NoBirthModel` | No spontaneous births (for labeled filters) |

## Multi-Sensor Fusion

The LMB filter supports multiple fusion strategies for combining tracks from different sensors:

| Strategy | Method | Best For |
|----------|--------|----------|
| **AA-LMB** | Arithmetic Average | Fast, simple fusion |
| **GA-LMB** | Geometric Average (Covariance Intersection) | Correlated sensor noise |
| **PU-LMB** | Parallel Update (Information Filter) | Independent sensors |
| **IC-LMB** | Iterated Corrector | Maximum accuracy |

```rust
use tracktor::filters::lmb::{MultisensorLmbFilter, SensorConfig};
use tracktor::filters::lmb::fusion::GeometricAverageMerger;

let merger = GeometricAverageMerger::default();
let filter = MultisensorLmbFilter::new(transition, vec![sensor1, sensor2], merger);
```

## State Extraction

Multiple strategies for extracting target estimates from the mixture:

- **Weight Threshold**: Extract components exceeding a weight threshold
- **Top-N**: Extract N highest-weighted components
- **Expected Count**: Extract based on rounded total weight
- **Local Maxima**: Extract local maxima with Mahalanobis distance-based suppression

```rust
let config = ExtractionConfig::default()
    .with_weight_threshold(0.5)
    .with_max_targets(10);

let targets = config.extract(&mixture);
```

## Examples

The `examples/` directory contains complete working examples:

| Example | Description |
|---------|-------------|
| `basic_tracking.rs` | Simple PHD filter introduction |
| `advanced_tracking.rs` | Vo & Ma benchmark scenario |
| `lmb_tracking.rs` | LMB filter with track identity |
| `lmbm_tracking.rs` | Multi-hypothesis tracking |
| `glmb_tracking.rs` | Full GLMB with hypothesis extraction |
| `lmb_multisensor.rs` | Multi-sensor fusion comparison |
| `measurement_driven_birth.rs` | Adaptive birth from measurements |

Run examples with:
```bash
cargo run --example basic_tracking
```

## References

### Multi-Target Filters

- **GM-PHD Filter**: Vo, B.-N., & Ma, W.-K. (2006). "The Gaussian Mixture Probability Hypothesis Density Filter." *IEEE Transactions on Signal Processing*, 54(11), 4091-4104.

- **LMB/GLMB Filters**: Vo, B.-T., & Vo, B.-N. (2013). "Labeled Random Finite Sets and Multi-Object Conjugate Priors." *IEEE Transactions on Signal Processing*, 61(13), 3460-3475.

- **LMB Filter**: Reuter, S., Vo, B.-T., Vo, B.-N., & Dietmayer, K. (2014). "The Labeled Multi-Bernoulli Filter." *IEEE Transactions on Signal Processing*, 62(12), 3246-3260.

### Single-Target Filters

- **Kalman Filter**: Kalman, R. E. (1960). "A New Approach to Linear Filtering and Prediction Problems." *Journal of Basic Engineering*, 82(1), 35-45.

- **Extended Kalman Filter**: Smith, G. L., Schmidt, S. F., & McGee, L. A. (1962). "Application of Statistical Filter Theory to the Optimal Estimation of Position and Velocity on Board a Circumlunar Vehicle." NASA Technical Report TR R-135.

- **Unscented Kalman Filter**: Julier, S. J., & Uhlmann, J. K. (1997). "A New Extension of the Kalman Filter to Nonlinear Systems." *Proc. SPIE 3068, Signal Processing, Sensor Fusion, and Target Recognition VI*.

## License

Licensed under either of AGPL-3.0 or commercial license (contact me).
