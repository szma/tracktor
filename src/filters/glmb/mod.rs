//! Generalized Labeled Multi-Bernoulli (GLMB) Filter
//!
//! Implementation of the delta-GLMB filter for multi-target tracking with
//! full joint hypothesis management.
//!
//! # Overview
//!
//! The GLMB filter maintains a mixture of hypotheses, each representing a
//! specific combination of:
//! - **Label set I**: Which tracks exist
//! - **Association history ξ**: Which measurements were associated to which tracks
//!
//! This enables representing correlations between track existences that cannot
//! be captured by the simpler LMB filter.
//!
//! # Comparison with LMB/LMBM
//!
//! | Feature | LMB | LMBM | GLMB |
//! |---------|-----|------|------|
//! | Track representation | Gaussian mixture | Single Gaussian | Single Gaussian |
//! | Hypothesis tracking | Implicit | Explicit | Explicit |
//! | Correlations | Independent | Partial | Full |
//! | Birth handling | Per-track | Per-hypothesis | Joint (integrated) |
//! | Computational cost | Low | Medium | Higher |
//!
//! # Type Safety
//!
//! Like other filters in tracktor, the GLMB filter uses phase markers to ensure
//! correct operation ordering at compile time:
//!
//! ```text
//! Updated  --predict()-->  Predicted  --update()-->  Updated
//! ```
//!
//! Attempting to call `update()` on an `Updated` state or `predict()` on a
//! `Predicted` state will result in a compile error.
//!
//! # Example
//!
//! ```ignore
//! use tracktor::prelude::*;
//! use tracktor::filters::glmb::*;
//!
//! // Create models
//! let transition = ConstantVelocity2D::new(1.0, 0.99);
//! let observation = PositionSensor2D::new(10.0, 0.9);
//! let clutter = UniformClutter2D::new(5.0, (0.0, 100.0), (0.0, 100.0));
//! let birth = MyLabeledBirthModel::new();
//!
//! // Create filter
//! let filter = GlmbFilter::new(transition, observation, clutter, birth, 10);
//! let mut state = filter.initial_state();
//!
//! // Process measurements
//! for measurements in measurement_sequences {
//!     let (new_state, stats) = filter.step(state, &measurements, dt);
//!     state = new_state;
//!
//!     // Extract estimates
//!     let estimates = extract_best_hypothesis(&state);
//!     for est in estimates {
//!         println!("Track {}: position = {:?}", est.label, est.state);
//!     }
//! }
//! ```
//!
//! # References
//!
//! - Vo, B.-T., & Vo, B.-N. (2013). "Labeled Random Finite Sets and
//!   Multi-Object Conjugate Priors"
//! - Vo, B.-T., Vo, B.-N., & Cantoni, A. (2009). "Analytic Implementations
//!   of the Cardinalized Probability Hypothesis Density Filter"
//! - Reuter, S., et al. (2014). "The Labeled Multi-Bernoulli Filter"

pub mod conversions;
pub mod filter;
pub mod types;

pub use conversions::*;
pub use filter::*;
pub use types::*;
