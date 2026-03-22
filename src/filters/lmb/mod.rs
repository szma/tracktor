//! Labeled Multi-Bernoulli (LMB) Filters
//!
//! Implementation of LMB and LMBM filters for multi-target tracking with
//! track identity preservation.
//!
//! # Filter Variants
//!
//! - [`LmbFilter`]: Single-sensor LMB with Gaussian mixture posteriors
//! - [`LmbmFilter`]: Single-sensor LMBM with multi-hypothesis tracking
//! - Multi-sensor variants: AA-LMB, GA-LMB, PU-LMB, IC-LMB
//!
//! # References
//!
//! - Reuter, S., et al. (2014). "The Labeled Multi-Bernoulli Filter"
//! - Vo, B.-T., & Vo, B.-N. (2013). "Labeled Random Finite Sets and
//!   Multi-Object Conjugate Priors"

pub mod cardinality;
#[cfg(feature = "alloc")]
pub mod filter;
#[cfg(feature = "alloc")]
pub mod fusion;
#[cfg(feature = "alloc")]
pub mod lmbm;
#[cfg(feature = "alloc")]
pub mod multisensor;
pub mod types;
#[cfg(feature = "alloc")]
pub mod updaters;

pub use cardinality::*;
#[cfg(feature = "alloc")]
pub use filter::*;
pub use types::*;
#[cfg(feature = "alloc")]
pub use updaters::*;

#[cfg(feature = "alloc")]
pub use fusion::*;
#[cfg(feature = "alloc")]
pub use lmbm::*;
#[cfg(feature = "alloc")]
pub use multisensor::*;
