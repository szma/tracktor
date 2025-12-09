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
pub mod filter;
pub mod fusion;
pub mod lmbm;
pub mod multisensor;
#[cfg(test)]
mod tests;
pub mod types;
pub mod updaters;

pub use cardinality::*;
pub use filter::*;
pub use types::*;
pub use updaters::*;

#[cfg(feature = "alloc")]
pub use fusion::*;
#[cfg(feature = "alloc")]
pub use lmbm::*;
#[cfg(feature = "alloc")]
pub use multisensor::*;
