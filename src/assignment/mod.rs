//! Assignment problem solvers
//!
//! Algorithms for optimal assignment between tracks and measurements.

pub mod hungarian;
pub mod traits;

pub use hungarian::*;
pub use traits::*;
