//! MTT-RS: Multi-Target Tracking Library for Rust
//!
//! A type-safe implementation of Random Finite Set (RFS) based tracking algorithms.
//!
//! # Features
//!
//! - **Type Safety**: Vector spaces and filter phases encoded in the type system
//! - **Compile-Time Checks**: Dimension mismatches caught at compile time
//! - **no_std Support**: Works in embedded environments

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;

pub mod types;
pub mod models;
pub mod filters;
pub mod assignment;
pub mod utils;

pub mod prelude {
    pub use crate::types::spaces::*;
    pub use crate::types::transforms::*;
    pub use crate::types::gaussian::*;
    pub use crate::types::labels::*;
    pub use crate::models::*;
    pub use crate::filters::phd::*;
    pub use crate::utils::*;
}

/// Error types for the library
#[derive(Debug, Clone, PartialEq)]
pub enum MttError {
    /// Matrix is singular and cannot be inverted
    SingularMatrix,
    /// Numerical computation became unstable
    NumericalInstability,
    /// Maximum number of components exceeded (for fixed-size storage)
    MaxComponentsExceeded,
    /// Assignment algorithm failed to find a solution
    AssignmentFailed,
}

#[cfg(feature = "std")]
impl std::error::Error for MttError {}

impl ::core::fmt::Display for MttError {
    fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
        match self {
            MttError::SingularMatrix => write!(f, "Matrix is singular"),
            MttError::NumericalInstability => write!(f, "Numerical instability detected"),
            MttError::MaxComponentsExceeded => write!(f, "Maximum components exceeded"),
            MttError::AssignmentFailed => write!(f, "Assignment algorithm failed"),
        }
    }
}

pub type Result<T> = ::core::result::Result<T, MttError>;
