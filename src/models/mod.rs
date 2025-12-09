//! Model traits for multi-target tracking
//!
//! This module defines the core traits that describe target dynamics,
//! sensor characteristics, clutter, and target birth processes.

mod transition;
mod observation;
mod clutter;
mod birth;

pub use transition::*;
pub use observation::*;
pub use clutter::*;
pub use birth::*;
