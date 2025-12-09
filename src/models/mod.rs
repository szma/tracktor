//! Model traits for multi-target tracking
//!
//! This module defines the core traits that describe target dynamics,
//! sensor characteristics, clutter, and target birth processes.

mod birth;
mod clutter;
mod observation;
mod transition;

pub use birth::*;
pub use clutter::*;
pub use observation::*;
pub use transition::*;
