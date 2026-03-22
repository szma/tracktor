//! Multi-target tracking filters
//!
//! Implementations of Kalman filter, EKF, UKF, PHD, CPHD, LMB, and GLMB filters.
//!
//! # Single-Target Filters
//!
//! - [`kalman::KalmanFilter`]: Standard linear Kalman filter
//! - [`ekf::ExtendedKalmanFilter`]: Extended Kalman Filter for nonlinear systems
//! - [`ukf::UnscentedKalmanFilter`]: Unscented Kalman Filter for nonlinear systems
//!
//! # Multi-Target Filters
//!
//! - [`PhdFilterState`](phd::PhdFilterState): Gaussian Mixture PHD filter
//! - [`CphdFilterState`](cphd::CphdFilterState): Cardinalized PHD filter with improved cardinality estimation
//! - [`lmb`]: Labeled Multi-Bernoulli filter
//! - [`glmb`]: Generalized Labeled Multi-Bernoulli filter

#[cfg(feature = "alloc")]
pub mod cphd;
pub mod ekf;
pub mod glmb;
pub mod kalman;
pub mod lmb;
#[cfg(feature = "alloc")]
pub mod phd;
pub mod ukf;
