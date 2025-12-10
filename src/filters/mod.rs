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
//! - [`phd::PhdFilter`]: Gaussian Mixture PHD filter
//! - [`cphd::CphdFilter`]: Cardinalized PHD filter with improved cardinality estimation
//! - [`lmb`]: Labeled Multi-Bernoulli filter
//! - [`glmb`]: Generalized Labeled Multi-Bernoulli filter

pub mod cphd;
pub mod ekf;
pub mod glmb;
pub mod kalman;
pub mod lmb;
pub mod phd;
pub mod ukf;
