//! Multi-target tracking filters
//!
//! Implementations of Kalman filter, EKF, UKF, PHD, LMB, and GLMB filters.
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
//! - [`lmb`]: Labeled Multi-Bernoulli filter
//! - [`glmb`]: Generalized Labeled Multi-Bernoulli filter

pub mod ekf;
pub mod glmb;
pub mod kalman;
pub mod lmb;
pub mod phd;
pub mod ukf;
