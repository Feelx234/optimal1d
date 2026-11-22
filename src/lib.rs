//! Shared core of `microagg1d` and `fast1dkmeans`.
//!
//! Both packages solve 1d clustering problems whose cost matrices are (totally)
//! monotone, which allows solving them with SMAWK based dynamic programs.

pub mod clustering;
pub mod concave;
pub mod cost;
pub mod kmeans;
pub mod microagg;
pub mod smawk;
pub mod stable;
