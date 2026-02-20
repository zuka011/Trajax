# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Initial public release of faran
- Dual-backend support: NumPy (prototyping) and JAX (GPU acceleration)
- MPPI (Model Predictive Path Integral) trajectory planning
- MPCC (Model Predictive Contouring Control) for path-following
- Dynamical models: Kinematic Bicycle, Unicycle, Integrator
- Samplers: Gaussian, Halton-spline
- Cost functions: Contouring, Lag, Progress, Collision, Boundary, Control smoothing
- Trajectory representations: Waypoints (spline), Line
- Risk metrics integration: Expected value, Mean-variance, VaR, CVaR, Entropic risk
- Obstacle handling: Circle and polygon collision checking with motion prediction
- Comprehensive documentation with interactive examples
