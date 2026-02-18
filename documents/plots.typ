#import "@preview/plotsy-3d:0.2.1": plot-3d-surface

/// Visualizes the hinge loss collision cost function using circle approximations.
///
/// This function plots a 3D surface where the height at each point $(x, y)$
/// represents the collision cost that would be incurred if a circular obstacle
/// were centered at that position (in the local coordinate system of ego).
///
/// The cost for each obstacle is computed as:
/// $ J(d) = sum_i max(0, d_0 - d_i) $
/// where:
/// - $d_i$ is the minimum distance from the obstacle to the i-th circle in the ego approximation (negative if penetrating).
/// - $d_0$ is the safety threshold
///
/// -> Content
#let plot-hinge-loss(
  /// The centers of the circular obstacles. Each center is a tuple (x, y).
  /// -> Float or Array of Float
  centers: (),
  /// The radiuses of the ego circles. If one value is provided, it is applied to all circles.
  /// -> Float or Array of Float
  ego-radius: 0.0,
  /// The radius of the circular obstacle.
  /// -> Float
  obstacle-radius: 0.0,
  /// The safety threshold distance at which cost begins to be incurred.
  /// -> Float
  safety-threshold: 1.0,
  /// The range of x values to plot as (min, max).
  /// -> (Float, Float)
  x-range: (-5, 5),
  /// The range of y values to plot as (min, max).
  /// -> (Float, Float)
  y-range: (-5, 5),
  /// The spacing between axis tick labels as (x, y, z).
  /// -> (Float, Float, Float)
  axis-step: (2, 2, 2),
  /// The camera viewing position as (x, y, z).
  /// -> (Float, Float, Float)
  camera: (-2, 2, 5),
  /// The up direction for camera orientation as (x, y, z).
  /// -> (Float, Float, Float)
  up: (0, -1, 0),
  /// Overall scale factor for the plot.
  /// -> Float
  scale: 0.15,
  /// Vertical scale factor (relative to xy).
  /// -> Float
  z-scale: 0.25,
  /// The size of the axis labels.
  /// -> Float
  axis-label-size: 10pt,
  /// The number of subdivisions for the surface mesh (higher = smoother but slower).
  /// -> Integer
  subdivisions: 4,
  /// The base color for the surface.
  /// -> Color
  color-scale: gradient.linear(..color.map.viridis),
) = {
  let ego-radii = if type(ego-radius) == array {
    ego-radius
  } else {
    centers.map(_ => ego-radius)
  }

  let d(x, y, center, ego-radius) = {
    let dx = x - center.at(0)
    let dy = y - center.at(1)
    let center-distance = calc.sqrt(dx * dx + dy * dy)

    center-distance - ego-radius - obstacle-radius
  }

  let hinge-loss-single(x, y, center, ego-radius) = {
    let minimum-distance = d(x, y, center, ego-radius)
    calc.max(0, safety-threshold - minimum-distance)
  }

  let hinge-loss(x, y) = (
    centers
      .zip(ego-radii)
      .map(((center, ego-radius)) => hinge-loss-single(
        x,
        y,
        center,
        ego-radius,
      ))
      .sum()
  )

  let color-function(x, y, z, x-lo, x-hi, y-lo, y-hi, z-lo, z-hi) = {
    let range = z-hi - z-lo
    let t = if range != 0 { calc.clamp(z / range, 0, 1) } else { 0 }
    color-scale.sample(t * 100%)
  }

  v(-3em)

  plot-3d-surface(
    hinge-loss,
    color-func: color-function,
    subdivisions: subdivisions,
    subdivision-mode: "increase",
    scale-dim: (0.3 * scale, 0.3 * scale, z-scale * scale),
    xdomain: x-range,
    ydomain: y-range,
    axis-step: axis-step,
    axis-labels: ($x$, $y$, $sans(J)$),
    axis-label-size: axis-label-size,
    rotation-matrix: (camera, up),
  )
}
