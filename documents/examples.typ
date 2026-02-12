#import "@preview/zero:0.5.0": num
#import "@preview/suiji:0.4.0": gen-rng-f, normal-f
#import "@preview/cetz:0.4.2"
#import "@local/roboter:0.3.9": (
  draw,
  curves,
  zero-inputs,
  rollout-trajectory,
  rollout-trajectories,
  sample-inputs-for,
  optimal-trajectory-for,
  visualize-robot,
  visualize-rollout,
  visualize-rollouts,
  with-visualizers,
  kinematic-bicycle,
  bicycle-theme,
  kinematic-unicycle,
  unicycle-theme,
  sat,
  vehicle-theme,
  vehicle-approximation,
)

#let grid-color = gray.transparentize(80%)
#let color-map = gradient.linear(
  ..color.map.plasma.map(it => it.transparentize(40%)),
)

#let squared-distance-cost(
  ego-position: (0, 0),
  other-position: (1, 1),
  threshold: 2,
) = calc.exp(
  -(
    calc.pow(ego-position.at(0) - other-position.at(0), 2)
      + calc.pow(ego-position.at(1) - other-position.at(1), 2)
  )
    / calc.pow(threshold, 2),
)

#let other-robot = (
  index: 2,
  position: (6, 0),
  rotation: 10deg,
  color: red,
  visualizers: (),
  time-step: 0.05,
  wheelbase: 1.25,
)

#let ego-robot = (
  index: 1,
  position: (1, 2),
  rotation: -15deg,
  color: blue,
  visualizers: (),
  time-step: 0.04,
  wheelbase: 1.25,
  get-initial-state: self => (
    x: self.position.at(0),
    y: self.position.at(1),
    v: 15,
    theta: self.rotation.rad(),
  ),
  cost-function: state => (
    squared-distance-cost(
      ego-position: (state.x, state.y),
      other-position: other-robot.position,
    )
  ),
  fully-visualized: (
    self,
    guess: (),
    optimal: (),
    rollouts: (),
  ) => with-visualizers(
    self,
    visualizers: (
      visualize-rollout(
        guess,
        color: (_, _) => color.green.lighten(25%),
        line-color: green,
        marker-type: "triangle",
        dash: "solid",
      ),
      visualize-rollout(
        optimal,
        color: (_, _) => color.blue.lighten(25%),
        marker-type: "square",
        dash: "solid",
      ),
      visualize-rollouts(
        rollouts,
        cost-function: self.cost-function,
        color-map: color-map,
        line-color: gray.transparentize(50%),
        dash: "dotted",
      ),
    ),
  ),
)

#let tracking-ego-robot = (..ego-robot, rotation: 30deg)

#let mppi-example-diagram(
  rollout-count: 12,
  horizon: 12,
  ego: ego-robot,
) = {
  let initial-state = (ego.get-initial-state)(ego)
  let guess-inputs = (
    acceleration: range(horizon).map(it => 5),
    steering: range(horizon).map(it => 0.02 - 0.01 * it),
  )
  let guess = rollout-trajectory(
    ego,
    initial-state: initial-state,
    inputs: guess-inputs,
  )
  let results = optimal-trajectory-for(
    ego,
    initial-state: initial-state,
    inputs: sample-inputs-for(
      ego,
      horizon: horizon,
      sample-count: rollout-count,
      means: guess-inputs,
      standard-deviations: (acceleration: 0.5, steering: 0.25),
    ),
  )

  draw.diagram({
    draw.grid((0, -2), (14, 3), color: grid-color)
    draw.label((12, -0), [*$<-$ Desired Route*])
    draw.trajectory(
      curves.bezier(start: ego.position, end: (12, -2), c1: (7, 0), c2: (8, 1)),
      color: red.transparentize(50%),
    )

    visualize-robot(other-robot)
    visualize-robot(
      (ego.fully-visualized)(
        ego,
        guess: guess,
        optimal: results.trajectory,
        rollouts: results.rollouts,
      ),
    )
  })
}

#let kinematic-bicycle-diagram() = {
  draw.diagram({
    draw.grid((-3, -3), (11, 8), color: grid-color)
    kinematic-bicycle(
      rear-position: (5, 0),
      chassis-width: 3.5,
      heading: 45deg,
      theme: bicycle-theme(angle: red, velocity: blue),
    )
  })
}

#let kinematic-unicycle-diagram() = {
  draw.diagram({
    draw.grid((0, 0), (8, 4), color: grid-color)
    kinematic-unicycle(
      position: (4, 2),
      heading: 45deg,
      theme: unicycle-theme(angle: red, velocity: blue),
    )
  })
}

#let nominal-point-marker(
  path-parameter: 0.0,
  reference-point: (0, 0),
  reference-heading: 0deg,
  reference-line-length: 4.0,
  reference-color: black,
  nominal-point-color: green.darken(25%),
) = {
  let (reference-x, reference-y) = reference-point

  let reference-line-start = (
    reference-x - reference-line-length * calc.cos(reference-heading),
    reference-y - reference-line-length * calc.sin(reference-heading),
  )
  let reference-line-end = (
    reference-x + reference-line-length * calc.cos(reference-heading),
    reference-y + reference-line-length * calc.sin(reference-heading),
  )

  draw.line(
    reference-line-start,
    reference-line-end,
    color: reference-color.transparentize(90%),
  )
  draw.angle-arc(
    reference-point,
    start: 0deg,
    stop: reference-heading,
    radius: 3.0,
    reference-length: 3.5,
    color: nominal-point-color,
    label-content: $theta_phi$,
    label-radius-offset: 0.35,
    show-reference-lines: true,
    position-is-center: true,
  )
  draw.markers(
    (reference-point,),
    color: nominal-point-color,
    size: 0.25,
    marker-type: "o",
  )
  draw.label(
    reference-point,
    text(fill: nominal-point-color, $(x_phi, y_phi)$),
    offset: (-0.1, 0.5),
  )
  draw.label(
    reference-point,
    text(fill: nominal-point-color, $phi = #path-parameter$),
    offset: (0.0, -0.5),
  )
}

#let tracking-errors(
  ego: tracking-ego-robot,
  reference-point: (0, 0),
  reference-heading: 0deg,
  contour-color: orange.darken(10%),
  lag-color: blue.darken(25%),
  info-position: (0, 5),
) = {
  let (robot-x, robot-y) = ego.position
  let (reference-x, reference-y) = reference-point
  let delta-x = robot-x - reference-x
  let delta-y = robot-y - reference-y

  let contouring-error = (
    calc.sin(reference-heading) * delta-x
      - calc.cos(reference-heading) * delta-y
  )
  let lag-error = (
    -calc.cos(reference-heading) * delta-x
      - calc.sin(reference-heading) * delta-y
  )

  let contouring-endpoint = (
    robot-x - contouring-error * calc.sin(reference-heading),
    robot-y + contouring-error * calc.cos(reference-heading),
  )

  draw.line(
    ego.position,
    contouring-endpoint,
    color: contour-color.transparentize(25%),
    dash: "solid",
    label-content: text(fill: contour-color, $e_c$),
    label-offset: (-0.3, 0),
  )

  draw.line(
    contouring-endpoint,
    reference-point,
    color: lag-color.transparentize(25%),
    dash: "solid",
    label-content: text(fill: lag-color, $e_l$),
    label-offset: (0, 0.3),
  )

  draw.markers(
    (contouring-endpoint,),
    size: 0.1,
    color: black,
  )

  draw.label(
    info-position,
    [
      #set text(size: 8pt)
      $#text(fill: contour-color, $e_c = #num(contouring-error, digits: 2)$) \
        #text(fill: lag-color, $e_l = #num(lag-error, digits: 2)$)$
    ],
    offset: (0, 0),
  )
}

#let path-parameter-markers(
  path-parameters: (0.0,),
  curve: (
    start: (0, 1),
    end: (9, 3),
    c1: (2, 5),
    c2: (6, 5),
  ),
  color: red.darken(25%),
) = {
  for path-parameter in path-parameters {
    let point = curves.cubic-bezier-at(path-parameter, ..curve)

    draw.markers(
      (point,),
      color: color,
      size: 0.2,
    )
    draw.label(
      point,
      text(size: 8pt, fill: color, $phi = #path-parameter$),
      offset: (0, -0.4),
    )
  }
}

#let tracking-error-diagram(
  ego: tracking-ego-robot,
  curve: (
    start: (0, 1),
    end: (9, 3),
    c1: (2, 5),
    c2: (6, 5),
  ),
  info-position: (0.5, 5),
  path-parameter: 0.45,
  path-parameter-examples: (0, 0.3, 0.9),
  extend-reference-line-by: 4.0,
  contour-color: orange.darken(10%),
  lag-color: blue.darken(25%),
  nominal-point-color: green.darken(25%),
  reference-color: black,
  path-parameter-marker-color: red.darken(25%),
) = {
  let reference-point = curves.cubic-bezier-at(path-parameter, ..curve)
  let reference-heading = curves.cubic-bezier-angle-at(path-parameter, ..curve)
  let path-points = curves.bezier(..curve, samples: 50)

  draw.diagram({
    draw.grid((-0.5, 0), (10, 6), color: grid-color)

    draw.trajectory(path-points, color: reference-color, dash: "solid")
    visualize-robot(ego)

    nominal-point-marker(
      path-parameter: path-parameter,
      reference-point: reference-point,
      reference-heading: reference-heading,
      reference-line-length: extend-reference-line-by,
      reference-color: reference-color,
      nominal-point-color: nominal-point-color,
    )

    tracking-errors(
      ego: ego,
      reference-point: reference-point,
      reference-heading: reference-heading,
      contour-color: contour-color,
      lag-color: lag-color,
      info-position: info-position,
    )

    path-parameter-markers(
      path-parameters: path-parameter-examples,
      curve: curve,
      color: path-parameter-marker-color,
    )
  })
}

#let sat-diagram() = {
  draw.diagram({
    draw.grid((-5, -4), (10, 4), color: grid-color)
    sat(
      rectangles: (
        (length: 3.0, width: 1.5, offset: (0, 1), rotation: -30deg, color: red),
        (
          length: 3.0,
          width: 1.5,
          offset: (6, 2),
          rotation: -15deg,
          color: blue,
        ),
      ),
      axes: (
        (
          origin: (0, -2),
          angle: -15deg,
          extents: (7.0, -5.0),
          label-offset: (0.25, 0.25),
          overlap-offset: (-0.25, -0.25),
        ),
        (
          origin: (9, 0),
          angle: 75deg,
          extents: (3.0, -4.1),
          label-offset: (0.5, 0.25),
          overlap-offset: (-0.75, 0.25),
        ),
      ),
    )
  })
}

#let circle-approximation-diagram(
  circle-count: 1,
  radius: none,
  safety-threshold: 2.0,
  show-safety-threshold-separately: true,
  theme: vehicle-theme(),
) = {
  draw.diagram({
    draw.grid((-3.75, -3), (3.75, 3), color: grid-color, step: 0.75)
    draw.transformed-group(
      vehicle-approximation(
        length: 5.0,
        width: 2.0,
        radius: radius,
        circle-count: circle-count,
        safety-threshold: safety-threshold,
        show-safety-threshold-separately: show-safety-threshold-separately,
        theme: theme,
      ),
      scale: 0.7,
    )
  })
}
