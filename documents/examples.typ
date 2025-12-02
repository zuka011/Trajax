#import "@preview/suiji:0.4.0": gen-rng-f, normal-f
#import "@local/roboter:0.2.13": (
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
      curves.bezier(ego.position, (12, -2), (7, 0), (8, 1)),
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
