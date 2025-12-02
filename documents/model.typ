#import "@preview/lovelace:0.3.0": pseudocode-list, indent
#import "examples.typ": mppi-example-diagram, kinematic-bicycle-diagram
#import "@local/roboter:0.2.11": algorithm, function, model, draw

#set math.equation(numbering: "(1)", supplement: "eq.")

= Model-Predictive Path Integral (MPPI) Control

== MPPI

#let rollout-count = 4
#let x-single = $sans(x)$
#let u-single = $sans(u)$
#let x = $bold(sans(x))$
#let u = $bold(sans(u))$
#let dynamics-(..args) = function(name: $f$, ..args)
#let dynamics = dynamics-(x, u)
#let cost-(..args) = function(name: $bold(sans(J))$, ..args)
#let cost = cost-(x, u)
#let distribution-(..args) = function(name: $bold(cal(Z))$, ..args)
#let distribution = distribution-()
#let K = $bold(K)$
#let T = $bold(T)$
#let N = $bold(N)$
#let lambda-t = $lambda$
#let perturbation = $bold(epsilon)$
#let w = $w$


The idea is to randomly try out many different variations of a "guess" control sequence and choose the one that works best. Visually that looks like this:

#figure(
  mppi-example-diagram(rollout-count: rollout-count),
  caption: [The MPPI planner computes #rollout-count different rollouts for the ego robot (blue) in the presence of another robot (red). Each rollout is colored based on its cost, with circular points with a lighter hue indicating higher cost. The green trajectory (triangles) represents the planner's initial guess, and the blue trajectory (squares) is the selected optimal path.],
)

The exact algorithm is as follows:

#algorithm(title: "MPPI Algorithm")[
  #pseudocode-list(hooks: .5em)[
    + *Given:*
      - Dynamical model #dynamics, cost function #cost, sampling distribution #distribution
      - Time horizon #T, Number of rollouts #K, Planning interval #N, Temperature #lambda-t
      - Initial state $#x-single _0$, Nominal control sequence $#u _0$

    + *Initialize:*
      - $#x-single arrow.l #x-single _0$
      - $#u arrow.l #u _0$

    + *while* task not completed *do*
      + *for* $k=0$ *to* $#K -1$ *do*
        + $#perturbation ^((k)) tilde #distribution$ #h(1fr) ➤ Sample control noise
        + $#u ^((k)) = #u + #perturbation ^((k))$ #h(1fr) ➤ Perturb nominal control sequence
        + $#x ^((k)) = "simulate"(#x-single, #u ^((k) ), #dynamics)$ #h(1fr) ➤ Simulate trajectory
        + $#cost-() ^((k)) = #cost-($#x ^((k))$, $#u ^((k))$)$ #h(1fr) ➤ Compute trajectory cost

      + $#cost-()_min arrow.l min_k #cost-()^((k))$ #h(1fr) ➤ Find minimum cost
      + $eta arrow.l sum_(k=0)^(#K -1) exp(-1 / #lambda-t (#cost-()^((k)) - #cost-()_min))$ #h(1fr) ➤ Compute normalizing constant

      + *for* $k=0$ *to* $#K -1$ *do*
        + $#w ^((k)) arrow.l 1 / eta exp(-1 / #lambda-t (#cost-()^((k)) - #cost-()_min))$ #h(1fr) ➤ Compute importance weights

      + $#u arrow.l sum_(k=0)^(#K -1) #w ^((k)) #u ^((k))$ #h(1fr) ➤ Update nominal control sequence
      + $#x-single arrow.l "execute"(#x-single, #u, #N)$ #h(1fr) ➤ Execute first #N control actions
      + $#u arrow.l {#u _(#N:#T -1), "pad"(#u) }$ #h(1fr) ➤ Shift control sequence
  ]
]

== Dynamical Model

A common choice for the dynamical model of a wheeled robot is the *kinematic bicycle model*. This model represents the robot as a bicycle with a single front and a single rear wheel. Other than simplifying the number of wheels, the model also assumes that there is no slip between the wheels and the ground. This assumption is not necessarily true at high speeds or during sharp turns, so we should keep that in mind. Here's a diagram for the model:

#figure(
  kinematic-bicycle-diagram(),
  caption: [The kinematic bicycle model. The robot state is defined by position $(x, y)$, velocity $v_r$, and heading $theta$. The control inputs are acceleration $a := dot(v)_r$ and steering angle $delta$. The wheelbase $L$ is the distance between the front and rear axles. $R$ denotes the turning radius for the current $delta$.],
)

The state $#x-single := [x quad y quad v quad theta]$ of the robot is represented by four variables:
- $x$: Position along the x-axis
- $y$: Position along the y-axis
- $v$: Velocity (speed) of the robot (corresponds to $v_r$ in the diagram)
- $theta$: Heading angle (orientation) in radians

The control inputs $#u-single := [a quad delta]$ are:
- $a$: Acceleration
- $delta$: Steering angle at the front wheel

#model(title: "Kinematic Bicycle Model")[
  The continuous-time dynamics of the kinematic bicycle model are given as:

  $
    dot(x) = v cos(theta), quad dot(y) = v sin(theta), quad dot(v) = a, quad dot(theta) = v / L tan(delta)
  $ <kinematic-bicycle-equations>

  where $L$ is the wheelbase (distance between the front and rear axles).
]

=== Euler Integration

#let delta-t = $Delta t$

To simulate the model, we discretize @kinematic-bicycle-equations with a time step size of #delta-t to get:

#align(
  center,
  block[
    #set align(left)
    $x_(t+1) = x_t + v_t cos(theta_t) dot #delta-t \
      y_(t+1) = y_t + v_t sin(theta_t) dot #delta-t \
      v_(t+1) = v_t + a_t dot #delta-t \
      theta_(t+1) = theta_t + v_t / L tan(delta_t) dot #delta-t$
  ],
)

Where $(dot)_t$ represents the value of $(dot)$ at time step $t$. A Euler integration step for this discretized model then looks like this:

#algorithm(title: "Euler Integration Step for the Kinematic Bicycle Model")[
  #pseudocode-list(hooks: .5em)[
    + *Given:*
      - Current state #x-single $= [x quad y quad v quad theta]$, control input #u-single $= [a quad delta]$
      - Time step #delta-t, Wheelbase $L$

    + *Compute next state:*
      + $x' arrow.l x + v dot cos(theta) dot #delta-t$ #h(1fr) ➤ Update x position
      + $y' arrow.l y + v dot sin(theta) dot #delta-t$ #h(1fr) ➤ Update y position
      + $v' arrow.l v + a dot #delta-t$ #h(1fr) ➤ Update velocity
      + $theta' arrow.l theta + (v / L) dot tan(delta) dot #delta-t$ #h(1fr) ➤ Update heading

    + *return* $#x-single ' = [x' quad y' quad v' quad theta']$
  ]
]
