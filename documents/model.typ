#import "@preview/lovelace:0.3.0": pseudocode-list, indent
#import "examples.typ": mppi-example-diagram, kinematic-bicycle-diagram
#import "@local/roboter:0.2.13": function, algorithm, definition, model, draw

#set math.equation(numbering: "(1)", supplement: "eq.")

= Model-Predictive Path Integral (MPPI) Control

== MPPI

#let example-rollout-count = 4
#let input-single = $sans(u)$
#let state-single = $sans(x)$
#let input = $bold(sans(u))$
#let state = $bold(sans(x))$
#let dynamics-(..args) = function(name: $f$, ..args)
#let dynamics = dynamics-(input, state)
#let cost-(..args) = function(name: $bold(sans(J))$, ..args)
#let cost = cost-(input, state)
#let distribution-(..args) = function(name: $bold(cal(Z))$, ..args)
#let distribution = distribution-()
#let rollouts = $M$
#let rollout = $m$
#let horizon = $T$
#let planning-horizon = $T_("plan")$
#let temperature = $lambda$
#let weight = $w$


The idea is to randomly try out many different variations of a "guess" control sequence and choose the one that works best. Visually that looks like this:

#figure(
  mppi-example-diagram(rollout-count: example-rollout-count),
  caption: [The MPPI planner computes #example-rollout-count different rollouts for the ego robot (blue) in the presence of another robot (red). Each rollout is colored based on its cost, with circular points with a lighter hue indicating higher cost. The green trajectory (triangles) represents the planner's initial guess, and the blue trajectory (squares) is the selected optimal path.],
)

The exact algorithm is as follows:

#algorithm(title: "MPPI Algorithm")[
  #pseudocode-list(hooks: .5em)[
    + *Given:*
      - Dynamical model #dynamics, cost function #cost, sampling distribution #distribution
      - Time horizon #horizon, Number of rollouts #rollouts, Planning period #planning-horizon, Temperature #temperature
      - Nominal control sequence $#input _0$, Initial state $#state-single _0$

    + *Initialize:*
      - $#input arrow.l #input _0$
      - $#state-single arrow.l #state-single _0$

    + *while* task not completed *do*
      + *for* $#rollout=1$ *to* $#rollouts$ *do*
        + $#input _#rollout = "sample"(#input, #distribution)$ #h(1fr) ➤ Sample around nominal control sequence
        + $#state _#rollout = "simulate"(#input _#rollout, #state-single, #dynamics)$ #h(1fr) ➤ Simulate trajectory
        + $#cost-() _#rollout = #cost-($#input _#rollout$, $#state _#rollout$)$ #h(1fr) ➤ Compute trajectory cost
      + $#cost-()_min arrow.l limits(min) #cost-()_#rollout$ #h(1fr) ➤ Find minimum cost for $#rollout = 1, ..., #rollouts$
      + $eta arrow.l sum_(#rollout=1)^(#rollouts) exp(-1 / #temperature (#cost-()_#rollout - #cost-()_min))$ #h(1fr) ➤ Compute normalizing constant

      + *for* $#rollout=1$ *to* $#rollouts$ *do*
        + $#weight _#rollout arrow.l 1 / eta exp(-1 / #temperature (#cost-()_#rollout - #cost-()_min))$ #h(1fr) ➤ Compute importance weights

      + $#input _("opt.") arrow.l sum_(#rollout=1)^(#rollouts) #weight _#rollout #input _#rollout$ #h(1fr) ➤ Compute optimal control sequence
      + $#input arrow.l "update"(#input, #input _("opt."))$ #h(1fr) ➤ Update nominal control sequence
      + $#state-single arrow.l "execute"(#input _("opt."), #state-single, #planning-horizon)$ #h(1fr) ➤ Execute first #planning-horizon control actions
      + $#input arrow.l {#input _(#planning-horizon:#horizon -1), "pad"(#input) }$ #h(1fr) ➤ Shift control sequence
  ]

  $"sample"(dot), "update"(dot), "and" "pad"(dot)$ can vary based on the specific implementation.
]

== Dynamical Model

A common choice for the dynamical model of a wheeled robot is the *kinematic bicycle model*. This model represents the robot as a bicycle with a single front and a single rear wheel. Other than simplifying the number of wheels, the model also assumes that there is no slip between the wheels and the ground. This assumption is not necessarily true at high speeds or during sharp turns, so we should keep that in mind. Here's a diagram for the model:

#figure(
  kinematic-bicycle-diagram(),
  caption: [The kinematic bicycle model. The robot state is defined by position $(x, y)$, heading $theta$, and velocity $v_r$. The control inputs are acceleration $a := dot(v)_r$ and steering angle $delta$. The wheelbase $L$ is the distance between the front and rear axles. $R$ denotes the turning radius for the current $delta$.],
)

The control inputs $#input-single := [a quad delta]$ are:
- $a$: Acceleration
- $delta$: Steering angle at the front wheel

The state $#state-single := [x quad y quad theta quad v]$ of the robot is represented by four variables:
- $x$: Position along the x-axis
- $y$: Position along the y-axis
- $theta$: Heading angle (orientation) in radians
- $v$: Velocity (speed) of the robot (corresponds to $v_r$ in the diagram)

#model(title: "Kinematic Bicycle Model")[
  The continuous-time dynamics of the kinematic bicycle model are given as:

  $
    dot(x) = v cos(theta), quad dot(y) = v sin(theta), quad dot(theta) = v / L tan(delta), quad dot(v) = a
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
      theta_(t+1) = theta_t + v_t / L tan(delta_t) dot #delta-t \
      v_(t+1) = v_t + a_t dot #delta-t$
  ],
)

Where $(dot)_t$ represents the value of $(dot)$ at time step $t$. A Euler integration step for this discretized model then looks like this:

#algorithm(title: "Euler Integration Step for the Kinematic Bicycle Model")[
  #pseudocode-list(hooks: .5em)[
    + *Given:*
      - Control input #input-single $= [a quad delta]$, Current state #state-single $= [x quad y quad theta quad v]$
      - Time step #delta-t, Wheelbase $L$

    + *Compute next state:*
      + $x' arrow.l x + v dot cos(theta) dot #delta-t$ #h(1fr) ➤ Update x position
      + $y' arrow.l y + v dot sin(theta) dot #delta-t$ #h(1fr) ➤ Update y position
      + $theta' arrow.l theta + (v / L) dot tan(delta) dot #delta-t$ #h(1fr) ➤ Update heading
      + $v' arrow.l v + a dot #delta-t$ #h(1fr) ➤ Update velocity

      + *return* $#state-single ' = [x' quad y' quad theta' quad v']$
  ]
]

== Computational Framework

#let state-single-space = $cal(X)$
#let input-space = $bold(cal(U))$
#let state-space = $bold(cal(X))$
#let input-batch = $bold(sans(U))$
#let state-batch = $bold(sans(X))$
#let dynamics-batch-(..args) = function(name: $bold(sans(F))$, ..args)
#let dynamics-batch = dynamics-batch-(input-batch, state-single)
#let cost-batch-(..args) = function(name: $bold(cal(J))$, ..args)
#let cost-batch = cost-batch-(input-batch, state-batch)

Since MPPI requires similar computations to be performed many times for each planning step, a parallel implementation will be significantly more performant than a sequential one. For this reason, it is more convenient to think of *batches* of states and inputs. Here's the corresponding notation:

#definition(title: "Batch Dynamics")[
  Let $#input-batch = {#input _1, #input _2, ..., #input _#rollouts}$ be a set of control inputs representing #rollouts sampled rollouts. Then the *batch dynamics function* #dynamics-batch : $#input-space^#rollouts times #state-single-space -> #state-space^#rollouts$ is defined as:

  #align(
    center,
    $#dynamics-batch-(input-batch, state-single) = {#dynamics-($#input _1$, state-single), #dynamics-($#input _2$, state-single), ..., #dynamics-($#input _#rollouts$, state-single)} = {#state _1, #state _2, ..., #state _#rollouts} := #state-batch$,
  )

  The dimensions of the inputs and states are as follows:
  - $#input-batch in bb(R)^(#horizon times D_u times #rollouts)$, $D_u = 2$ is the dimension of the control input, and
  - $#state-batch in bb(R)^(#horizon times D_x times #rollouts)$, $D_x = 4$ is the dimension of the state.
]

Analogously, we can define a *batch cost function*:

#definition(title: "Batch Cost Function")[
  For a control input batch $#input-batch$ and corresponding state batch $#state-batch$, the *batch cost function* #cost-batch : $#input-space^#rollouts times #state-space^#rollouts -> bb(R)^#rollouts$ is defined as:

  #align(
    center,
    $#cost-batch-(input-batch, state-batch) = {#cost-($#input _1$, $#state _1$), #cost-($#input _2$, $#state _2$), ..., #cost-($#input _#rollouts$, $#state _#rollouts$)} = {#cost-() _1, #cost-() _2, ..., #cost-() _#rollouts} := #cost-batch-()$,
  )
]
