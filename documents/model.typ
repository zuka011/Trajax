#import "@preview/lovelace:0.3.0": pseudocode-list, indent
#import "examples.typ": (
  mppi-example-diagram,
  kinematic-bicycle-diagram,
  tracking-error-diagram,
)
#import "@local/roboter:0.3.1": function, algorithm, definition, model, draw

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
#let planning-horizon = $P$
#let temperature = $lambda$
#let weight = $w$


The idea is to randomly try out many different variations of a "guess" control sequence and choose the one that works best. Visually that looks like this:

#figure(
  mppi-example-diagram(rollout-count: example-rollout-count),
  caption: [The MPPI planner computes #example-rollout-count different rollouts for the ego robot (blue) in the presence of another robot (red). Each rollout is colored based on its cost, with circular points with a lighter hue indicating higher cost. The green trajectory (triangles) represents the planner's initial guess, and the blue trajectory (squares) is the selected optimal path.],
)

The exact algorithm is as follows:

#algorithm(title: [MPPI Algorithm @Williams2015 @Williams2016])[
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

      + $#input _("opt.") arrow.l "filter"(sum_(#rollout=1)^(#rollouts) #weight _#rollout #input _#rollout)$ #h(1fr) ➤ Compute optimal control sequence
      + $#input arrow.l "update"(#input, #input _("opt."))$ #h(1fr) ➤ Update nominal control sequence
      + $#state-single arrow.l "execute"(#input _("opt."), #state-single, #planning-horizon)$ #h(1fr) ➤ Execute first #planning-horizon control actions
      + $#input arrow.l {#input _(#planning-horizon:#horizon -1), "pad"(#input) }$ #h(1fr) ➤ Shift control sequence
  ]

  $"sample"(dot), "filter"(dot), "update"(dot), "and" "pad"(dot)$ can vary based on the specific implementation.
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

#model(title: [Kinematic Bicycle Model @Polack2017])[
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
#let state-dimension = $D_x$
#let control-dimension = $D_u$

Since MPPI requires similar computations to be performed many times for each planning step, a parallel implementation will be significantly more performant than a sequential one. For this reason, it is more convenient to think of *batches* of states and inputs. Here's the corresponding notation:

#definition(title: "Batch Dynamics")[
  Let $#input-batch = {#input _1, #input _2, ..., #input _#rollouts}$ be a set of control inputs representing #rollouts sampled rollouts. Then the *batch dynamics function* #dynamics-batch : $#input-space^#rollouts times #state-single-space -> #state-space^#rollouts$ is defined as:

  #align(
    center,
    $#dynamics-batch-(input-batch, state-single) = {#dynamics-($#input _1$, state-single), #dynamics-($#input _2$, state-single), ..., #dynamics-($#input _#rollouts$, state-single)} = {#state _1, #state _2, ..., #state _#rollouts} := #state-batch$,
  )

  The dimensions of the inputs and states are as follows:
  - $#input-batch in bb(R)^(#horizon times #control-dimension times #rollouts)$, $#control-dimension = 2$ is the dimension of the control input, and
  - $#state-batch in bb(R)^(#horizon times #state-dimension times #rollouts)$, $#state-dimension = 4$ is the dimension of the state.
]

Analogously, we can define a *batch cost function*:

#definition(title: "Batch Cost Function")[
  For a control input batch $#input-batch$ and corresponding state batch $#state-batch$, the *batch cost function* #cost-batch : $#input-space^#rollouts times #state-space^#rollouts -> bb(R)^#rollouts$ is defined as:

  #align(
    center,
    $#cost-batch-(input-batch, state-batch) = {#cost-($#input _1$, $#state _1$), #cost-($#input _2$, $#state _2$), ..., #cost-($#input _#rollouts$, $#state _#rollouts$)} = {#cost-() _1, #cost-() _2, ..., #cost-() _#rollouts} := #cost-batch-()$,
  )
]

== Cost Functions

When using MPPI to control the motion of a mobile robot, the terms of the cost function #cost can be split into the following categories:
- *Tracking:* Costs that encourage the robot to follow a desired trajectory or reach a specific goal.
- *Safety:* Costs that discourage the robot from getting too close to obstacles, or unsafe areas.
- *Comfort:* Costs that promote smooth and comfortable motion.

=== Tracking Cost

// TODO: Path Parameter is actually called Arc Length.
#let path-parameter = $phi$
#let path-parameter-change = $beta$
#let reference-trajectory = $#state _("ref")$
#let reference-point = $#state-single _("ref")$
#let reference-x = $x_("ref")$
#let reference-y = $y_("ref")$
#let reference-theta = $theta_("ref")$
#let x-at-arc-length = $x_(#path-parameter)$
#let y-at-arc-length = $y_(#path-parameter)$
#let theta-at-arc-length = $theta_(#path-parameter)$
#let augmented-state-single = $#state-single _phi$
#let augmented-input-single = $#input-single _phi$
#let contouring-cost = $#cost-()_c$
#let lag-cost = $#cost-()_l$
#let progress-cost = $#cost-()_p$
#let contouring-weight = $k_c$
#let lag-weight = $k_l$
#let contouring-error = $e_c$
#let lag-error = $e_l$
#let progress-weight = $k_p$

The tracking cost requires a global reference trajectory that is to be followed. We can denote this trajectory as #reference-trajectory and assume it is given. A single point in this trajectory is expected to contain the position and heading of the robot at a specific time step, i.e. $#reference-point = [#reference-x quad #reference-y quad #reference-theta]$.

#definition(title: [Contouring & Lag Cost @Liniger2015])[
  The *contouring* and *lag* costs penalize the robot for being far from a nominal point corresponding to $#path-parameter in [0, L]$ along the reference trajectory #reference-trajectory. For these cost terms, the parameterization #path-parameter of the reference trajectory is a virtual state that moves along #reference-trajectory with a velocity $#path-parameter-change := dot(#path-parameter)$. Additionally, #path-parameter-change is also a control input to be optimized by the MPPI planner. The state and control input spaces are therefore augmented as follows:

  - $#augmented-state-single := [- #state-single - quad #path-parameter]$, #state-single is the original robot state, and
  - $#augmented-input-single := [- #input-single - quad #path-parameter-change]$, #input-single is the original robot control input.

  The dynamics of #path-parameter can be simply integrated as $#path-parameter _(t + 1) = #path-parameter _t + #path-parameter-change _t dot #delta-t$. Additionally, it is important that #reference-trajectory is parameterized by #path-parameter in such a way that the position and heading at a given #path-parameter can be efficiently queried. These queried quantities are denoted as:

  #align(
    center,
    $#x-at-arc-length := #reference-x (#path-parameter), quad #y-at-arc-length := #reference-y (#path-parameter), quad #theta-at-arc-length := #reference-theta (#path-parameter)$,
  )

  The *contouring error* #contouring-error and *lag error* #lag-error are defined as the orthogonal and parallel distances, respectively, between the robot's current position $(x, y)$ and the reference point $(#x-at-arc-length, #y-at-arc-length)$ along the direction defined by #theta-at-arc-length. This is illustrated in @tracking-error-figure.

  #align(
    center,
    $#contouring-error = sin(#theta-at-arc-length) (x - #x-at-arc-length) - cos(#theta-at-arc-length) (y - #y-at-arc-length) \
      #lag-error = - cos(#theta-at-arc-length) (x - #x-at-arc-length) - sin(#theta-at-arc-length) (y - #y-at-arc-length)$,
  )

  Finally, the *contouring cost* #contouring-cost and *lag cost* #lag-cost are defined as:

  $
    #contouring-cost = #contouring-weight dot #contouring-error^2, quad
    #lag-cost = #lag-weight dot #lag-error^2
  $ <contouring-lag-cost-equations>

  where $#contouring-weight, #lag-weight > 0$ are weighting factors.
]

#figure(
  tracking-error-diagram(),
  caption: [The black curve represents the desired path. The robot's current position has a contour error $e_c$ and a lag error $e_l$ relative to a nominal point (green) on the path. The red markers show the nominal points corresponding to different values of the path parameterization $#path-parameter in [0, 1]$.],
) <tracking-error-figure>

#definition(title: [Progress Cost @Liniger2015])[
  Since the costs given by @contouring-lag-cost-equations alone do not encourage progress along the reference trajectory, an additional *progress cost* #progress-cost needs to be added. This cost is defined as:

  $
    #progress-cost = - #progress-weight dot #path-parameter-change #delta-t
  $ <progress-cost-equation>

  where $#progress-weight > 0$ is a weighting factor.
]

The progress cost given by @progress-cost-equation pushes #path-parameter to move forward along the reference trajectory as fast as possible, pulling the robot along with it. Simultaneously, the contouring and lag costs given by @contouring-lag-cost-equations also pull the #path-parameter back, preventing it from moving too far ahead of the robot. If the weights #contouring-weight, #lag-weight and #progress-weight are chosen appropriately, the robot will try to follow the reference trajectory closely and maximize progress.

=== Comfort Cost

#let smoothing-cost = $#cost-()_s$
#let input-change = $Delta #input-single$
#let input-smooth-weight = $K_u$

#definition(title: [Control Smoothing Cost @Liniger2015])[
  To prevent erratic control behavior a *smoothing cost* #smoothing-cost can be used, which penalizes the rate of change of the control inputs.

  Let $#input-change _t = #input-single _t - #input-single _(t-1)$ be the change in control inputs at time step $t$ and $#input-smooth-weight := "diag"(k_1, ..., k_(#control-dimension))$ be a positive definite weighting matrix. The smoothing cost is then defined as:

  $
    #smoothing-cost = sum_t^(#horizon) || #input-smooth-weight #input-change _t ||^2
  $ <smoothing-cost-equation>
]

#bibliography("references.bib")
