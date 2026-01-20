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

#let ego-parts = $V$
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
#let min-distance-(..args, sub: none) = function(
  name: $bold(sans(d))$,
  ..args,
  sub: sub,
)
#let min-distance = min-distance-()
#let min-distance-batch-(..args, sub: none) = function(
  name: $bold(sans(D))$,
  ..args,
  sub: sub,
)
#let min-distance-batch = min-distance-batch-()

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

Cost terms related to obstacle avoidance typically also require measuring distances between parts of the robot and the environment. We can define a *batch distance function* for this purpose:

#definition(title: "Batch Distance Function")[
  Let $#state-batch = {#state _1, #state _2, ..., #state _#rollouts}$ be a batch of robot states. The *batch distance function* #min-distance-batch-(state-batch) : $#state-space^#rollouts -> bb(R)^(#horizon times #ego-parts times #rollouts)$ computes the minimum distances between $V$ parts of the robot and the nearest obstacles for each time step and rollout in the batch. The output is defined as:

  #align(
    center,
    $#min-distance-batch-(state-batch) = {#min-distance-($#state _1$), #min-distance-($#state _2$), ..., #min-distance-($#state _#rollouts$)} = {#min-distance _1, #min-distance _2, ..., #min-distance _#rollouts} := #min-distance-batch$,
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

  The dynamics of #path-parameter can be simply integrated as:

  $
    #path-parameter _(t + 1) = #path-parameter _t + #path-parameter-change _t dot #delta-t
  $ <path-parameter-dynamics-equation>

  Additionally, it is important that #reference-trajectory is parameterized by #path-parameter in such a way that the position and heading at a given #path-parameter can be efficiently queried. These queried quantities are denoted as:

  #align(
    center,
    $#x-at-arc-length := #reference-x (#path-parameter), quad #y-at-arc-length := #reference-y (#path-parameter), quad #theta-at-arc-length := #reference-theta (#path-parameter)$,
  )

  Let $t$ be the current time step, $(x, y)$ - the robot's current position, and #path-parameter - the current path parameter. The *contouring error* #contouring-error and *lag error* #lag-error are defined as the orthogonal and parallel distances, respectively, between $(x, y)$ and the reference point $(#x-at-arc-length, #y-at-arc-length)$ along the direction defined by #theta-at-arc-length. This is illustrated in @tracking-error-figure.

  #align(
    center,
    $#contouring-error = sin(#theta-at-arc-length) (x - #x-at-arc-length) - cos(#theta-at-arc-length) (y - #y-at-arc-length) \
      #lag-error = - cos(#theta-at-arc-length) (x - #x-at-arc-length) - sin(#theta-at-arc-length) (y - #y-at-arc-length)$,
  )

  Then, the *contouring cost* #contouring-cost and *lag cost* #lag-cost for time step $t$ are defined as:

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
  Since the costs given by @contouring-lag-cost-equations alone do not encourage progress along the reference trajectory, an additional *progress cost* #progress-cost needs to be added. For a time step $t$, this cost is defined as:

  $
    #progress-cost = - #progress-weight dot #path-parameter-change #delta-t
  $ <progress-cost-equation>

  where $#progress-weight > 0$ is a weighting factor, and #path-parameter-change is the velocity of the path parameter #path-parameter at time step $t$ (see @path-parameter-dynamics-equation).
]

The progress cost given by @progress-cost-equation pushes #path-parameter to move forward along the reference trajectory as fast as possible, pulling the robot along with it. Simultaneously, the contouring and lag costs given by @contouring-lag-cost-equations also pull the #path-parameter back, preventing it from moving too far ahead of the robot. If the weights #contouring-weight, #lag-weight and #progress-weight are chosen appropriately, the robot will try to follow the reference trajectory closely and maximize progress.

=== Safety Cost

#let collision-cost-(sub: none) = function(
  name: $#cost-()$,
  sub: $"col" #sub$,
)
#let collision-cost = collision-cost-()
#let boundary-cost-(sub: none) = function(
  name: $#cost-()$,
  sub: $"bnd" #sub$,
)
#let boundary-cost = boundary-cost-()
#let min-distance-single-(sub: none) = function(
  name: $sans(d)$,
  sub: sub,
)
#let min-distance-single = min-distance-single-()
#let distance-threshold = min-distance-single-(sub: 0)
#let collision-weight = $k_("col")$

#definition(title: [Collision Cost @Schulman2013])[
  To avoid collisions with obstacles, a *collision cost* #collision-cost can be used.

  Let $V$ be the number of parts the ego robot consists of, and #min-distance-single-(sub: $i$) - the minimum distance between the $i$-th part of the robot and the obstacles nearest to it at a time step $t$. Given a safety distance threshold #distance-threshold, the
  collision cost at time step $t$ is defined as:

  $
    #collision-cost = sum^V_(i=1) #collision-cost-(sub: $,i$), quad
    #collision-cost-(sub: $,i$) = cases(
      #collision-weight (#distance-threshold - #min-distance-single-(sub: $i$)) comma quad "if" #min-distance-single-(sub: $i$) < #distance-threshold,
      0 comma quad "otherwise"
    )
  $ <collision-cost-equation>

  where $#collision-weight > 0$ is a weighting factor.
]

#definition(title: "Boundary Cost")[
  Although the contouring cost already encourages the robot to stay close to the reference trajectory, it may be possible that an environment boundary (e.g. a wall) is close to the reference trajectory. In such cases, an additional *boundary cost* similar to the collision cost can be used to prevent the robot from getting too close to the boundary.

  The boundary cost formulation is similar to @collision-cost-equation. Let #min-distance-single-(sub: none) be the minimum distance between the robot and the environment boundary at a time step $t$. Given a safety distance threshold #distance-threshold, the boundary cost at time step $t$ is defined as:

  $
    #boundary-cost = cases(
      #collision-weight (#distance-threshold - #min-distance-single-(sub: none)) comma quad "if" #min-distance-single-(sub: none) < #distance-threshold,
      0 comma quad "otherwise"
    )
  $ <boundary-cost-equation>

  Note that the distance #min-distance-single-(sub: none) is measured w.r.t. the entire robot, not individual parts as in @collision-cost-equation.
]

=== Comfort Cost

#let smoothing-cost = $#cost-()_s$
#let effort-cost = $#cost-()_n$
#let input-change = $Delta #input-single$
#let input-smooth-weight = $K_u$
#let control-effort-weight = $K_n$
#let control-effort-weight-single = $k_n$
#let control-effort-cost-note = footnote[Although this cost is theoretically necessary for correct importance sampling, in practice the planner can work without it. We use it as a regularization/comfort term. Hence, it is written in the section corresponding to comfort costs.]

#definition(title: [Control Smoothing Cost @Liniger2015])[
  To prevent erratic control behavior a *smoothing cost* #smoothing-cost can be used, which penalizes the rate of change of the control inputs.

  Let $#input-change _t = #input-single _t - #input-single _(t-1)$ be the change in control inputs at time step $t$ and $#input-smooth-weight := "diag"(k_1, ..., k_(#control-dimension))$ be a positive definite weighting matrix. The smoothing cost at time step $t$ is then defined as:

  $
    #smoothing-cost = || #input-smooth-weight #input-change _t ||^2
  $ <smoothing-cost-equation>
]

#definition(
  title: [Control Effort Cost#control-effort-cost-note @Williams2017],
)[
  Penalizing extraneous control effort can also lead to more consistent motion. We can achieve this by adding a *control effort cost* #effort-cost that penalizes the magnitude of the control inputs.

  If the rollouts are sampled from a Gaussian distribution with covariance $Sigma$ @Williams2017, the control effort cost at time step $t$ can be computed as:

  $
    #effort-cost = #temperature / 2 #input-single _t^top Sigma^(-1) #input-single _t
  $

  With #control-effort-weight-single being the cost weight, and #temperature - the temperature parameter. However, for flexibility, we define this cost more generally as:

  $
    #effort-cost = #control-effort-weight #input-single _t^top #input-single _t = #control-effort-weight || #input-single _t ||^2
  $ <control-effort-cost-equation>

  And the user can decide what the weighting factors #control-effort-weight should be.
]

== Motion Prediction

=== Constant Velocity (CV) Model @Schubert2008

To effectively avoid collisions, only considering the current state of moving obstacles is insufficient in most cases. Instead, we need to predict how these obstacles will move in the near future. A simple method to predict the future motion of obstacles is to assume they will continue moving with their current velocity. We can call this a *constant velocity model*.

For example, assume a robot follows the kinematic bicycle model described in @kinematic-bicycle-equations. The state of this robot can then be represented as $#state-single _("obs") = [x quad y quad theta quad v]$. In this case, we can assume the state variable $v$, representing the robot's linear velocity, remains constant over the prediction horizon. This means that we predict the robot will continue moving on a straight line in the direction of its current heading $theta$ with speed $v$ for the next #horizon time steps.

=== Constant Steering Angle & Velocity (CSAV) Model @Schubert2008

Since the above model ignores the steering angle of the robot, it will yield inaccurate predictions when the robot is turning, or simply following a curved path. A slightly more sophisticated option is to also consider the steering angle $delta$ of the robot and assume it also remains constant over the prediction horizon. The resulting model is called a *constant steering angle & velocity model*.

== State Estimation

#let observed-state-single = $#state-single _("obs")$

Typically, only a small subset of the full robot state is directly observable through sensors. Let's assume we can directly observe the position $(x, y)$ and heading $theta$ of a robot, but not its velocity $v$. The observed state can then be represented as $#observed-state-single = [x quad y quad theta]$. The remaining information has to be inferred from these measurements.

=== Velocity

A simple approach to estimate the velocity $v$ of the robot is to compute the backward finite difference in position over time.

#definition(title: "Finite Difference Velocity Estimate")[
  Considering only the last two observed positions $(x_(t-1), y_(t-1))$ and $(x_t, y_t)$ at time steps $t-1$ and $t$, we can estimate the velocity at time step $t$ as:

  #align(
    center,
    $v_t approx sqrt((x_t - x_(t-1))^2 + (y_t - y_(t-1))^2) / #delta-t$,
  )
]

Although this is not exactly consistent with the kinematic bicycle model (e.g. if the robot's tires are slipping), it provides a reasonable approximation of the robot's speed.

=== Steering Angle

From @kinematic-bicycle-equations, we know that the heading rate $dot(theta)$ is related to the steering angle $delta$ as:

#align(
  center,
  $dot(theta) = v / L tan(delta)$,
)

Inverting this relationship gives:

#align(
  center,
  $delta = arctan(L dot(theta) / v)$,
)

#definition(title: "Finite Difference Steering Angle Estimate")[

  Given observations of the heading $theta$ at time steps $t-1$ and $t$, and the (estimated) velocity $v_t$, we can estimate the heading rate, and subsequently the steering angle, as:

  #align(
    center,
    $dot(theta)_t approx (theta_t - theta_(t-1)) / #delta-t, quad delta_t approx arctan(L dot(theta)_t / v_t)$,
  )
]

If the velocity $v_t$ is very small, the steering angle estimate $delta_t$ will be unreliable. In such cases, we can assume the steering angle is zero.

== Incorporating Uncertainty into Predictions

When using motion prediction models like the CV or CSAV models, we make assumptions about the future motion of obstacles that we know may not hold exactly. Furthermore, sensor noise and inaccuracies in state estimation can also lead to uncertainty about the current state of obstacles. We can represent this uncertainty in a probabilistic manner in our predictions in several ways.

=== Gaussian Noise

#let velocity-covariance-(sub: none) = function(
  name: $Sigma$,
  sub: if sub != none { $v,#sub$ } else { $v$ },
)
#let position-covariance-(sub: none) = function(
  name: $Sigma$,
  sub: if sub != none { $p,#sub$ } else { $p$ },
)
#let velocity-covariance = velocity-covariance-()
#let position-covariance = position-covariance-()

A simple approach to represent the uncertainty in predictions is to consider future states of obstacles as Gaussian random variables. In this context, the predictions provided by the motion model (e.g. CV or CSAV) can be interpreted as the means of these Gaussian distributions. The only remaining component is the variance of these distributions, which can be defined depending on the particular motion model we are using.

#definition(title: [Gaussian Noise for CV Model @Salzmann2020])[
  Assume an obstacle follows the CV model. Let $#velocity-covariance-(sub: $t$)$ be the covariance matrix representing the uncertainty in the velocity estimate of the obstacle at time step $t$. Then, the covariance matrix $#position-covariance-(sub: $t + 1$)$ for the position at the next time step $t + 1$ can be computed as:

  #align(
    center,
    $#position-covariance-(sub: $t + 1$) = #position-covariance-(sub: $t$) + #delta-t^2 #velocity-covariance-(sub: $t$)$,
  )

  Assuming #velocity-covariance-(sub: $t$) remains constant over time, we can unroll this equation to get the covariance at time step $t + k$ as:

  #align(
    center,
    $#position-covariance-(sub: $t + k$) = #position-covariance-(sub: $t$) + k #delta-t^2 #velocity-covariance-(sub: $t$)$,
  )
]

Although not entirely accurate, the above approach can also be applied to the CSAV model. In this case, the uncertainty in the steering angle estimate will be ignored.

#pagebreak()

#bibliography("references.bib")
