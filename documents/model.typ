#import "@preview/lovelace:0.3.0": pseudocode-list, indent
#import "@preview/subpar:0.2.2"

#import "examples.typ": (
  mppi-example-diagram,
  kinematic-bicycle-diagram,
  kinematic-unicycle-diagram,
  tracking-error-diagram,
  sat-diagram,
  circle-approximation-diagram,
)
#import "plots.typ": plot-hinge-loss

#import "@local/roboter:0.3.9": (
  function,
  algorithm,
  definition,
  model,
  draw,
  vehicle-theme,
)

#set math.equation(numbering: "(1)", supplement: "eq.")
#set heading(numbering: "1.")
#outline()
#pagebreak()

= Model-Predictive Path Integral (MPPI) Control

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

= Dynamical Model

== Kinematic Bicycle Model (KBM)

#let speed-footnote = footnote[Note that the speed $v$ in this context is defined as the projection of the robot's velocity onto its heading direction, not the magnitude of the velocity. This means the sign of $v$ can be negative, e.g. if the vehicle is moving backwards.]

A common choice for the dynamical model of a wheeled robot is the *kinematic bicycle model*. This model represents the robot as a bicycle with a single front and a single rear wheel. Other than simplifying the number of wheels, the model also assumes that there is no slip between the wheels and the ground. This assumption is not necessarily true at high speeds or during sharp turns, so we should keep that in mind. Here's a diagram for the model:

#figure(
  kinematic-bicycle-diagram(),
  caption: [The kinematic bicycle model. The robot state is defined by position $(x, y)$, heading $theta$, and speed $v_r$. The control inputs are acceleration $a := dot(v)_r$ and steering angle $delta$. The wheelbase $L$ is the distance between the front and rear axles. $R$ denotes the turning radius for the current $delta$.],
)

The control inputs $#input-single := [a quad delta]$ are:
- $a$: Acceleration
- $delta$: Steering angle at the front wheel

The state $#state-single := [x quad y quad theta quad v]$ of the robot is represented by four variables:
- $x$: Position along the x-axis
- $y$: Position along the y-axis
- $theta$: Heading angle (orientation) in radians
- $v$: Speed#speed-footnote of the robot (corresponds to $v_r$ in the diagram)

#model(title: [Kinematic Bicycle Model @Polack2017])[
  The continuous-time dynamics of the kinematic bicycle model are given as:

  $
    dot(x) = v cos(theta), quad dot(y) = v sin(theta), quad dot(theta) = v / L tan(delta), quad dot(v) = a
  $ <kinematic-bicycle-equations>

  where $L$ is the wheelbase (distance between the front and rear axles).
]

=== Euler Integration for KBM

#let delta-t = $Delta t$

To simulate the model, we discretize @kinematic-bicycle-equations with a time step size of #delta-t to get:

$
  x_(t+1) = x_t + v_t cos(theta_t) dot #delta-t \
  y_(t+1) = y_t + v_t sin(theta_t) dot #delta-t \
  theta_(t+1) = theta_t + v_t / L tan(delta_t) dot #delta-t \
  v_(t+1) = v_t + a_t dot #delta-t
$ <kinematic-bicycle-discretized-equations>

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
      + $v' arrow.l v + a dot #delta-t$ #h(1fr) ➤ Update speed

      + *return* $#state-single ' = [x' quad y' quad theta' quad v']$
  ]
]

== Unicycle Model (UM)

A simpler alternative to the kinematic bicycle model is the *unicycle model*. This model represents the robot as a single point with a heading, abstracting away the physical wheel configuration. It's particularly suitable for differential-drive robots or when high-fidelity steering dynamics are not required.

#figure(
  kinematic-unicycle-diagram(),
  caption: [The unicycle model. The robot state is defined by position $(x, y)$ and heading $theta$. The control inputs are linear velocity $v$ and angular velocity $omega$.],
)

The control inputs $#input-single := [v quad omega]$ are:
- $v$: Linear velocity
- $omega$: Angular velocity (yaw rate)

The state $#state-single := [x quad y quad theta]$ of the robot is represented by three variables:
- $x$: Position along the x-axis
- $y$: Position along the y-axis
- $theta$: Heading angle (orientation) in radians

#model(title: [Unicycle Model @Oriolo2002])[
  The continuous-time dynamics of the unicycle model are given as:

  $
    dot(x) = v cos(theta), quad dot(y) = v sin(theta), quad dot(theta) = omega
  $ <unicycle-equations>
]

=== Euler Integration for Unicycle Model

To simulate the model, we discretize @unicycle-equations with a time step size of #delta-t to get:

$
  x_(t+1) = x_t + v_t cos(theta_t) dot #delta-t \
  y_(t+1) = y_t + v_t sin(theta_t) dot #delta-t \
  theta_(t+1) = theta_t + omega_t dot #delta-t
$ <unicycle-discretized-equations>

#algorithm(title: "Euler Integration Step for the Unicycle Model")[
  #pseudocode-list(hooks: .5em)[
    + *Given:*
      - Control input #input-single $= [v quad omega]$, Current state #state-single $= [x quad y quad theta]$
      - Time step #delta-t

    + *Compute next state:*
      + $x' arrow.l x + v dot cos(theta) dot #delta-t$ #h(1fr) ➤ Update x position
      + $y' arrow.l y + v dot sin(theta) dot #delta-t$ #h(1fr) ➤ Update y position
      + $theta' arrow.l theta + omega dot #delta-t$ #h(1fr) ➤ Update heading

      + *return* $#state-single ' = [x' quad y' quad theta']$
  ]
]

= Cost Functions

When using MPPI to control the motion of a mobile robot, the terms of the cost function #cost can be split into the following categories:
- *Tracking:* Costs that encourage the robot to follow a desired trajectory or reach a specific goal.
- *Safety:* Costs that discourage the robot from getting too close to obstacles, or unsafe areas.
- *Comfort:* Costs that promote smooth and comfortable motion.

== Tracking Cost

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

== Safety Cost

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

== Comfort Cost

#let input-single-(at: none) = $#input-single _(#at)$
#let state-dimension = $D_x$
#let control-dimension = $D_u$
#let smoothing-cost = $#cost-()_s$
#let effort-cost = $#cost-()_n$
#let input-change = $Delta #input-single$
#let input-smooth-weight = $K_u$
#let control-effort-weight = $K_n$
#let control-effort-cost-note = footnote[Although this cost is theoretically necessary for correct importance sampling, in practice the planner can work without it. We use it as a regularization/comfort term. Hence, it is written in the section corresponding to comfort costs.]

#definition(title: [Control Smoothing Cost @Liniger2015])[
  To prevent erratic control behavior a *smoothing cost* #smoothing-cost can be used, which penalizes the rate of change of the control inputs.

  Let $#input-change _t = #input-single-(at: $t$) - #input-single-(at: $t-1$)$ be the change in control inputs at time step $t$ and $#input-smooth-weight := "diag"(k_1, ..., k_(#control-dimension))$ be a positive definite weighting matrix. The smoothing cost at time step $t$ is then defined as:

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
    #effort-cost = #temperature / 2 #input-single-(at: $t$)^top Sigma^(-1) #input-single-(at: $t$)
  $

  With #temperature being the temperature parameter. However, for flexibility, we define this cost more generally as:

  $
    #effort-cost = #control-effort-weight #input-single-(at: $t$)^top #input-single-(at: $t$) = #control-effort-weight || #input-single-(at: $t$) ||^2
  $ <control-effort-cost-equation>

  And the user can decide what the weighting factors #control-effort-weight should be.
]

= Distance Computation

For the collision cost, we need to be able to compute minimum distances to obstacles. Although not necessary, distance measurement can also be used for collision detection. Depending on the geometry of the ego and obstacles, as well as the desired precision and available computational resources, different methods can be used.

== Separating Axis Theorem (SAT) Method @Gottschalk1997

In the simplest case, we can assume that both the ego robot and the obstacles can be represented as rectangles. For example, when dealing with vehicles or wheeled mobile robots, this is a reasonable assumption. We can then compute the minimum distance between the ego and an obstacle by using the *separating axis theorem*.

#definition(title: "Separating Axis")[
  An axis is *separating* if the projections of the two rectangles onto this axis do not overlap.

  More formally, given rectangles $A$ and $B$ with vertices ${a_i}$ and ${b_j}$, define their projections onto axis $n$ as:
  $
    I_A = [min_i (a_i dot n), quad max_i (a_i dot n)] quad quad I_B = [min_j (b_j dot n), quad max_j (b_j dot n)]
  $

  Then $n$ is a separating axis iff $I_A inter I_B = emptyset$.
]

If we find an axis that separates the two rectangles, we can compute the minimum distance between them as the gap between their projections onto this axis. Now there's a lot of possible axes we could check (infinitely many), but we only need to check just a few of them to find the minimum distance.

#definition(title: "Separating Axis Theorem (SAT)")[
  Two convex polygons do not intersect if and only if there exists a separating axis. Furthermore, if such an axis exists, it must be parallel to an edge of one of the polygons.

  For two rectangles, this reduces to checking only 4 axes: the two edge normals of each rectangle. The minimum distance is the maximum gap between projections across all separating axes:
  $
    d_min = max_n "gap"(I_A, I_B)
  $
  where $"gap"(I_A, I_B) = max(min I_B - max I_A, min I_A - max I_B)$. If the gap is negative, the rectangles intersect and the numerical value represents the penetration depth.
]

You can see the idea in the below figure#footnote[This approach is not limited to rectangles and can be applied to any convex polygon.].

#figure(
  sat-diagram(),
  caption: [The ego robot (blue) and an obstacle (red) are represented as rectangles. The minimum distance between them can be computed by projecting their corners onto the separating axes (dashed gray lines) and finding the largest gap between the projections. In this example, the minimum distance is given by the gap along the separating axis $n_0$. Only two of the four possible separating axes are shown for clarity.],
) <sat-figure>

== Circle Approximation Method @Tolksdorf2024

Although using the SAT method is accurate and comparatively efficient, even faster methods exist. One such method is to approximate the rectangular shapes of the ego robot and obstacles with circles. Computing distances between circles is much easier and faster than using the SAT method. Depending on how accurately we want to approximate the rectangles with circles, we can use more or fewer circles.

#definition(title: "Circle Approximation Distance")[
  Let the ego robot be approximated by $N$ circles with centers $c_i$ and radii $r_i$, $i = 1, ..., N$. Similarly, let an obstacle be approximated by $M$ circles with centers $d_j$ and radii $s_j$, $j = 1, ..., M$. The minimum distance between the ego and the obstacle can then be computed as:

  $
    d_min = min_(i=1,...,N) min_(j=1,...,M) || c_i - d_j || - (r_i + s_j)
  $
]

Typically, the circles are used to get a conservative approximation of the rectangle. This means that the circles could indicate a collision even when the rectangles do not actually collide, but never the other way around. Here's what the circle approximation looks like:

#subpar.grid(
  figure(
    circle-approximation-diagram(
      circle-count: 2,
      radius: calc.sqrt(2),
      safety-threshold: 0.0,
      show-safety-threshold-separately: false,
      theme: vehicle-theme(label-offsets: (radius: (2.0, 0.5))),
    ),
    caption: [Two-circle approximation (not conservative)],
  ),
  figure(
    circle-approximation-diagram(
      circle-count: 3,
      radius: calc.sqrt(2),
      safety-threshold: 0.0,
      show-safety-threshold-separately: false,
      theme: vehicle-theme(
        label-offsets: (
          radius: (2.0, 0.5),
          intersection-markers: (-0.25, 0.4),
        ),
      ),
    ),
    caption: [Three-circle approximation],
  ),
  figure(
    circle-approximation-diagram(
      circle-count: 2,
      radius: calc.sqrt(2),
      safety-threshold: 1.5,
      show-safety-threshold-separately: true,
      theme: vehicle-theme(
        label-offsets: (
          radius: (-0.25, 2.0),
          safety-threshold: (0.5, 1.0),
          minimum-threshold: (-0.25, 0.0),
        ),
      ),
    ),
    caption: [Two-circle approximation with additional safety margin],
  ),
  figure(
    circle-approximation-diagram(
      circle-count: 3,
      radius: calc.sqrt(2),
      safety-threshold: 1.5,
      show-safety-threshold-separately: true,
      theme: vehicle-theme(
        label-offsets: (
          radius: (-0.25, 2.0),
          safety-threshold: (0.5, 1.0),
          minimum-threshold: (-0.25, 0.0),
        ),
      ),
    ),
    caption: [Three-circle approximation with additional safety margin],
  ),
  columns: (1fr, 1fr),
  caption: [A vehicle (red rectangle) approximated by circles (blue). The green lines show the smallest safety distance from the rectangle sides to the circles. As long as this safety distance is not negative, the rectangle is fully contained within the circles, and the approximation is conservative (b, c, d). Depending on the cost function, an additional safety margin ($#min-distance-single-(sub: 0)$) can be added to the circle radii.],
)

=== Choosing Enough Circles

If the distance measurement is used for penalizing collisions in the cost function, the cost function can be visualized to determine whether the approximation leads to the cost landscape having desirable properties.

In the case of the hinge loss more circles lead to a better cost landscape, as can be seen in @collision-cost-visualization, but the computational complexity is $O(n times m)$, where $n$ is the number of ego circles and $m$ is the number of obstacle circles. Thus, approximating both ego and obstacles with 3 circles instead of 2 leads to a 2.25x increase in the number of distance computations, which is a noteworthy tradeoff.

#subpar.grid(
  figure(
    plot-hinge-loss(
      centers: ((-1.5, 0), (1.5, 0)),
      ego-radius: calc.sqrt(2),
      safety-threshold: 0.0,
      obstacle-radius: calc.sqrt(2),
    ),
    caption: [Two-circle approximation (not conservative)],
    placement: alignment.bottom,
  ),
  figure(
    plot-hinge-loss(
      centers: ((-1.5, 0), (0, 0), (1.5, 0)),
      ego-radius: calc.sqrt(2),
      safety-threshold: 0.0,
      obstacle-radius: calc.sqrt(2),
    ),
    caption: [Three-circle approximation],
    placement: alignment.bottom,
  ),
  figure(
    plot-hinge-loss(
      centers: ((-1.5, 0), (1.5, 0)),
      ego-radius: calc.sqrt(2),
      safety-threshold: 1.5,
      obstacle-radius: calc.sqrt(2),
    ),
    caption: [Two-circle approximation with additional safety margin],
    placement: alignment.bottom,
  ),
  figure(
    plot-hinge-loss(
      centers: ((-1.5, 0), (0, 0), (1.5, 0)),
      ego-radius: calc.sqrt(2),
      safety-threshold: 1.5,
      obstacle-radius: calc.sqrt(2),
    ),
    caption: [Three-circle approximation with additional safety margin],
    placement: alignment.bottom,
  ),
  columns: (1fr, 1fr),
  caption: [
    The collision cost given by @collision-cost-equation can be visualized by plotting the cost #collision-cost as a function of the position of a circular obstacle ($r = sqrt(2)$) at position $(x, y)$ relative to the ego robot. We can see the function then has saddle points in the two-circle cases (a, c), but has only one peak in the cases (b, d). Furthermore, the points between the approximating circles in case (a) have slightly lower costs than other points, which could be undesirable. Ego circle radius $r = sqrt(2)$ in all cases, #min-distance-single-(sub: 0) = 1.5 in cases (c) and (d).
  ],
  label: <collision-cost-visualization>,
)

= Motion Prediction

To effectively avoid collisions, only considering the current state of moving obstacles is insufficient in most cases. Instead, we need to predict how the obstacles will move in the near future. Several of the many possible motion prediction models are described below. The assumed system dynamics for the prediction models are formulated differently from the underlying dynamical models. Namely, the state of the system is always augmented to include the inputs of the system as well. As will be seen in the state estimation section (@state-estimation), formulating the prediction models in this way allows us to easily incorporate knowledge about the uncertainty of our assumptions into the predictions.

== Constant Velocity (CV) Model @Schubert2008 <cv-model>

#let state-single-(at: none) = $#state-single _(#at)$
#let state-transition-matrix-(at: none) = function(name: $sans(A)$, sub: at)
#let state-transition-matrix = state-transition-matrix-()

A simple method to predict the future motion of obstacles is to assume they will continue moving with their current velocity. We call this a *constant velocity model*.

#definition(title: "Constant Velocity Model for Planar Obstacles")[
  Let the state of an obstacle moving on a plane be given as $#state-single = [x quad y quad v_x quad v_y]$. Assuming the obstacle will continue moving at a constant velocity $(v_x, v_y)$, its motion can be approximated by the following equations:
  $
    #state-single-(at: $t$) = #state-single-(at: $t-1$) + vec(v_(x,t-1) dot #delta-t, v_(y,t-1) dot #delta-t, 0, 0, delim: "[") quad <=> quad #state-single-(at: $t$) = #state-transition-matrix-(at: $t$) #state-single-(at: $t-1$), quad #state-transition-matrix-(at: $t$) := mat(
      1, 0, #delta-t, 0, ;
      0, 1, 0, #delta-t, ;
      0, 0, 1, 0, ;
      0, 0, 0, 1, ;
      delim: "[",
    )
  $

  Where #delta-t is the time step size for prediction. Since the dynamics are linear, we can also represent it in matrix form as shown above.
]

== Constant Steering Angle & Acceleration (CSAA) Model @Schubert2008 <csaa-model>

#let state-transition-function-(..args) = function(name: $f$, ..args)
#let state-transition-function = state-transition-function-()

For many applications#footnote[For pedestrian motion prediction, the constant velocity model is surprisingly performant @Schoeller2020.], the above model makes two unlikely assumptions:
1. The obstacle will continue moving with the same velocity,
2. The obstacle's heading will remain constant.

If the obstacle is a vehicle, both assumptions are frequently violated (e.g. when following a curved road, turning or slowing down at an intersection). Thus, the model will frequently yield inaccurate predictions. A more sophisticated option is to assume the obstacle follows the kinematic bicycle model (@kinematic-bicycle-equations) and that the acceleration $a$ and
the steering angle $delta$ remain constant. The resulting model is called a *constant steering angle & acceleration model*.

#definition(title: "Constant Steering Angle & Acceleration Model")[
  Let the state of an obstacle moving on a plane be given as $#state-single = [x quad y quad theta quad v quad a quad delta]$. Assuming the obstacle will continue moving with constant acceleration $a$ and steering angle $delta$, its motion can be approximated by the following equations:

  $
    #state-single-(at: $t$) = #state-single-(at: $t-1$) + vec(v_(t-1) cos(theta_(t-1)) dot #delta-t, v_(t-1) sin(theta_(t-1)) dot #delta-t, v_(t-1) / L tan(delta_(t-1)) dot #delta-t, a_(t-1) dot #delta-t, 0, 0, delim: "[") quad <=> quad #state-single-(at: $t$) = #state-transition-function-(state-single-(at: $t-1$)), quad #state-transition-function-(state-single) := vec(
      x + v cos(theta) dot #delta-t,
      y + v sin(theta) dot #delta-t,
      theta + v / L tan(delta) dot #delta-t,
      v + a dot #delta-t,
      a,
      delta,
      delim: "[",
    ) quad
  $

  Where $L$ is the wheelbase of the obstacle and #delta-t is the time step size for prediction.
]

== Constant Turn Rate & Velocity (CTRV) Model @Schubert2008 <ctrv-model>

Alternatively, we can assume the obstacle follows the unicycle model (@unicycle-equations) and that both the linear velocity $v$ and angular velocity $omega$ remain constant. The resulting model is called a *constant turn rate & velocity model*.

This model can be used, for example, if the obstacles are known to be other robots using a differential drive.

#definition(title: "Constant Turn Rate & Velocity Model for Planar Obstacles")[
  Let the state of an obstacle moving on a plane be given as $#state-single = [x quad y quad theta quad v quad omega]$. Assuming the obstacle will continue moving with constant linear velocity $v$ and angular velocity $omega$, its motion can be approximated by the following equations:

  $
    #state-single-(at: $t$) = #state-single-(at: $t-1$) + vec(v_(t-1) cos(theta_(t-1)) dot #delta-t, v_(t-1) sin(theta_(t-1)) dot #delta-t, omega_(t-1) dot #delta-t, 0, 0, delim: "[") quad <=> quad #state-single-(at: $t$) = #state-transition-function-(state-single-(at: $t-1$)), quad #state-transition-function-(state-single) := vec(
      x + v cos(theta) dot #delta-t,
      y + v sin(theta) dot #delta-t,
      theta + omega dot #delta-t,
      v,
      omega,
      delim: "[",
    ) quad
  $

  Where #delta-t is the time step size for prediction.
]

== Curvilinear Motion Models @Schubert2008 <curvilinear-models>

The CV, CSAA and CTRV models described above can all be classified as *curvilinear motion models*. Furthermore, additional assumptions can be imposed onto CSAA and CTRV to create more specialized curvilinear motion models. For example, the CSAA model can be simplified to a *constant steering angle & velocity (CSAV) model* by assuming the acceleration $a$ is zero.

= State Estimation <state-estimation>

#let observed-state-single-(at: none) = function(name: $sans(z)$, sub: at)
#let observed-state-single = observed-state-single-()

Typically, only a small subset of the full robot state is directly observable through sensors. Let's assume we can directly observe the position $(x, y)$ and heading $theta$ of another moving obstacle, but not its speed $v$. The observed state can then be represented as $#observed-state-single = [x quad y quad theta]$. The remaining information has to be inferred from these measurements.

== Speed & Acceleration

A simple approach to estimate the speed $v$ of the robot is to compute the backward finite difference in position over time.

#definition(title: "Finite Difference Speed Estimate")[
  Considering only the last two observed poses $(x_(t-1), y_(t-1), theta_(t-1))$ and $(x_t, y_t, theta_t)$ at time steps $t-1$ and $t$, we can estimate the speed at time step $t$ as:

  $
    v_t approx ((x_t - x_(t-1)) cos(theta_t) + (y_t - y_(t-1)) sin(theta_t)) / #delta-t
  $
]

If more observations are available, the acceleration $a$ can also be estimated.

#definition(title: "Finite Difference Acceleration Estimate")[
  Given the speed estimates at time steps $t-1$ and $t$, we can estimate the acceleration at time step $t$ as:

  $
    a_t := (v_t - v_(t-1)) / #delta-t
  $
]

== Heading Rate & Steering Angle

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

  Given observations of the heading $theta$ at time steps $t-1$ and $t$, and the (estimated) speed $v_t$, we can estimate the heading rate as:

  $
    omega_t := dot(theta)_t approx (theta_t - theta_(t-1)) / #delta-t
  $ <finite-difference-heading-rate-equation>

  Subsequently the steering angle is:

  $
    delta_t approx arctan(L dot(theta)_t / v_t)
  $ <finite-difference-steering-angle-equation>
]

If the speed $v_t$ is very small, the steering angle estimate $delta_t$ will be unreliable. In such cases, we can assume the steering angle is zero. If a unicycle model is used instead of the kinematic bicycle model, then the angular velocity $omega_t$ can be estimated using @finite-difference-heading-rate-equation.

== Kalman Filter (KF)

#let gaussian-(..args) = function(name: $cal(N)$, ..args)
#let mean-(sub: none, at: none, prediction: false) = function(
  name: if prediction { $overline(mu)$ } else { $mu$ },
  sub: if at != none { $#sub,#at$ } else { sub },
)
#let covariance-(sub: none, at: none, prediction: false) = function(
  name: if prediction { $overline(Sigma)$ } else { $Sigma$ },
  sub: if at != none { $#sub,#at$ } else { sub },
)
#let mean-state-single-(at: none, prediction: false) = mean-(
  sub: state-single,
  at: at,
  prediction: prediction,
)
#let mean-state-single = mean-state-single-()
#let covariance-state-single-(at: none, prediction: false) = covariance-(
  sub: state-single,
  at: at,
  prediction: prediction,
)
#let covariance-state-single = covariance-state-single-()
#let belief-state-single-(at: none) = {
  $b(#function(name: state-single, sub: at))$
}
#let control-input-matrix-(at: none) = function(name: $sans(B)$, sub: at)
#let control-input-matrix = control-input-matrix-()
#let process-noise-covariance-(at: none) = function(name: $sans(R)$, sub: at)
#let process-noise-covariance = process-noise-covariance-()
#let observation-noise-covariance-(at: none) = function(
  name: $sans(Q)$,
  sub: at,
)
#let observation-noise-covariance = observation-noise-covariance-()
#let observation-noise-(at: none) = function(name: $sans(epsilon)$, sub: at)
#let observation-matrix-(at: none) = function(name: $sans(H)$, sub: at)
#let observation-matrix = observation-matrix-()
#let kalman-gain-(at: none) = function(name: $sans(K)$, sub: at)
#let kalman-gain = kalman-gain-()
#let observation-dimension = $D_z$

Although using finite differences as in the previous sections is simple and gives a single number as an estimate for the unobserved states, it does not capture the uncertainty in these estimates. To capture this information as well, we can instead use a Kalman filter.

Unlike the previous approaches, we no longer consider exact values of the unobserved states (e.g. speed and steering angle), but rather a belief about the state, represented as a Gaussian distribution. The goal is then to combine the information we get from a new observation with our existing belief without discarding information from one or the other.

#definition(title: [Kalman Filter State Estimate @Thrun2005])[
  Let $#state-single-(at: $t-1$)$ be the true state of the system at time step $t-1$, and $#belief-state-single-(at: $t-1$) := #gaussian-(mean-state-single-(at: $t-1$), covariance-state-single-(at: $t-1$))$ - the belief about this state. To compute the belief for the next time step, #belief-state-single-(at: $t$), we perform two steps:

  1. *Predict* what the state should be based on our knowledge of the system dynamics, and
  2. *Update* the prediction once we receive a new observation.

  *Prediction Step:*

  Let #state-transition-matrix-(at: $t$) and #control-input-matrix-(at: $t$) be the state transition and control input matrices describing the *linear* system dynamics, $#input-single-(at: $t$)$ be the assumed control input (intent) of the system at time step $t$, and #process-noise-covariance-(at: $t$) be the covariance matrix representing our uncertainty about the system dynamics#footnote[For example, we may assume for simplicity that the state follows the constant velocity model, but we're not exactly sure of it.]. Because our belief is a Gaussian distribution, we can directly write the transformation of the belief through the system dynamics (the prediction) in closed form as:

  #align(
    center,
    $#mean-state-single-(at: $t$, prediction: true) = #state-transition-matrix-(at: $t$) #mean-state-single-(at: $t-1$) + #control-input-matrix-(at: $t$) #input-single-(at: $t$), quad #covariance-state-single-(at: $t$, prediction: true) = #state-transition-matrix-(at: $t$) #covariance-state-single-(at: $t-1$) #state-transition-matrix-(at: $t$) ^top + #process-noise-covariance-(at: $t$)$,
  )

  *Update Step:*

  Let #observation-matrix-(at: $t$) be the observation matrix that describes how the true state maps to the observed state, and #observation-noise-covariance-(at: $t$) be the covariance matrix representing our uncertainty about the observation#footnote[For example, we may assume that the position and heading measurements are noisy.]. The new observation at time step $t$ is given by $#observed-state-single-(at: $t$) = #observation-matrix-(at: $t$) #state-single _t + #observation-noise-(at: $t$)$ with $#observation-noise-(at: $t$) := #gaussian-($0$, observation-noise-covariance-(at: $t$))$. In order to compute the updated belief, we first need to compute the *Kalman gain* as:

  #align(
    center,
    $#kalman-gain-(at: $t$) = #covariance-state-single-(at: $t$, prediction: true) #observation-matrix-(at: $t$) ^top (#observation-matrix-(at: $t$) #covariance-state-single-(at: $t$, prediction: true) #observation-matrix-(at: $t$) ^top + #observation-noise-covariance-(at: $t$))^(-1)$,
  )

  Then, we can compute the updated belief $#belief-state-single-(at: $t$) := #gaussian-(mean-state-single-(at: $t$), covariance-state-single-(at: $t$))$ as:

  #align(
    center,
    $#mean-state-single-(at: $t$) = #mean-state-single-(at: $t$, prediction: true) + #kalman-gain-(at: $t$) (#observed-state-single-(at: $t$) - #observation-matrix-(at: $t$) #mean-state-single-(at: $t$, prediction: true)), quad
      #covariance-state-single-(at: $t$) = (I - #kalman-gain-(at: $t$) #observation-matrix-(at: $t$)) #covariance-state-single-(at: $t$, prediction: true)$,
  )

  The dimensions of the matrices and vectors in the above equations are as follows:
  - $#state-single, #mean-state-single in bb(R)^#state-dimension, quad #input-single in bb(R)^#control-dimension, quad #observed-state-single in bb(R)^#observation-dimension$
  - $#covariance-state-single, #state-transition-matrix, #process-noise-covariance in bb(R)^(#state-dimension times #state-dimension), quad #control-input-matrix in bb(R)^(#state-dimension times #control-dimension), quad #observation-matrix in bb(R)^(#observation-dimension times #state-dimension), quad #observation-noise-covariance in bb(R)^(#observation-dimension times #observation-dimension), quad #kalman-gain in bb(R)^(#state-dimension times #observation-dimension)$
]

Formulating this as an algorithm, we get:

#algorithm(title: "Kalman Filter State Estimation")[
  #pseudocode-list(hooks: .5em)[
    + *Given:*
      - Current belief #mean-state-single-(at: $t-1$), #covariance-state-single-(at: $t-1$), System dynamics #state-transition-matrix-(at: $t$), #control-input-matrix-(at: $t$)
      - Process noise #process-noise-covariance-(at: $t$), Observation noise #observation-noise-covariance-(at: $t$)
      - Observation matrix #observation-matrix-(at: $t$), observation $#observed-state-single-(at: $t$)$.

    + *Predict:*
      + $#mean-state-single-(at: $t$, prediction: true) arrow.l #state-transition-matrix-(at: $t$) #mean-state-single-(at: $t-1$) + #control-input-matrix-(at: $t$) #input-single-(at: $t$)$ #h(1fr) ➤ Predict the mean state
      + $#covariance-state-single-(at: $t$, prediction: true) arrow.l #state-transition-matrix-(at: $t$) #covariance-state-single-(at: $t-1$) #state-transition-matrix-(at: $t$) ^top + #process-noise-covariance-(at: $t$)$ #h(1fr) ➤ Predict the state covariance
    + *Update:*
      + $#kalman-gain-(at: $t$) arrow.l #covariance-state-single-(at: $t$, prediction: true) #observation-matrix-(at: $t$) ^top (#observation-matrix-(at: $t$) #covariance-state-single-(at: $t$, prediction: true) #observation-matrix-(at: $t$) ^top + #observation-noise-covariance-(at: $t$))^(-1)$ #h(1fr) ➤ Compute the Kalman gain
      + $#mean-state-single-(at: $t$) arrow.l #mean-state-single-(at: $t$, prediction: true) + #kalman-gain-(at: $t$) (#observed-state-single-(at: $t$) - #observation-matrix-(at: $t$) #mean-state-single-(at: $t$, prediction: true))$ #h(1fr) ➤ Update the mean state
      + $#covariance-state-single-(at: $t$) arrow.l (I - #kalman-gain-(at: $t$) #observation-matrix-(at: $t$)) #covariance-state-single-(at: $t$, prediction: true)$ #h(1fr) ➤ Update the state covariance
      - *return* $#belief-state-single-(at: $t$) = #gaussian-(mean-state-single-(at: $t$), covariance-state-single-(at: $t$))$.
  ]
]

*Caveats:*
1. The above formulation assumes the belief is Gaussian, and
2. the system dynamics are linear.

The second assumption often does not hold for mobile robotics applications (e.g. automated driving). For the problem of nonlinear dynamics, we can use one of the extensions of the Kalman filter: the *Extended Kalman Filter* or the *Unscented Kalman Filter*#footnote[There are numerous other methods that handle more general problems, e.g. the particle filter].

=== Example: Velocity from Position

#let estimation-example-entry(body, title: "Item") = [
  - #title:
  #align(left, pad(left: 0.75em, body))
]

For estimating the linear velocity of a moving obstacle using the Kalman filter, we can use the constant velocity model described in @cv-model. The state and matrices are defined as follows (time indices omitted for brevity):

#estimation-example-entry(title: [*State*])[
  $#mean-state-single-() := vec(x, y, v_x, v_y, delim: "["), quad #covariance-state-single-() := mat(
      sigma_x^2, rho_(x y), rho_(x v_x), rho_(x v_y), ;
      rho_(x y), sigma_y^2, rho_(y v_x), rho_(y v_y), ;
      rho_(x v_x), rho_(y v_x), sigma_(v_x)^2, rho_(v_x v_y), ;
      rho_(x v_y), rho_(y v_y), rho_(v_x v_y), sigma_(v_y)^2, ;
      delim: "["
    )$
]

#estimation-example-entry(title: [*Noise Covariances*])[
  $#process-noise-covariance-() := "diag"(0, quad 0, quad sigma_(a_x)^2 #delta-t^2, quad sigma_(a_y)^2 #delta-t^2), quad #observation-noise-covariance-() := "diag"(sigma_(z_x)^2, quad sigma_(z_y)^2)$

  where $sigma_(a_x)^2$ and $sigma_(a_y)^2$ represent uncertainty about potential accelerations. $sigma_(z_x)^2$ and $sigma_(z_y)^2$ can be set to zero if we assume perfect position measurements.
]

#estimation-example-entry(title: [*Observation Matrix*])[
  $#observation-matrix-() := mat(
      1, 0, 0, 0;
      0, 1, 0, 0;
      delim: "["
    )$
]

Although this approach works, we cannot use it when assuming that the obstacle follows the kinematic bicycle model, or any nonlinear model in general#footnote[We could compute the mean of the speed with the KF estimate of the linear velocity, but not the covariance.].

== Extended Kalman Filter (EKF)

#let observation-function-(..args) = function(name: $h$, ..args)
#let observation-function = observation-function-()
#let jacobian-(
  of: state-transition-function,
  wrt: state-single,
  at: none,
) = function(
  name: [$(partial of) / (partial wrt)$ #if at != none {
      $|_#at$
    } else { }],
  at: at,
)
#let state-transition-jacobian-(wrt: state-single, at: none) = jacobian-(
  of: state-transition-function,
  wrt: state-single,
  at: at,
)
#let observation-jacobian-(wrt: state-single, at: none) = jacobian-(
  of: observation-function,
  wrt: state-single,
  at: at,
)

For nonlinear dynamics, we can use the Extended Kalman Filter (EKF). The only difference from the standard KF is that we linearize the nonlinear dynamics with a first-order Taylor expansion at each time step, and use this approximation to propagate the mean and covariance of the belief.

#definition(title: [Extended Kalman Filter State Estimate @Thrun2005])[
  Let #state-transition-function-(state-single, input-single) be the system dynamics and #observation-function-(state-single) be the observation function. Then the linearized state transition and observation matrices at time step $t$ are:

  #align(
    center,
    $#state-transition-matrix-(at: $t$) = #state-transition-jacobian-(wrt: state-single, at: $#state-single = #mean-state-single-(at: $t-1$), #input-single = #input-single-(at: $t$)$), quad
      #observation-matrix-(at: $t$) = #observation-jacobian-(wrt: state-single, at: $#state-single = #mean-state-single-(at: $t-1$), #input-single = #input-single-(at: $t$)$)$,
  )

  Note that the approximation at time step $t$ is done around the predicted mean state at time step $t-1$, but uses the (assumed) control input at time step $t$.
]

The EKF algorithm is then formulated similarly to KF, but with the above linearized matrices:

#algorithm(title: "Extended Kalman Filter State Estimation")[
  #pseudocode-list(hooks: .5em)[
    + *Given:*
      - Current belief #mean-state-single-(at: $t-1$), #covariance-state-single-(at: $t-1$)
      - System dynamics #state-transition-function-(state-single, input-single), observation function #observation-function-(state-single)
      - Process noise #process-noise-covariance-(at: $t$), Observation noise #observation-noise-covariance-(at: $t$)
      - Control input $#input-single-(at: $t$)$, observation $#observed-state-single-(at: $t$)$.

    + *Predict:*
      + $#state-transition-matrix-(at: $t$) arrow.l #state-transition-jacobian-(wrt: state-single, at: $#state-single = #mean-state-single-(at: $t-1$), #input-single = #input-single-(at: $t$)$)$ #h(1fr) ➤ Jacobian of dynamics w.r.t. state
      + $#mean-state-single-(at: $t$, prediction: true) arrow.l #state-transition-function-(mean-state-single-(at: $t-1$), $#input-single-(at: $t$)$)$ #h(1fr) ➤ Predict mean using system dynamics
      + $#covariance-state-single-(at: $t$, prediction: true) arrow.l #state-transition-matrix-(at: $t$) #covariance-state-single-(at: $t-1$) #state-transition-matrix-(at: $t$) ^top + #process-noise-covariance-(at: $t$)$ #h(1fr) ➤ Predict covariance using linearized dynamics

    + *Update:*
      + $#observation-matrix-(at: $t$) arrow.l #observation-jacobian-(wrt: state-single, at: $#state-single = #mean-state-single-(at: $t$, prediction: true)$)$ #h(1fr) ➤ Jacobian of observation w.r.t. state
      + $#kalman-gain-(at: $t$) arrow.l #covariance-state-single-(at: $t$, prediction: true) #observation-matrix-(at: $t$) ^top (#observation-matrix-(at: $t$) #covariance-state-single-(at: $t$, prediction: true) #observation-matrix-(at: $t$) ^top + #observation-noise-covariance-(at: $t$))^(-1)$ #h(1fr) ➤ Compute Kalman gain
      + $#mean-state-single-(at: $t$) arrow.l #mean-state-single-(at: $t$, prediction: true) + #kalman-gain-(at: $t$) (#observed-state-single-(at: $t$) - #observation-function-(mean-state-single-(at: $t$, prediction: true)))$ #h(1fr) ➤ Update mean
      + $#covariance-state-single-(at: $t$) arrow.l (I - #kalman-gain-(at: $t$) #observation-matrix-(at: $t$)) #covariance-state-single-(at: $t$, prediction: true)$ #h(1fr) ➤ Update covariance
      - *return* $#belief-state-single-(at: $t$) = #gaussian-(mean-state-single-(at: $t$), covariance-state-single-(at: $t$))$.
  ]
]

=== Example: Acceleration & Steering Angle from Pose

For estimating the acceleration and steering angle of a moving obstacle using the EKF, we can use the CSAA model (@csaa-model). The state, dynamics, and matrices are defined as follows:

#estimation-example-entry(title: [*State*])[
  $#mean-state-single-() := vec(x, y, theta, v, a, delta, delim: "["), quad #covariance-state-single-() := mat(
      sigma_x^2, dots.c, rho_(x delta), ;
      dots.v, dots.down, dots.v, ;
      rho_(x delta), dots.c, sigma_delta^2, ;
      delim: "["
    )$
]

#estimation-example-entry(title: [*Jacobian of System Dynamics*])[
  $#state-transition-matrix-() := mat(
      1, 0, -v sin(theta) #delta-t, cos(theta) #delta-t, 0, 0, ;
      0, 1, v cos(theta) #delta-t, sin(theta) #delta-t, 0, 0, ;
      0, 0, 1, tan(delta) / L dot #delta-t, 0, v / L sec(delta)^2 #delta-t, ;
      0, 0, 0, 1, #delta-t, 0, ;
      0, 0, 0, 0, 1, 0, ;
      0, 0, 0, 0, 0, 1, ;
      delim: "["
    )$
]

#estimation-example-entry(title: [*Observation Function and Jacobian*])[
  $#observation-function-(state-single) := vec(x, y, theta, delim: "["), quad #observation-matrix-() := mat(
      1, 0, 0, 0, 0, 0;
      0, 1, 0, 0, 0, 0;
      0, 0, 1, 0, 0, 0;
      delim: "["
    )$
]

#estimation-example-entry(title: [*Noise Covariances*])[
  $#process-noise-covariance-() := "diag"(0, quad 0, quad 0, quad 0, quad sigma_(dot(a))^2 #delta-t^2, quad sigma_(dot(delta))^2 #delta-t^2)$
]

=== Example: Linear & Angular Velocity from Pose

Alternatively, we can use the constant turn rate and velocity (CTRV) model (@ctrv-model).

#estimation-example-entry(title: [*State*])[
  $#mean-state-single-() := vec(x, y, theta, v, omega, delim: "["), quad #covariance-state-single-() := mat(
      sigma_x^2, dots.c, rho_(x omega), ;
      dots.v, dots.down, dots.v, ;
      rho_(x omega), dots.c, sigma_omega^2, ;
      delim: "["
    )$
]

#estimation-example-entry(title: [*Jacobian of System Dynamics*])[
  $#state-transition-matrix-() := mat(
      1, 0, -v sin(theta) #delta-t, cos(theta) #delta-t, 0, ;
      0, 1, v cos(theta) #delta-t, sin(theta) #delta-t, 0, ;
      0, 0, 1, 0, #delta-t, ;
      0, 0, 0, 1, 0, ;
      0, 0, 0, 0, 1, ;
      delim: "["
    )$
]

#estimation-example-entry(title: [*Observation Function and Jacobian*])[
  $#observation-function-(state-single) := vec(x, y, theta, delim: "["), quad #observation-matrix-() := mat(
      1, 0, 0, 0, 0;
      0, 1, 0, 0, 0;
      0, 0, 1, 0, 0;
      delim: "["
    )$
]

#estimation-example-entry(title: [*Noise Covariances*])[
  $#process-noise-covariance-() := "diag"(0, quad 0, quad 0, quad sigma_a^2 #delta-t^2, quad sigma_(dot(omega))^2 #delta-t^2), quad #observation-noise-covariance-() := "diag"(sigma_(z_x)^2, quad sigma_(z_y)^2, quad sigma_(z_theta)^2)$
]

== Unscented Kalman Filter (UKF)

#let sigma-point-(index, prediction: false) = {
  $#{ if prediction { $overline(chi)$ } else { $chi$ } } _(#index)$
}
#let transformed-sigma-point-(index) = $cal(Y)_(#index)$
#let predicted-observation-(index) = $overline(cal(Z))_(#index)$
#let sigma-mean-weight-(index) = $w_(m,#index)$
#let sigma-covariance-weight-(index) = $w_(c,#index)$
#let mean-predicted-observation-(at: none) = function(
  name: $overline(sans(z))$,
  sub: at,
)
#let innovation-covariance(at: none) = function(
  name: $sans(S)$,
  sub: at,
)
#let cross-covariance-(at: none) = function(
  name: $overline(Sigma)$,
  sub: if at != none { $#state-single, #observed-state-single, #at$ } else {
    $#state-single, #observed-state-single$
  },
)
#let primary-scaling-parameter = $lambda$
#let sigma-point-spread = $alpha$
#let prior-knowledge-parameter = $beta$
#let state-dimension-short = $n$

#let secondary-scaling-parameter-note = footnote[The typical formula for the scaling parameter is $#primary-scaling-parameter = #sigma-point-spread^2 (#state-dimension-short + kappa) - #state-dimension-short$, where $kappa$ is a secondary scaling parameter that is often set to zero. For simplicity, we will ignore it here.]

The UKF is another extension of KF that can handle nonlinear dynamics. Instead of linearizing the dynamics to approximate it as in EKF, the UKF tries to approximate the transformed Gaussian distribution.

#definition(title: [Unscented Transform @Thrun2005])[
  Let #state-single be a random variable with mean #mean-state-single and covariance #covariance-state-single. The unscented transform approximates the distribution of a transformed random variable #state-transition-function-(state-single) with the following steps:

  1. Generate a set of *sigma points* around and including the mean of the original distribution.
  2. Propagate each sigma point through the nonlinear function #state-transition-function.
  3. Compute the mean and covariance of the transformed sigma points to reconstruct #state-transition-function-(state-single).

  *Generating Sigma Points:*

  Let $n$ be the dimension of the state #state-single, and #sigma-point-spread be a parameter that determines how spread out the sigma points are. The $2n + 1$ sigma points are generated#secondary-scaling-parameter-note as follows:

  $\
    #sigma-point-(0) = #mean-state-single\
    #sigma-point-($i$) = #mean-state-single + (sqrt((#state-dimension-short + #primary-scaling-parameter) #covariance-state-single))_i "for" i = 1, ..., #state-dimension-short\
    #sigma-point-($i$) = #mean-state-single - (sqrt((#state-dimension-short + #primary-scaling-parameter) #covariance-state-single))_(i-#state-dimension-short) "for" i = #state-dimension-short+1, ..., 2#state-dimension-short\
    #primary-scaling-parameter := (#sigma-point-spread ^2 - 1) #state-dimension-short$

  Where $( dot )_i$ denotes the $i$-th column of the matrix.

  Furthermore, let #prior-knowledge-parameter be a parameter that can be used to incorporate prior knowledge about the distribution (set $#prior-knowledge-parameter = 2$ if the distribution is Gaussian). Then, each sigma point is assigned a weight for computing the mean and covariance of the transformed distribution. The weights are computed as:

  $\
    #sigma-mean-weight-(0) = #primary-scaling-parameter / (#state-dimension-short + #primary-scaling-parameter), quad #sigma-covariance-weight-(0) = #primary-scaling-parameter / (#state-dimension-short + #primary-scaling-parameter) + (1 - #sigma-point-spread^2 + #prior-knowledge-parameter)\
    #sigma-mean-weight-($i$) = #sigma-covariance-weight-($i$) = 1 / (2(#state-dimension-short + #primary-scaling-parameter)) "for" i = 1, ..., 2#state-dimension-short$

  *Propagating Sigma Points:*

  For each sigma point #sigma-point-($i$), we compute the transformed sigma point as $#transformed-sigma-point-($i$) = #state-transition-function-(sigma-point-($i$))$.

  *Reconstructing the Transformed Distribution:*

  Finally, we can compute the mean and covariance of the transformed distribution as:

  $\
    #mean-state-single-(prediction: true) = sum_(i=0)^(2#state-dimension-short) #sigma-mean-weight-($i$) #transformed-sigma-point-($i$), quad #covariance-state-single-(prediction: true) = sum_(i=0)^(2#state-dimension-short) #sigma-covariance-weight-($i$) (#transformed-sigma-point-($i$) - #mean-state-single-(prediction: true))(#transformed-sigma-point-($i$) - #mean-state-single-(prediction: true))^top$
]

We now formulate the UKF algorithm for state estimation similarly to KF and EKF:

#algorithm(title: "Unscented Kalman Filter State Estimation")[
  #pseudocode-list(hooks: .5em)[
    + *Given:*
      - Current belief #mean-state-single-(at: $t-1$), #covariance-state-single-(at: $t-1$)
      - System dynamics #state-transition-function-(state-single, input-single), observation function #observation-function-(state-single)
      - Process noise #process-noise-covariance-(at: $t$), Observation noise #observation-noise-covariance-(at: $t$)
      - Control input $#input-single-(at: $t$)$, observation $#observed-state-single-(at: $t$)$.

    + *Define compute-sigma-point(i):*
      + *if* i = 0 *then* $#sigma-point-(0) = #mean-state-single-(at: $t-1$)$
      + *else if* $1 <= i <= #state-dimension-short$ *then* $#sigma-point-($i$) = #mean-state-single-(at: $t-1$) + (sqrt((#state-dimension-short + #primary-scaling-parameter) #covariance-state-single-(at: $t-1$)))_i$
      + *else* $#sigma-point-($i$) = #mean-state-single-(at: $t-1$) - (sqrt((#state-dimension-short + #primary-scaling-parameter) #covariance-state-single-(at: $t-1$)))_(i - #state-dimension-short)$

    + *Define compute-observation-sigma-point(i):*
      + *if* i = 0 *then* $#sigma-point-(0, prediction: true) = #mean-state-single-(at: $t$, prediction: true)$
      + *else if* $1 <= i <= #state-dimension-short$ *then* $#sigma-point-($i$, prediction: true) = #mean-state-single-(at: $t$, prediction: true) + (sqrt((#state-dimension-short + #primary-scaling-parameter) #covariance-state-single-(at: $t$, prediction: true)))_i$
      + *else* $#sigma-point-($i$, prediction: true) = #mean-state-single-(at: $t$, prediction: true) - (sqrt((#state-dimension-short + #primary-scaling-parameter) #covariance-state-single-(at: $t$, prediction: true)))_(i - #state-dimension-short)$

    + *Define compute-sigma-weights(i):*
      + *if* i = 0 *then* $#sigma-mean-weight-(0) = #primary-scaling-parameter / (#state-dimension-short + #primary-scaling-parameter), quad #sigma-covariance-weight-(0) = #primary-scaling-parameter / (#state-dimension-short + #primary-scaling-parameter) + (1 - #sigma-point-spread^2 + #prior-knowledge-parameter)$
      + *else* $#sigma-mean-weight-($i$) = #sigma-covariance-weight-($i$) = 1 / (2(#state-dimension-short + #primary-scaling-parameter))$

    + *Predict:*
      + *for* i = 0 *to* 2#state-dimension-short *do*
        + $#sigma-point-($i$) arrow.l "compute-sigma-point"($i$)$
        + $#sigma-mean-weight-($i$), #sigma-covariance-weight-($i$) arrow.l "compute-sigma-weights"($i$)$
        + $#transformed-sigma-point-($i$) arrow.l #state-transition-function-(sigma-point-($i$), $#input-single-(at: $t$)$)$ #h(1fr) ➤ Propagate sigma point
      + $#mean-state-single-(at: $t$, prediction: true) arrow.l sum_(i=0)^(2#state-dimension-short) #sigma-mean-weight-($i$) #transformed-sigma-point-($i$)$ #h(1fr) ➤ Compute predicted mean
      + $#covariance-state-single-(at: $t$, prediction: true) arrow.l sum_(i=0)^(2#state-dimension-short) #sigma-covariance-weight-($i$) (#transformed-sigma-point-($i$) - #mean-state-single-(at: $t$, prediction: true))(#transformed-sigma-point-($i$) - #mean-state-single-(at: $t$, prediction: true))^top + #process-noise-covariance-(at: $t$)$ #h(1fr) ➤ Compute predicted covariance

    + *Update:*
      + *for* i = 0 *to* 2#state-dimension-short *do*
        + $#sigma-point-($i$, prediction: true) arrow.l "compute-observation-sigma-point"($i$)$ #h(1fr) ➤ Sigma point for observation
        + $#predicted-observation-($i$) arrow.l #observation-function-(sigma-point-($i$, prediction: true))$ #h(1fr) ➤ Predict observation for sigma point
      + $#mean-predicted-observation-(at: $t$) arrow.l sum_(i=0)^(2#state-dimension-short) #sigma-mean-weight-($i$) #predicted-observation-($i$)$ #h(1fr) ➤ Predict observation mean
      + $#innovation-covariance(at: $t$) arrow.l sum_(i=0)^(2#state-dimension-short) #sigma-covariance-weight-($i$) (#predicted-observation-($i$) - #mean-predicted-observation-(at: $t$))(#predicted-observation-($i$) - #mean-predicted-observation-(at: $t$))^top + #observation-noise-covariance-(at: $t$)$ #h(1fr) ➤ Predict innovation covariance
      + $#cross-covariance-(at: $t$) arrow.l sum_(i=0)^(2#state-dimension-short) #sigma-covariance-weight-($i$) (#sigma-point-($i$, prediction: true) - #mean-state-single-(at: $t$, prediction: true))(#predicted-observation-($i$) - #mean-predicted-observation-(at: $t$))^top$ #h(1fr) ➤ Compute cross covariance
      + $#kalman-gain-(at: $t$) arrow.l #cross-covariance-(at: $t$) (#innovation-covariance(at: $t$))^(-1)$ #h(1fr) ➤ Compute Kalman gain
      + $#mean-state-single-(at: $t$) arrow.l #mean-state-single-(at: $t$, prediction: true) + #kalman-gain-(at: $t$) (#observed-state-single-(at: $t$) - #mean-predicted-observation-(at: $t$))$ #h(1fr) ➤ Update mean
      + $#covariance-state-single-(at: $t$) arrow.l #covariance-state-single-(at: $t$, prediction: true) - #kalman-gain-(at: $t$) #innovation-covariance(at: $t$) #kalman-gain-(at: $t$) ^top$ #h(1fr) ➤ Update covariance
      - *return* $#belief-state-single-(at: $t$) = #gaussian-(mean-state-single-(at: $t$), covariance-state-single-(at: $t$))$.
  ]
]

The above algorithm is slightly more complex, since it accounts for nonlinear observation functions as well. However, if the observation function is known to be linear, then we can use the same update step as in the standard KF:

#algorithm(title: "UKF State Estimation with Linear Observations")[
  #pseudocode-list(hooks: .5em)[
    + *Given:*
      - Current belief #mean-state-single-(at: $t-1$), #covariance-state-single-(at: $t-1$)
      - System dynamics #state-transition-function-(state-single, input-single), observation matrix #observation-matrix-(at: $t$)
      - Process noise #process-noise-covariance-(at: $t$), Observation noise #observation-noise-covariance-(at: $t$)
      - Control input $#input-single-(at: $t$)$, observation $#observed-state-single-(at: $t$)$.

    + *Predict:*
      + same as before...

    + *Update:*
      + $#kalman-gain-(at: $t$) arrow.l #covariance-state-single-(at: $t$, prediction: true) #observation-matrix-(at: $t$) ^top (#observation-matrix-(at: $t$) #covariance-state-single-(at: $t$, prediction: true) #observation-matrix-(at: $t$) ^top + #observation-noise-covariance-(at: $t$))^(-1)$ #h(1fr) ➤ Compute Kalman gain
      + $#mean-state-single-(at: $t$) arrow.l #mean-state-single-(at: $t$, prediction: true) + #kalman-gain-(at: $t$) (#observed-state-single-(at: $t$) - #observation-matrix-(at: $t$) #mean-state-single-(at: $t$, prediction: true))$ #h(1fr) ➤ Update mean
      + $#covariance-state-single-(at: $t$) arrow.l (I - #kalman-gain-(at: $t$) #observation-matrix-(at: $t$)) #covariance-state-single-(at: $t$, prediction: true)$ #h(1fr) ➤ Update covariance
      - *return* $#belief-state-single-(at: $t$) = #gaussian-(mean-state-single-(at: $t$), covariance-state-single-(at: $t$))$.
  ]
]

= Incorporating Uncertainty into Predictions

When using motion prediction models, such as the curvilinear models described in @curvilinear-models, we make assumptions about the future motion of obstacles that we know may not hold exactly. Furthermore, sensor noise and inaccuracies in state estimation can also lead to uncertainty about the current state of obstacles. A simple way to incorporate information on uncertainty into predictions is to use the prediction steps of the KF algorithms.

== KF Covariance Propagation

We can propagate the uncertainty of the state estimate through time by repeating the prediction step of the KF over the prediction horizon. Afterwards, we can combine these covariances with the predicted means (e.g. from the CV model) to obtain a distribution over future states.

#algorithm(title: "KF Covariance Propagation for Linear Models")[
  #pseudocode-list(hooks: .5em)[
    + *Given:*
      - Initial state estimate #mean-state-single-(at: $0$), #covariance-state-single-(at: $0$)
      - State transition matrix #state-transition-matrix-()
      - Process noise #process-noise-covariance-()
      - Prediction horizon #horizon

    + *Initialize:*
      + $#mean-state-single-(at: 0, prediction: true) arrow.l #mean-state-single-(at: 0)$
      + $#covariance-state-single-(at: 0, prediction: true) arrow.l #covariance-state-single-(at: 0)$

    + *Predict:*
      + *for* $tau = 1$ *to* #horizon *do*
        + $#mean-state-single-(at: $tau$, prediction: true) arrow.l #state-transition-matrix-() #mean-state-single-(at: $tau - 1$, prediction: true)$ #h(1fr) ➤ Predict mean
        + $#covariance-state-single-(at: $tau$, prediction: true) arrow.l #state-transition-matrix-() #covariance-state-single-(at: $tau - 1$, prediction: true) #state-transition-matrix-()^top + #process-noise-covariance-()$ #h(1fr) ➤ Propagate covariance

      - *return* #mean-state-single-(prediction: true), #covariance-state-single-(prediction: true)
  ]
]

== EKF Covariance Propagation

To add uncertainty information to the predictions of the nonlinear models (e.g. CSAA, CTRV), we can use the prediction step from EKF. The idea is the same as for the covariance propagation of the linear models, but we need to compute the Jacobian of the nonlinear dynamics at each time step to propagate the covariance.

#algorithm(title: "EKF Covariance Propagation for Nonlinear Models")[
  #pseudocode-list(hooks: .5em)[
    + *Given:*
      - Initial state estimate #mean-state-single-(at: $0$), #covariance-state-single-(at: $0$)
      - System dynamics #state-transition-function-(state-single)
      - Process noise #process-noise-covariance-()
      - Prediction horizon #horizon

    + *Initialize:*
      + $#mean-state-single-(at: 0, prediction: true) arrow.l #mean-state-single-(at: 0)$
      + $#covariance-state-single-(at: 0, prediction: true) arrow.l #covariance-state-single-(at: 0)$

    + *Predict:*
      + *for* $tau = 1$ *to* #horizon *do*
        + $#state-transition-matrix-(at: $tau$) arrow.l #state-transition-jacobian-(wrt: state-single, at: $#state-single = #mean-state-single-(at: $tau - 1$, prediction: true)$)$ #h(1fr) ➤ Compute Jacobian at current estimate
        + $#mean-state-single-(at: $tau$, prediction: true) arrow.l #state-transition-function-(mean-state-single-(at: $tau - 1$, prediction: true))$ #h(1fr) ➤ Predict mean using system dynamics
        + $#covariance-state-single-(at: $tau$, prediction: true) arrow.l #state-transition-matrix-(at: $tau$) #covariance-state-single-(at: $tau - 1$, prediction: true) #state-transition-matrix-(at: $tau$)^top + #process-noise-covariance-()$ #h(1fr) ➤ Predict covariance using Jacobian

      - *return* #mean-state-single-(prediction: true), #covariance-state-single-(prediction: true)
  ]
]

== UKF Covariance Propagation

Alternatively, the Unscented transform (prediction step of UKF) can be used if higher accuracy is desired for the nonlinear models, or if the Jacobians are difficult to compute.

#algorithm(title: "UKF Covariance Propagation for Nonlinear Models")[
  #pseudocode-list(hooks: .5em)[
    + *Given:*
      - Initial state estimate #mean-state-single-(at: $0$), #covariance-state-single-(at: $0$)
      - System dynamics #state-transition-function-(state-single)
      - Process noise #process-noise-covariance-()
      - Prediction horizon #horizon
      - Sigma point spread $#sigma-point-spread = 10^(-3)$, prior knowledge $#prior-knowledge-parameter = 2$

    + *Initialize:*
      + $#state-dimension-short arrow.l dim(#mean-state-single-(at: $0$))$
      + $#primary-scaling-parameter arrow.l (#sigma-point-spread^2 - 1) #state-dimension-short$
      + $#mean-state-single-(at: 0, prediction: true) arrow.l #mean-state-single-(at: 0)$
      + $#covariance-state-single-(at: 0, prediction: true) arrow.l #covariance-state-single-(at: 0)$

    + *Define compute-sigma-point(i, tau):*
      + *if* i = 0 *then* $#sigma-point-($0,tau$) = #mean-state-single-(at: $tau - 1$, prediction: true)$
      + *else if* $1 <= i <= #state-dimension-short$ *then* $#sigma-point-($i,tau$) = #mean-state-single-(at: $tau - 1$, prediction: true) + (sqrt((#state-dimension-short + #primary-scaling-parameter) #covariance-state-single-(at: $tau - 1$, prediction: true)))_i$
      + *else* $#sigma-point-($i,tau$) = #mean-state-single-(at: $tau - 1$, prediction: true) - (sqrt((#state-dimension-short + #primary-scaling-parameter) #covariance-state-single-(at: $tau - 1$, prediction: true)))_(i - #state-dimension-short)$

    + *Define compute-sigma-weights(i):*
      + *if* i = 0 *then* $#sigma-mean-weight-(0) = #primary-scaling-parameter / (#state-dimension-short + #primary-scaling-parameter), quad #sigma-covariance-weight-(0) = #primary-scaling-parameter / (#state-dimension-short + #primary-scaling-parameter) + (1 - #sigma-point-spread^2 + #prior-knowledge-parameter)$
      + *else* $#sigma-mean-weight-($i$) = #sigma-covariance-weight-($i$) = 1 / (2(#state-dimension-short + #primary-scaling-parameter))$

    + *Predict:*
      + *for* i = 0 *to* 2#state-dimension-short *do* #h(1fr) ➤ Compute sigma weights (once)
        + $#sigma-mean-weight-($i$), #sigma-covariance-weight-($i$) arrow.l "compute-sigma-weights"($i$)$

      + *for* $tau$ = 1 *to* #horizon
        + *for* i = 0 *to* $2#state-dimension-short$ *do*
          + $#sigma-point-($i,tau$) arrow.l "compute-sigma-point"($i$,$tau$)$ #h(1fr) ➤ Compute sigma points
          + $#transformed-sigma-point-($i,tau$) arrow.l #state-transition-function-(sigma-point-($i,tau$))$ #h(1fr) ➤ Propagate sigma points
        + $#mean-state-single-(at: $tau$, prediction: true) arrow.l sum_(i=0)^(2#state-dimension-short) #sigma-mean-weight-($i$) #transformed-sigma-point-($i,tau$)$ #h(1fr) ➤ Predict mean
        + $#covariance-state-single-(at: $tau$, prediction: true) arrow.l sum_(i=0)^(2#state-dimension-short) #sigma-covariance-weight-($i$) (#transformed-sigma-point-($i,tau$) - #mean-state-single-(at: $tau$, prediction: true))(#transformed-sigma-point-($i,tau$) - #mean-state-single-(at: $tau$, prediction: true))^top + #process-noise-covariance-()$ #h(1fr) ➤ Predict covariance
        - *return* #mean-state-single-(prediction: true), #covariance-state-single-(prediction: true)
  ]
]

= Computational Framework

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

#pagebreak()

#bibliography("references.bib")
