# model

Dynamical models define the state transition function $f(\mathbf{u}, \mathbf{x})$ used to simulate rollouts during MPPI planning.

## Kinematic Bicycle Model

The kinematic bicycle model [^1] represents a wheeled vehicle with four state variables and two control inputs, discretized via Euler integration:

$$
\begin{gathered}
x_{t+1} = x_t + v_t \cos(\theta_t) \, \Delta t, \quad
y_{t+1} = y_t + v_t \sin(\theta_t) \, \Delta t \\
\theta_{t+1} = \theta_t + \frac{v_t}{L} \tan(\delta_t) \, \Delta t, \quad
v_{t+1} = v_t + a_t \, \Delta t
\end{gathered}
$$

where $L$ is the wheelbase and $\Delta t$ the time step size.

| Component | Variables |
|-----------|-----------|
| State | $[x, y, \theta, v]$ — position, heading, speed |
| Controls | $[a, \delta]$ — acceleration, steering angle |

[^1]: P. Polack et al., "The Kinematic Bicycle Model: A Consistent Model for Planning Feasible Trajectories for Autonomous Vehicles?," IEEE IV, 2017.

::: faran.models.bicycle.basic.NumPyBicycleModel
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - create

::: faran.models.bicycle.accelerated.JaxBicycleModel
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - create

## Unicycle Model

The unicycle model [^2] represents a point robot with direct velocity and angular velocity control:

$$
x_{t+1} = x_t + v_t \cos(\theta_t) \Delta t, \quad
y_{t+1} = y_t + v_t \sin(\theta_t) \Delta t, \quad
\theta_{t+1} = \theta_t + \omega_t \Delta t
$$

| Component | Variables |
|-----------|-----------|
| State | $[x, y, \theta]$ — position, heading |
| Controls | $[v, \omega]$ — linear velocity, angular velocity |

[^2]: G. Oriolo, A. De Luca, M. Vendittelli, "WMR Control via Dynamic Feedback Linearization," IEEE TCST, 2002.

::: faran.models.unicycle.basic.NumPyUnicycleModel
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - create

::: faran.models.unicycle.accelerated.JaxUnicycleModel
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - create

## Integrator Model

An $n$-dimensional integrator: $x_{t+1} = x_t + v_t \Delta t$. Used internally for the MPCC virtual state (path parameter $\phi$), but also available as a general-purpose model.

::: faran.models.integrator.basic.NumPyIntegratorModel
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - create

::: faran.models.integrator.accelerated.JaxIntegratorModel
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - create

## DynamicalModel Protocol

::: faran.types.DynamicalModel
    options:
      show_root_heading: true
      heading_level: 3

## Obstacle Models

Obstacle models are used by predictors to propagate obstacle states forward in time. They follow the same Euler integration as the corresponding dynamical models but operate on observed obstacle states.

::: faran.models.bicycle.basic.NumPyBicycleObstacleModel
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - create

::: faran.models.unicycle.basic.NumPyUnicycleObstacleModel
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - create
