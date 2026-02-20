# costs

Cost functions evaluate the quality of sampled rollouts. The total cost for a rollout is the sum of per-timestep costs across all active cost components.

## Tracking Costs

### Contouring Cost

Penalizes lateral (orthogonal) deviation from the reference trajectory [^1]:

$$
J_c = k_c \, e_c^2, \quad e_c = \sin(\theta_\phi)(x - x_\phi) - \cos(\theta_\phi)(y - y_\phi)
$$

where $(x_\phi, y_\phi, \theta_\phi)$ is the reference point at path parameter $\phi$.

::: faran.costs.basic.NumPyContouringCost
    options:
      show_root_heading: true
      heading_level: 4
      members:
        - create
        - error

### Lag Cost

Penalizes longitudinal (tangential) deviation from the reference point [^1]:

$$
J_l = k_l \, e_l^2, \quad e_l = -\cos(\theta_\phi)(x - x_\phi) - \sin(\theta_\phi)(y - y_\phi)
$$

::: faran.costs.basic.NumPyLagCost
    options:
      show_root_heading: true
      heading_level: 4
      members:
        - create
        - error

### Progress Cost

Rewards forward motion along the reference trajectory [^1]:

$$
J_p = -k_p \, \dot{\phi} \, \Delta t
$$

where $\dot{\phi}$ is the virtual path velocity.

[^1]: A. Liniger, A. Domahidi, M. Morari, "Optimization-based Autonomous Racing of 1:43 Scale RC Cars," Optimal Control Applications and Methods, 2015.

::: faran.costs.basic.NumPyProgressCost
    options:
      show_root_heading: true
      heading_level: 4
      members:
        - create

## Safety Costs

### Collision Cost

Hinge-loss collision avoidance cost [^2] that activates when any vehicle part is closer than a threshold $d_0$ to an obstacle:

$$
J_{\text{col}} = \sum_{i=1}^{V} \begin{cases}
k_{\text{col}}(d_0 - d_i) & \text{if } d_i < d_0 \\
0 & \text{otherwise}
\end{cases}
$$

where $d_i$ is the signed distance between vehicle part $i$ and the nearest obstacle, and $V$ is the number of vehicle parts.

[^2]: J. Schulman et al., "Finding Locally Optimal, Collision-Free Trajectories with Sequential Convex Optimization," RSS, 2013.

::: faran.costs.collision.basic.NumPyCollisionCost
    options:
      show_root_heading: true
      heading_level: 4
      members:
        - create

### Boundary Cost

Same hinge-loss formulation as the collision cost, applied to corridor boundary distances instead of obstacle distances.

::: faran.costs.boundary.basic.NumPyBoundaryCost
    options:
      show_root_heading: true
      heading_level: 4
      members:
        - create

## Comfort Costs

### Control Smoothing Cost

Penalizes the rate of change of control inputs between consecutive time steps [^1]:

$$
J_s = \| K_u \, \Delta \mathbf{u}_t \|^2, \quad \Delta \mathbf{u}_t = \mathbf{u}_t - \mathbf{u}_{t-1}
$$

::: faran.costs.basic.NumPyControlSmoothingCost
    options:
      show_root_heading: true
      heading_level: 4
      members:
        - create

### Control Effort Cost

Penalizes large control input magnitudes [^3]:

$$
J_n = K_n \| \mathbf{u}_t \|^2
$$

[^3]: G. Williams, N. Wagener et al., "Information Theoretic MPC for Model-Based Reinforcement Learning," IEEE ICRA, 2017.

::: faran.costs.basic.NumPyControlEffortCost
    options:
      show_root_heading: true
      heading_level: 4
      members:
        - create

## Combining Costs

The `combined` factory creates a cost function that sums all component costs per timestep:

```python
from faran.numpy import costs

total = costs.combined(contouring, lag, progress, collision, smoothing)
```

::: faran.costs.combined.CombinedCost
    options:
      show_root_heading: true
      heading_level: 3

## Distance Functions

Distance extractors compute signed distances between vehicle parts and obstacles. These are used by the collision cost.

### Circle Distance

Computes pairwise signed distances between circular vehicle approximations and circular obstacle approximations [^4].

[^4]: L. Tolksdorf et al., "Fast Collision Probability Estimation for Automated Driving using Multi-circular Shape Approximations," IEEE IV, 2024.

::: faran.costs.distance.circles.basic.NumPyCircleDistanceExtractor
    options:
      show_root_heading: true
      heading_level: 4
      members:
        - create

### SAT Distance

Computes signed distances between convex polygons using the Separating Axis Theorem.

::: faran.costs.distance.sat.basic.NumPySatDistanceExtractor
    options:
      show_root_heading: true
      heading_level: 4
      members:
        - create

## Risk Metrics

When obstacle positions are uncertain, collision costs can be evaluated under different risk measures via the [riskit](https://pypi.org/project/riskit/) library. Available risk metrics include expected value, mean-variance, VaR, CVaR, and entropic risk.

::: faran.costs.risk.base.RisKitRiskMetric
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - create

## CostFunction Protocol

::: faran.types.CostFunction
    options:
      show_root_heading: true
      heading_level: 3
