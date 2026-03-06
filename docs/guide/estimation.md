# State Estimation

State estimation recovers unobserved state variables (speed, acceleration, steering angle) and quantifies uncertainty from noisy position/heading observations. In Faran, estimators are primarily used for **obstacle state estimation** — tracking other vehicles and predicting their future motion.

## Why Estimate State?

A typical obstacle detector provides position $(x, y)$ and heading $\theta$ at each time step. But prediction models need additional variables:

- **[Bicycle model](models.md#kinematic-bicycle-model)** needs speed $v$, acceleration $a$, and steering angle $\delta$
- **[Unicycle model](models.md#unicycle-model)** needs speed $v$ and angular velocity $\omega$
- **[Integrator model](models.md#integrator-model)** needs velocity components

Estimation fills in these unobserved variables and provides a covariance matrix that captures how uncertain each state component is. This uncertainty propagates through motion prediction and feeds into [risk-aware collision costs](obstacles.md#risk-aware-collision-cost).

## Available Estimators

### Finite Difference

The simplest approach: estimate derivatives from consecutive observations.

$$
v_t \approx \frac{(x_t - x_{t-1})\cos\theta_t + (y_t - y_{t-1})\sin\theta_t}{\Delta t}
$$

$$
a_t = \frac{v_t - v_{t-1}}{\Delta t}, \quad \delta_t \approx \arctan\!\left(\frac{L \, \dot\theta_t}{v_t}\right)
$$

```python
from faran.numpy import model

estimator = model.bicycle.estimator.finite_difference(
    time_step_size=0.1, wheelbase=2.5,
)
```

**Pros:** No tuning needed, no noise assumptions. **Cons:** Very sensitive to observation noise; does not produce uncertainty estimates (no covariance).

### Kalman Filter (KF)

For **linear** models (integrator). Maintains a Gaussian belief $(\mu, \Sigma)$ and updates it optimally using the Kalman gain:

**Prediction:**
$$
\hat\mu_t = A \, \mu_{t-1}, \quad \hat\Sigma_t = A \, \Sigma_{t-1} \, A^\top + R
$$

**Update:**
$$
K_t = \hat\Sigma_t \, H^\top (H \, \hat\Sigma_t \, H^\top + Q)^{-1}
$$
$$
\mu_t = \hat\mu_t + K_t (z_t - H \, \hat\mu_t), \quad \Sigma_t = (I - K_t H) \, \hat\Sigma_t
$$

```python
estimator = model.integrator.estimator.kf(
    time_step_size=0.1,
    process_noise_covariance=1e-3,
    observation_noise_covariance=1e-2,
)
```

**When to use:** Integrator (constant-velocity) obstacle models. Optimal for linear-Gaussian systems.

### Extended Kalman Filter (EKF)

For **nonlinear** models (bicycle, unicycle). Linearizes the dynamics around the current estimate using the Jacobian $A_t = \frac{\partial f}{\partial x}\big|_{x=\mu_{t-1}}$, then applies the standard Kalman update.

```python
estimator = model.bicycle.estimator.ekf(
    time_step_size=0.1,
    wheelbase=2.5,
    process_noise_covariance=1e-3,
    observation_noise_covariance=1e-2,
)
```

**When to use:** Bicycle or unicycle obstacle models. Good default for nonlinear systems, but can diverge when the dynamics are highly nonlinear.

### Unscented Kalman Filter (UKF)

Higher-accuracy nonlinear estimation. Instead of linearizing with Jacobians, the UKF propagates a set of carefully chosen **sigma points** through the nonlinear dynamics and reconstructs the output distribution from those points.

The UKF captures mean and covariance to second order (vs. first order for EKF), making it more accurate for strongly nonlinear dynamics.

```python
estimator = model.bicycle.estimator.ukf(
    time_step_size=0.1,
    wheelbase=2.5,
    process_noise_covariance=1e-3,
    observation_noise_covariance=1e-2,
)
```

**When to use:** Same scenarios as EKF, but when you need better accuracy or when the EKF diverges. Slightly more expensive than EKF.

## Noise Covariance

All Kalman-family estimators require two noise covariance parameters:

| Parameter                      | Symbol | Meaning                                           |
|--------------------------------|--------|---------------------------------------------------|
| `process_noise_covariance`     | $R$    | How much the true dynamics deviate from the model |
| `observation_noise_covariance` | $Q$    | How noisy the sensor observations are             |

Both can be specified as:

- A **scalar** — expanded to a diagonal matrix: `1e-3` → $10^{-3} I$
- A **1D array** — diagonal entries: `[1e-3, 1e-2, 1e-3]`
- A **2D array** — full covariance matrix

Larger process noise makes the estimator trust observations more; larger observation noise makes it trust the model more.

## Adaptive Noise

When the true noise covariances are unknown or change over time, **Innovation-based Adaptive Estimation (IAE)** adapts them online using the observation innovation sequence.

The adaptive noise model monitors the innovation (difference between predicted and actual observations) over a sliding window and adjusts both process and observation noise covariances to match the observed statistics.

```python
from faran.numpy import noise

adaptive = noise.adaptive(window_size=10)

estimator = model.bicycle.estimator.ekf(
    time_step_size=0.1,
    wheelbase=2.5,
    process_noise_covariance=1e-3,
    observation_noise_covariance=1e-2,
    noise_model=adaptive,
)
```

### Clamped Noise

The adaptive model may produce very small noise values, causing the filter to become overconfident. The **clamped** decorator wraps any noise model and enforces a minimum floor on the diagonal entries:

```python
clamped = noise.clamped(
    noise.adaptive(window_size=10),
    floor=noise.covariances(
        process=1e-5,
        observation=1e-5,
        process_dimension=6,
        observation_dimension=3,
    ),
)
```

This is composable — you can clamp any noise model, not just the adaptive one.

## Choosing an Estimator

| Estimator         | Model Type          | Covariance | Tuning                      | Cost     |
|-------------------|---------------------|------------|-----------------------------|----------|
| Finite Difference | Any                 | No         | None                        | Cheapest |
| KF                | Linear (integrator) | Yes        | $R$, $Q$                    | Low      |
| EKF               | Nonlinear           | Yes        | $R$, $Q$                    | Medium   |
| UKF               | Nonlinear           | Yes        | $R$, $Q$, $\alpha$, $\beta$ | Highest  |

If you just need obstacle positions without uncertainty, use Finite Difference. If you need covariance for [risk-aware planning](obstacles.md#risk-aware-collision-cost), use EKF or UKF with appropriate noise settings.

## API Reference

See the [model API reference](../api/model.md) for estimator factory signatures and the [predictor API reference](../api/predictor.md) for how estimators connect to motion prediction.
