from typing import NamedTuple
from dataclasses import dataclass

from trajax.types import jaxtyped

from numtypes import Array, Dims
from jaxtyping import Array as JaxArray, Float, Scalar

import jax
import jax.numpy as jnp


type JaxNoiseCovarianceArrayDescription[D_c: int] = (
    Array[Dims[D_c, D_c]]
    | Array[Dims[D_c]]
    | Float[JaxArray, "D_c D_c"]
    | Float[JaxArray, "D_c"]
)

type JaxNoiseCovarianceDescription[D_c: int] = (
    JaxNoiseCovarianceArrayDescription[D_c] | Scalar | float
)


class JaxGaussianBelief(NamedTuple):
    mean: Float[JaxArray, "D_x K"]
    covariance: Float[JaxArray, "D_x D_x K"]


class ObstaclePartitioning(NamedTuple):
    should_update: Float[JaxArray, "K"]
    should_initialize: Float[JaxArray, "K"]


@dataclass(kw_only=True)
class JaxKalmanFilter:
    """Standard Kalman Filter for linear systems."""

    @staticmethod
    def create() -> "JaxKalmanFilter":
        return JaxKalmanFilter()

    @staticmethod
    @jax.jit
    @jaxtyped
    def filter(
        observations: Float[JaxArray, "T D_z K"],
        *,
        initial_state_covariance: Float[JaxArray, "D_x D_x"],
        state_transition_matrix: Float[JaxArray, "D_x D_x"],
        process_noise_covariance: Float[JaxArray, "D_x D_x"],
        observation_noise_covariance: Float[JaxArray, "D_z D_z"],
        observation_matrix: Float[JaxArray, "D_z D_x"],
    ) -> JaxGaussianBelief:
        """Run the Kalman filter over a sequence of observations.

        Args:
            observations: The observed state history up to the current time step.
            initial_state_covariance: Sigma_0 matrix representing initial state uncertainty.
            state_transition_matrix: A matrix for state prediction.
            process_noise_covariance: R matrix representing the covariance of process noise.
            observation_noise_covariance: Q matrix representing the covariance of observation noise.
            observation_matrix: H matrix mapping state to observation space.
        """
        belief = JaxKalmanFilter.initial_belief_from(
            observations,
            initial_state_covariance=initial_state_covariance,
        )

        for observation in observations:
            belief = JaxKalmanFilter.predict(
                belief=belief,
                state_transition_matrix=state_transition_matrix,
                process_noise_covariance=process_noise_covariance,
            )
            belief = JaxKalmanFilter.update(
                observation=observation,
                prediction=belief,
                observation_matrix=observation_matrix,
                observation_noise_covariance=observation_noise_covariance,
                initial_state_covariance=initial_state_covariance,
            )

        return belief

    @staticmethod
    @jax.jit
    @jaxtyped
    def predict(
        *,
        belief: JaxGaussianBelief,
        state_transition_matrix: Float[JaxArray, "D_x D_x"],
        process_noise_covariance: Float[JaxArray, "D_x D_x"],
    ) -> JaxGaussianBelief:
        """Performs the prediction step of the Kalman filter from the existing belief.

        Args:
            belief: The current belief about the state.
            state_transition_matrix: A matrix for state prediction.
            process_noise_covariance: R matrix representing the covariance of process noise.
        """
        return jax_kalman_filter.predict(
            belief=belief,
            state_transition_matrix=state_transition_matrix,
            process_noise_covariance=process_noise_covariance,
        )

    @staticmethod
    @jax.jit
    @jaxtyped
    def update(
        observation: Float[JaxArray, "D_z K"],
        *,
        prediction: JaxGaussianBelief,
        observation_matrix: Float[JaxArray, "D_z D_x"],
        observation_noise_covariance: Float[JaxArray, "D_z D_z"],
        initial_state_covariance: Float[JaxArray, "D_x D_x"],
    ) -> JaxGaussianBelief:
        """Performs the update step of the Kalman filter using a new observation.

        Args:
            observation: The newly observed state.
            prediction: The predicted belief from the prediction step.
            observation_matrix: H matrix mapping state to observation space.
            observation_noise_covariance: Q matrix representing the covariance of observation noise.
            initial_state_covariance: Sigma_0 matrix representing initial state uncertainty.
        """
        return jax_kalman_filter.update(
            observation,
            prediction=prediction,
            observation_matrix=observation_matrix,
            observation_noise_covariance=observation_noise_covariance,
            initial_state_covariance=initial_state_covariance,
        )

    @staticmethod
    @jax.jit
    @jaxtyped
    def initial_belief_from(
        observations: Float[JaxArray, "T D_z K"],
        *,
        initial_state_covariance: Float[JaxArray, "D_x D_x"],
    ) -> JaxGaussianBelief:
        """Initializes the belief state from the first observation using a pseudo-inverse.

        This method provides a simple way to initialize the state mean based on the first
        observation, while using the provided initial state covariance for uncertainty.

        Args:
            observations: The observed state history up to the current time step.
            initial_state_covariance: Sigma_0 matrix representing initial state uncertainty.
        """
        return jax_kalman_filter.initial_belief_from(
            observations,
            initial_state_covariance=initial_state_covariance,
        )


class jax_kalman_filter:
    @staticmethod
    @jax.jit
    @jaxtyped
    def predict(
        *,
        belief: JaxGaussianBelief,
        state_transition_matrix: Float[JaxArray, "D_x D_x"],
        process_noise_covariance: Float[JaxArray, "D_x D_x"],
    ) -> JaxGaussianBelief:
        A = state_transition_matrix
        R = process_noise_covariance
        mu, sigma = belief

        return JaxGaussianBelief(
            mean=A @ mu,
            covariance=jnp.einsum("ij,jlk,ml->imk", A, sigma, A) + R[..., jnp.newaxis],
        )

    @staticmethod
    @jax.jit
    @jaxtyped
    def update(
        observation: Float[JaxArray, "D_z K"],
        *,
        prediction: JaxGaussianBelief,
        observation_matrix: Float[JaxArray, "D_z D_x"],
        observation_noise_covariance: Float[JaxArray, "D_z D_z"],
        initial_state_covariance: Float[JaxArray, "D_x D_x"],
    ) -> JaxGaussianBelief:
        H = observation_matrix
        Q = observation_noise_covariance

        def partition(
            *, observation: Float[JaxArray, "D_z K"], mean: Float[JaxArray, "D_x K"]
        ) -> ObstaclePartitioning:
            observation_valid = ~jnp.any(jnp.isnan(observation), axis=0)
            prediction_valid = ~jnp.any(jnp.isnan(mean), axis=0)

            return ObstaclePartitioning(
                should_update=observation_valid & prediction_valid,
                should_initialize=observation_valid & ~prediction_valid,
            )

        def substitute_missing_values(
            *,
            observation: Float[JaxArray, "D_z K"],
            mean: Float[JaxArray, "D_x K"],
            covariance: Float[JaxArray, "D_x D_x K"],
        ) -> tuple[
            Float[JaxArray, "D_z K"],
            Float[JaxArray, "D_x K"],
            Float[JaxArray, "D_x D_x K"],
        ]:
            covariance_is_nan = jnp.any(jnp.isnan(covariance), axis=(0, 1))

            return (
                jnp.where(jnp.isnan(observation), 0.0, observation),
                jnp.where(jnp.isnan(mean), 0.0, mean),
                jnp.where(
                    covariance_is_nan[jnp.newaxis, jnp.newaxis, :],
                    initial_state_covariance[..., jnp.newaxis],
                    covariance,
                ),
            )

        def update(
            *,
            observation: Float[JaxArray, "D_z K"],
            mean: Float[JaxArray, "D_x K"],
            covariance: Float[JaxArray, "D_x D_x K"],
        ) -> JaxGaussianBelief:
            D_x = prediction.mean.shape[0]

            S = jnp.einsum("ij,jlk,ml->imk", H, covariance, H) + Q[..., jnp.newaxis]
            rhs = jnp.einsum("ijk,lj->ilk", covariance, H)
            K = jnp.linalg.solve(
                S.transpose(2, 0, 1), rhs.transpose(2, 1, 0)
            ).transpose(2, 1, 0)

            innovation = observation - H @ mean
            KH = jnp.einsum("ijk,jl->ilk", K, H)

            return JaxGaussianBelief(
                mean=mean + jnp.einsum("ijk,jk->ik", K, innovation),
                covariance=jnp.einsum(
                    "ijk,jlk->ilk", jnp.eye(D_x)[..., jnp.newaxis] - KH, covariance
                ),
            )

        def initialize_from(
            observation: Float[JaxArray, "D_z K"],
        ) -> JaxGaussianBelief:
            K = observation.shape[1]

            # NOTE: The mean state is the observation when available, 0 otherwise.
            return JaxGaussianBelief(
                mean=jnp.linalg.pinv(H) @ observation,
                covariance=jnp.repeat(
                    initial_state_covariance[..., jnp.newaxis], K, axis=2
                ),
            )

        def blend(
            partitioning: ObstaclePartitioning,
            *,
            initial: JaxGaussianBelief,
            prediction: JaxGaussianBelief,
            update: JaxGaussianBelief,
        ) -> JaxGaussianBelief:
            return JaxGaussianBelief(
                mean=jnp.select(
                    [partitioning.should_update, partitioning.should_initialize],
                    [update.mean, initial.mean],
                    default=prediction.mean,
                ),
                covariance=jnp.select(
                    [partitioning.should_update, partitioning.should_initialize],
                    [update.covariance, initial.covariance],
                    default=prediction.covariance,
                ),
            )

        partitioning = partition(observation=observation, mean=prediction.mean)
        observation, mean, covariance = substitute_missing_values(
            observation=observation,
            mean=prediction.mean,
            covariance=prediction.covariance,
        )

        return blend(
            partitioning,
            initial=initialize_from(observation),
            prediction=prediction,
            update=update(observation=observation, mean=mean, covariance=covariance),
        )

    @staticmethod
    @jax.jit
    @jaxtyped
    def initial_belief_from(
        observations: Float[JaxArray, "T D_z K"],
        *,
        initial_state_covariance: Float[JaxArray, "D_x D_x"],
    ) -> JaxGaussianBelief:
        D_x = initial_state_covariance.shape[0]
        K = observations.shape[2]

        return JaxGaussianBelief(
            mean=jnp.full((D_x, K), jnp.nan),
            covariance=jnp.full((D_x, D_x, K), jnp.nan),
        )

    @staticmethod
    def standardize_noise_covariance[D_c: int](
        covariance: JaxNoiseCovarianceDescription[D_c],
        *,
        dimension: D_c,
    ) -> Float[JaxArray, "D_c D_c"]:
        if isinstance(covariance, (int, float)):
            return covariance * jnp.eye(dimension)

        covariance = jnp.asarray(covariance)

        match covariance.ndim:
            case 2:
                assert covariance.shape == (dimension, dimension), (
                    f"Expected covariance shape ({dimension}, {dimension}), got {covariance.shape}."
                )
                return covariance
            case 1:
                assert covariance.shape == (dimension,), (
                    f"Expected covariance shape ({dimension},), got {covariance.shape}."
                )
                return jnp.diag(covariance)
            case _:
                assert False, f"Invalid covariance shape: {covariance.shape}"
