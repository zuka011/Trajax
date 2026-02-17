from typing import cast

from trajax.types import HasShape
from trajax.filters import JaxNoiseCovarianceDescription


def observation_dimension_from[D_o: int, D_x: int](
    *,
    process_noise_covariance: JaxNoiseCovarianceDescription[D_x],
    observation_noise_covariance: JaxNoiseCovarianceDescription[D_o],
    initial_state_covariance: HasShape | None,
    observation_dimension: D_o | None,
) -> D_o:
    if observation_dimension is not None:
        return observation_dimension

    if isinstance(observation_noise_covariance, HasShape):
        return cast(D_o, observation_noise_covariance.shape[0])

    if isinstance(process_noise_covariance, HasShape):
        return cast(D_o, observation_dimension_for(process_noise_covariance.shape[0]))

    if initial_state_covariance is not None:
        return cast(D_o, observation_dimension_for(initial_state_covariance.shape[0]))

    assert False, (
        "Observation dimension must be specified if noise covariances are not provided as arrays."
    )


def kf_state_dimension_for[D_o: int = int, D_x: int = int](
    observation_dimension: D_o,
) -> D_x:
    # NOTE: For the integrator model, both the states and state velocities are combined
    # into the estimated state vector.
    return cast(D_x, 2 * observation_dimension)


def observation_dimension_for[D_o: int = int, D_x: int = int](
    kf_state_dimension: D_x,
) -> D_o:
    assert kf_state_dimension % 2 == 0, (
        "KF state dimension must be even for the integrator model, "
        "e.g. if you observe x and y positions, the KF state is [x, y, x_dot, y_dot] with dimension 4."
    )

    # NOTE: For the integrator model, both the states and state velocities are combined
    # into the estimated state vector.
    return cast(D_o, kf_state_dimension // 2)
