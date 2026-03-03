from faran.types import HasShape, JaxNoiseCovarianceDescription


def observation_dimension_from(
    *,
    process_noise_covariance: JaxNoiseCovarianceDescription,
    observation_noise_covariance: JaxNoiseCovarianceDescription,
    initial_state_covariance: HasShape | None,
    observation_dimension: int | None,
) -> int:
    if observation_dimension is not None:
        return observation_dimension

    if isinstance(observation_noise_covariance, HasShape):
        return observation_noise_covariance.shape[0]

    if isinstance(process_noise_covariance, HasShape):
        return observation_dimension_for(process_noise_covariance.shape[0])

    if initial_state_covariance is not None:
        return observation_dimension_for(initial_state_covariance.shape[0])

    assert False, (
        "Observation dimension must be specified if noise covariances are not provided as arrays."
    )


def kf_state_dimension_for(observation_dimension: int) -> int:
    # NOTE: For the integrator model, both the states and state velocities are combined
    # into the estimated state vector.
    return 2 * observation_dimension


def observation_dimension_for(kf_state_dimension: int) -> int:
    assert kf_state_dimension % 2 == 0, (
        "KF state dimension must be even for the integrator model, "
        "e.g. if you observe x and y positions, the KF state is [x, y, x_dot, y_dot] with dimension 4."
    )

    # NOTE: For the integrator model, both the states and state velocities are combined
    # into the estimated state vector.
    return kf_state_dimension // 2
