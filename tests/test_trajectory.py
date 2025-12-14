from trajax import trajectory, types, Trajectory, PathParameters, ReferencePoints

import numpy as np
import jax.numpy as jnp
from numtypes import array

from pytest import mark


@mark.parametrize(
    ["trajectory", "path_parameters", "expected"],
    [
        (
            trajectory.numpy.line(start=(0.0, 0.0), end=(10.0, 0.0), path_length=10),
            path_parameters := types.numpy.path_parameters(
                array([[0.0, 10.0], [5.0, 5.0], [10.0, 0.0]], shape=(T := 3, M := 2))
            ),
            expected := types.numpy.reference_points(
                x=array([[0.0, 10.0], [5.0, 5.0], [10.0, 0.0]], shape=(T, M)),
                y=array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], shape=(T, M)),
                heading=array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], shape=(T, M)),
            ),
        ),
        (
            trajectory.numpy.line(start=(0.0, 0.0), end=(3.0, 4.0), path_length=5),
            path_parameters := types.numpy.path_parameters(
                array([[0.0, 5.0], [2.5, 2.5], [5.0, 0.0]], shape=(T := 3, M := 2))
            ),
            expected := types.numpy.reference_points(
                x=array([[0.0, 3.0], [1.5, 1.5], [3.0, 0.0]], shape=(T, M)),
                y=array([[0.0, 4.0], [2.0, 2.0], [4.0, 0.0]], shape=(T, M)),
                heading=array(
                    np.full((T, M), np.arctan2(4.0, 3.0)).tolist(), shape=(T, M)
                ),
            ),
        ),
        (
            trajectory.jax.line(start=(0.0, 0.0), end=(10.0, 0.0), path_length=10),
            path_parameters := types.jax.path_parameters(
                array([[0.0, 10.0], [5.0, 5.0], [10.0, 0.0]], shape=(T := 3, M := 2))
            ),
            expected := types.jax.reference_points(
                x=array([[0.0, 10.0], [5.0, 5.0], [10.0, 0.0]], shape=(T, M)),
                y=array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], shape=(T, M)),
                heading=array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], shape=(T, M)),
            ),
        ),
        (
            trajectory.jax.line(start=(0.0, 0.0), end=(3.0, 4.0), path_length=5),
            path_parameters := types.jax.path_parameters(
                array([[0.0, 5.0], [2.5, 2.5], [5.0, 0.0]], shape=(T := 3, M := 2))
            ),
            expected := types.jax.reference_points(
                x=jnp.array([[0.0, 3.0], [1.5, 1.5], [3.0, 0.0]]),
                y=jnp.array([[0.0, 4.0], [2.0, 2.0], [4.0, 0.0]]),
                heading=jnp.array(jnp.full((T, M), jnp.arctan2(4.0, 3.0)).tolist()),
                horizon=T,
                rollout_count=M,
            ),
        ),
    ],
)
def test_that_line_trajectory_batch_query_returns_correct_positions_and_headings[
    PathParametersT: PathParameters,
    ReferencePointsT: ReferencePoints,
](
    trajectory: Trajectory[PathParametersT, ReferencePointsT],
    path_parameters: PathParametersT,
    expected: ReferencePointsT,
) -> None:
    actual = trajectory.query(path_parameters)

    assert np.allclose(actual, expected)
