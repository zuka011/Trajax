from trajax import FilterFunction, ControlInputSequence, filters

from numtypes import array, Array

import numpy as np

from tests.dsl import mppi as data, clear_type
from pytest import mark


@mark.parametrize(
    ["savgol", "inputs", "margin", "expected"],
    [
        (
            savgol := filters.numpy.savgol(
                window_length=(window_length := 11), polynomial_order=3
            ),
            inputs := data.numpy.control_input_sequence(
                expected := array(
                    np.array(
                        [
                            t := np.linspace(0, 20, T := 30),
                            # Cubic polynomial: f(t) = 0.1t^3 - 0.5t^2 + 2t + 1
                            0.1 * t**3 - 0.5 * t**2 + 2 * t + 1,
                            # Quadratic polynomial: f(t) = -0.2t^2 + 3t + 0.5
                            -0.2 * t**2 + 3 * t + 0.5,
                            # Linear polynomial: f(t) = 1.5t + 2
                            1.5 * t + 2,
                            # Constant polynomial: f(t) = 4
                            4 * np.ones_like(t),
                        ][1:]
                    ).T.tolist(),
                    shape=(T, 4),
                )
            ),
            margin := window_length // 2,  # Inputs at edges may not be preserved.
            expected,
        ),
        (  # Analogous test for JAX
            savgol := filters.jax.savgol(
                window_length=(window_length := 11), polynomial_order=3
            ),
            inputs := data.jax.control_input_sequence(
                expected := array(
                    np.array(
                        [
                            t := np.linspace(0, 20, T := 30),
                            0.1 * t**3 - 0.5 * t**2 + 2 * t + 1,
                            -0.2 * t**2 + 3 * t + 0.5,
                            1.5 * t + 2,
                            4 * np.ones_like(t),
                        ][1:]
                    ).T.tolist(),
                    shape=(T, 4),
                )
            ),
            margin := window_length // 2,
            expected,
        ),
    ],
)
def test_that_polynomials_are_preserved_by_savgol_filter_when_order_is_less_than_or_equal_to_filter_order[
    InputSequenceT: ControlInputSequence
](
    savgol: FilterFunction[InputSequenceT],
    inputs: InputSequenceT,
    margin: int,
    expected: Array,
) -> None:
    assert np.allclose(
        np.asarray(savgol(optimal_input=inputs))[margin:-margin],
        expected[margin:-margin],
    )


T = clear_type


@mark.parametrize(
    ["savgol", "clean", "noisy"],
    [
        (
            savgol := filters.numpy.savgol(window_length=15, polynomial_order=2),
            clean := data.numpy.control_input_sequence(
                signal := array(
                    np.column_stack(
                        [
                            np.sin(t := np.linspace(0, 4 * np.pi, T := 100)),
                            np.cos(t),
                            np.sin(2 * t),
                        ]
                    ).tolist(),
                    shape=(T, D_u := 3),
                )
            ),
            noisy := data.numpy.control_input_sequence(
                array(
                    (
                        signal + np.random.default_rng(0).normal(0, 0.1, (T, D_u))  # type: ignore
                    ).tolist(),
                    shape=(T, D_u),
                )
            ),
        ),
        (  # Analogous test for JAX
            savgol := filters.jax.savgol(window_length=15, polynomial_order=2),
            clean := data.jax.control_input_sequence(
                signal := array(
                    np.column_stack(
                        [
                            np.sin(t := np.linspace(0, 4 * np.pi, T := 100)),
                            np.cos(t),
                            np.sin(2 * t),
                        ]
                    ).tolist(),
                    shape=(T, D_u := 3),
                )
            ),
            noisy := data.jax.control_input_sequence(
                array(
                    (
                        signal + np.random.default_rng(0).normal(0, 0.1, (T, D_u))  # type: ignore
                    ).tolist(),
                    shape=(T, D_u),
                )
            ),
        ),
    ],
)
def test_that_savgol_filter_reduces_noise_while_preserving_trend[
    InputSequenceT: ControlInputSequence
](
    savgol: FilterFunction[InputSequenceT],
    clean: InputSequenceT,
    noisy: InputSequenceT,
) -> None:
    filtered = savgol(optimal_input=noisy)

    clean_array = np.asarray(clean)
    noisy_array = np.asarray(noisy)
    filtered_array = np.asarray(filtered)

    assert np.var(filtered_array - clean_array) < np.var(noisy_array - clean_array)


@mark.parametrize(
    ["savgol", "inputs"],
    [
        (
            savgol := filters.numpy.savgol(window_length=5, polynomial_order=3),
            inputs := data.numpy.control_input_sequence(np.ones((T := 50, D := 4))),
        ),
        (
            savgol := filters.jax.savgol(window_length=5, polynomial_order=3),
            inputs := data.jax.control_input_sequence(np.ones((T := 50, D := 4))),
        ),
    ],
)
def test_that_savgol_filter_preserves_input_shape[InputSequenceT: ControlInputSequence](
    savgol: FilterFunction[InputSequenceT],
    inputs: InputSequenceT,
) -> None:
    assert np.asarray(savgol(optimal_input=inputs)).shape == np.asarray(inputs).shape
