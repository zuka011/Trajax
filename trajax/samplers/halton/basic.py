from typing import Final
from dataclasses import dataclass
from functools import lru_cache

from trajax.types import (
    NumPyControlInputBatchCreator,
    NumPyControlInputSequence,
    NumPyControlInputBatch,
    NumPySampler,
)

from numtypes import Array, Dims

from scipy.stats.qmc import Halton
from scipy.stats import norm
from scipy.interpolate import CubicSpline

import numpy as np


@dataclass(kw_only=True)
class NumPyHaltonSplineSampler[
    BatchT: NumPyControlInputBatch,
    D_u: int = int,
    M: int = int,
](NumPySampler[NumPyControlInputSequence, BatchT]):
    standard_deviation: Final[Array[Dims[D_u]]]
    to_batch: Final[NumPyControlInputBatchCreator[BatchT, D_u, M]]
    knot_count: Final[int]
    seed: Final[int]
    halton_start_index: int

    _rollout_count: Final[M]

    @staticmethod
    def create[B: NumPyControlInputBatch, D_u_: int, M_: int](
        *,
        standard_deviation: Array[Dims[D_u_]],
        rollout_count: M_,
        knot_count: int,
        to_batch: NumPyControlInputBatchCreator[B, D_u_, M_],
        seed: int,
    ) -> "NumPyHaltonSplineSampler[B, D_u_, M_]":
        """Creates a sampler generating temporally smooth perturbations using Halton
        sequences and cubic splines around the specified control input sequence.
        """
        return NumPyHaltonSplineSampler(
            standard_deviation=standard_deviation,
            to_batch=to_batch,
            knot_count=knot_count,
            seed=seed,
            halton_start_index=0,
            _rollout_count=rollout_count,
        )

    def __post_init__(self) -> None:
        assert self.knot_count >= 2, "Knot count must be at least 2."

    def sample(self, *, around: NumPyControlInputSequence) -> BatchT:
        assert self.knot_count <= around.horizon, (
            f"Knot count ({self.knot_count}) cannot exceed time horizon ({around.horizon})."
        )

        halton_samples = self._generate_halton_samples()
        gaussian_knots = self._transform_to_gaussian(halton_samples)
        perturbations = self._interpolate_knots(
            gaussian_knots=gaussian_knots, time_horizon=around.horizon
        )

        scaled_perturbations = perturbations * self.standard_deviation[None, :, None]
        samples = around.array[..., None] + scaled_perturbations

        return self.to_batch(array=samples)

    @property
    def dimension(self) -> D_u:
        return self.standard_deviation.shape[0]

    @property
    def rollout_count(self) -> M:
        return self._rollout_count

    def _generate_halton_samples[N_knot: int = int](self) -> Array[Dims[M, N_knot]]:
        halton_dimensions = self.knot_count * self.dimension
        halton_sequence = Halton(d=halton_dimensions, scramble=True, seed=self.seed)
        halton_sequence.fast_forward(self.halton_start_index)

        uniform_samples = halton_sequence.random(n=self.rollout_count)
        self.halton_start_index += self.rollout_count

        return uniform_samples

    def _transform_to_gaussian[N_knot: int](
        self, halton_samples: Array[Dims[M, N_knot]]
    ) -> Array[Dims[M, N_knot, D_u]]:
        rollout_count = halton_samples.shape[0]

        # NOTE: Clip to avoid infinities from ppf at 0 and 1.
        clipped = np.clip(halton_samples, 1e-10, 1 - 1e-10)
        gaussian_samples = norm.ppf(clipped)

        return gaussian_samples.reshape(rollout_count, self.knot_count, self.dimension)

    def _interpolate_knots[T: int, N_knot: int = int](
        self, *, gaussian_knots: Array[Dims[M, N_knot, D_u]], time_horizon: T
    ) -> Array[Dims[T, D_u, M]]:
        knot_times = knot_times_for(
            time_horizon=time_horizon, knot_count=self.knot_count
        )

        evaluation_times = np.arange(time_horizon)
        knots_batched = gaussian_knots.transpose(1, 0, 2).reshape(self.knot_count, -1)
        spline = CubicSpline(knot_times, knots_batched)

        perturbations_flat = spline(evaluation_times)
        perturbations = perturbations_flat.reshape(
            time_horizon, self.rollout_count, self.dimension
        )

        return perturbations.transpose(0, 2, 1)


@lru_cache
def knot_times_for[N_knot: int](
    *, time_horizon: int, knot_count: N_knot
) -> Array[Dims[N_knot]]:
    return np.linspace(0, time_horizon - 1, knot_count)
