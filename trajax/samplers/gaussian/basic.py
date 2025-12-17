from typing import Protocol
from dataclasses import dataclass

from trajax.mppi import NumPyControlInputSequence, NumPyControlInputBatch, NumPySampler

from numtypes import Array, Dims

import numpy as np


class NumPyControlInputBatchCreator[
    ControlInputBatchT: NumPyControlInputBatch,
    D_u: int = int,
    M: int = int,
](Protocol):
    def __call__(self, *, array: Array[Dims[int, D_u, M]]) -> ControlInputBatchT:
        """Creates a ControlInputBatch from the given array."""
        ...


@dataclass(frozen=True)
class NumPyGaussianSampler[
    BatchT: NumPyControlInputBatch,
    D_u: int = int,
    M: int = int,
](NumPySampler[NumPyControlInputSequence, BatchT]):
    standard_deviation: Array[Dims[D_u]]
    to_batch: NumPyControlInputBatchCreator[BatchT, D_u, M]
    rng: np.random.Generator

    _rollout_count: M

    @staticmethod
    def create[B: NumPyControlInputBatch, D_u_: int, M_: int](
        *,
        standard_deviation: Array[Dims[D_u_]],
        rollout_count: M_,
        to_batch: NumPyControlInputBatchCreator[B, D_u_, M_],
        seed: int,
    ) -> "NumPyGaussianSampler[B, D_u_, M_]":
        """Creates a sampler generating Gaussian noise around the specified control input
        sequence.
        """
        return NumPyGaussianSampler(
            standard_deviation=standard_deviation,
            to_batch=to_batch,
            rng=np.random.default_rng(seed),
            _rollout_count=rollout_count,
        )

    def sample(self, *, around: NumPyControlInputSequence) -> BatchT:
        nominal = np.asarray(around)
        time_horizon, control_dimension = nominal.shape

        samples = nominal[..., None] + self.rng.normal(
            loc=0.0,
            scale=self.standard_deviation[None, :, None],
            size=(time_horizon, control_dimension, self.rollout_count),
        )

        return self.to_batch(array=samples)

    @property
    def rollout_count(self) -> M:
        return self._rollout_count
