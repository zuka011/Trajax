from typing import Final
from dataclasses import dataclass

from trajax.types import (
    NumPyControlInputBatchCreator,
    NumPyControlInputSequence,
    NumPyControlInputBatch,
    NumPySampler,
)

from numtypes import Array, Dims

import numpy as np


@dataclass(frozen=True)
class NumPyGaussianSampler[
    BatchT: NumPyControlInputBatch,
    D_u: int = int,
    M: int = int,
](NumPySampler[NumPyControlInputSequence, BatchT]):
    standard_deviation: Final[Array[Dims[D_u]]]
    to_batch: Final[NumPyControlInputBatchCreator[BatchT, D_u, M]]
    rng: np.random.Generator

    _rollout_count: Final[M]

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
        samples = around.array[..., None] + self.rng.normal(
            loc=0.0,
            scale=self.standard_deviation[None, :, None],
            size=(around.horizon, around.dimension, self.rollout_count),
        )

        return self.to_batch(array=samples)

    @property
    def rollout_count(self) -> M:
        return self._rollout_count
