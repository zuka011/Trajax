from typing import Protocol
from dataclasses import dataclass

from trajax.model import ControlInputSequence, ControlInputBatch

import numpy as np
from numtypes import Array, Dim3


class ControlInputBatchCreator[ControlInputBatchT: ControlInputBatch](Protocol):
    def __call__(self, *, array: Array[Dim3]) -> ControlInputBatchT:
        """Creates a ControlInputBatch from the given array."""
        ...


@dataclass(frozen=True)
class NumPyGaussianSampler[SequenceT: ControlInputSequence, BatchT: ControlInputBatch]:
    standard_deviation: float
    rollout_count: int
    to_batch: ControlInputBatchCreator[BatchT]
    rng: np.random.Generator

    @staticmethod
    def create(
        *,
        standard_deviation: float,
        rollout_count: int,
        to_batch: ControlInputBatchCreator[BatchT],
        seed: int,
    ) -> "NumPyGaussianSampler[SequenceT, BatchT]":
        """Creates a sampler generating Gaussian noise around the specified control input
        sequence.
        """
        return NumPyGaussianSampler(
            standard_deviation=standard_deviation,
            rollout_count=rollout_count,
            to_batch=to_batch,
            rng=np.random.default_rng(seed),
        )

    def sample(self, *, around: SequenceT) -> BatchT:
        nominal = np.asarray(around)
        time_horizon, control_dimension = nominal.shape

        samples = nominal[..., None] + self.rng.normal(
            loc=0.0,
            scale=self.standard_deviation,
            size=(time_horizon, control_dimension, self.rollout_count),
        )

        return self.to_batch(array=samples)
