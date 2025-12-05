from typing import Protocol, Final
from dataclasses import dataclass

from trajax.type import jaxtyped
from trajax.model import (
    ControlInputSequence as AnyControlInputSequence,
    ControlInputBatch as AnyControlInputBatch,
)

import jax
import jax.random as jrandom

from jaxtyping import Array, Float, PRNGKeyArray, Scalar


class ControlInputSequence[T: int, D_u: int](AnyControlInputSequence[T, D_u], Protocol):
    @property
    def array(self) -> Float[Array, "T D_u"]:
        """Returns the underlying JAX array representing the control input sequence."""
        ...


class ControlInputBatch[T: int, D_u: int, M: int](
    AnyControlInputBatch[T, D_u, M], Protocol
):
    @property
    def array(self) -> Float[Array, "T D_u M"]:
        """Returns the underlying JAX array representing the control input batch."""
        ...


class ControlInputBatchCreator[ControlInputBatchT: ControlInputBatch](Protocol):
    def __call__(self, *, array: Float[Array, "T D_u M"]) -> ControlInputBatchT:
        """Creates a ControlInputBatch from the given array."""
        ...


@dataclass
class JaxGaussianSampler[SequenceT: ControlInputSequence, BatchT: ControlInputBatch]:
    standard_deviation: Final[float]
    rollout_count: Final[int]
    to_batch: Final[ControlInputBatchCreator[BatchT]]

    key: PRNGKeyArray

    @staticmethod
    def create(
        *,
        standard_deviation: float,
        rollout_count: int,
        to_batch: ControlInputBatchCreator[BatchT],
        key: PRNGKeyArray,
    ) -> "JaxGaussianSampler[SequenceT, BatchT]":
        """Creates a sampler generating Gaussian noise around the specified control input
        sequence.
        """
        return JaxGaussianSampler(
            standard_deviation=standard_deviation,
            rollout_count=rollout_count,
            to_batch=to_batch,
            key=key,
        )

    def sample(self, *, around: SequenceT) -> BatchT:
        self.key, samples = sample(
            self.key,
            around=around.array,
            standard_deviation=self.standard_deviation,
            rollout_count=self.rollout_count,
        )

        return self.to_batch(array=samples)


@jax.jit(static_argnames=("rollout_count",))
@jaxtyped
def sample(
    key: PRNGKeyArray,
    *,
    around: Float[Array, "T D_u"],
    standard_deviation: Scalar,
    rollout_count: int,
) -> tuple[PRNGKeyArray, Float[Array, "T D_u M"]]:
    time_horizon, control_dimension = around.shape

    key, subkey = jrandom.split(key)
    samples = around[..., None] + standard_deviation * jrandom.normal(
        subkey, shape=(time_horizon, control_dimension, rollout_count)
    )

    return key, samples
