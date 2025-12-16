from typing import Protocol, Final, overload, cast
from dataclasses import dataclass

from trajax.type import jaxtyped
from trajax.model import (
    ControlInputSequence as AnyControlInputSequence,
    ControlInputBatch as AnyControlInputBatch,
)
from trajax.mppi import Sampler

from jaxtyping import Array as JaxArray, Float, PRNGKeyArray
from numtypes import Array, Dims

import jax
import jax.random as jrandom
import jax.numpy as jnp


class ControlInputSequence[T: int = int, D_u: int = int](
    AnyControlInputSequence[T, D_u], Protocol
):
    @property
    def array(self) -> Float[JaxArray, "T D_u"]:
        """Returns the underlying JAX array representing the control input sequence."""
        ...


class ControlInputBatch[T: int = int, D_u: int = int, M: int = int](
    AnyControlInputBatch[T, D_u, M], Protocol
):
    @property
    def array(self) -> Float[JaxArray, "T D_u M"]:
        """Returns the underlying JAX array representing the control input batch."""
        ...


class ControlInputBatchCreator[ControlInputBatchT: ControlInputBatch](Protocol):
    def __call__(self, *, array: Float[JaxArray, "T D_u M"]) -> ControlInputBatchT:
        """Creates a ControlInputBatch from the given array."""
        ...


@dataclass(kw_only=True)
class JaxGaussianSampler[BatchT: ControlInputBatch, D_u: int = int, M: int = int](
    Sampler[ControlInputSequence, BatchT, M]
):
    standard_deviation: Final[Float[JaxArray, "D_u"]]
    to_batch: Final[ControlInputBatchCreator[BatchT]]

    _control_dimension: Final[D_u]
    _rollout_count: Final[M]

    key: PRNGKeyArray

    @overload
    @staticmethod
    def create[B: ControlInputBatch, D_u_: int, M_: int](
        *,
        standard_deviation: Array[Dims[D_u_]],
        rollout_count: M_,
        to_batch: ControlInputBatchCreator[B],
        key: PRNGKeyArray,
    ) -> "JaxGaussianSampler[B, D_u_, M_]":
        """Creates a sampler generating Gaussian noise around the specified control input
        sequence.
        """
        ...

    @overload
    @staticmethod
    def create[B: ControlInputBatch, D_u_: int, M_: int](
        *,
        standard_deviation: Float[JaxArray, "D_u"],
        control_dimension: D_u_,
        rollout_count: M_,
        to_batch: ControlInputBatchCreator[B],
        key: PRNGKeyArray,
    ) -> "JaxGaussianSampler[B, D_u_, M_]":
        """Creates a sampler generating Gaussian noise around the specified control input
        sequence.
        """
        ...

    @staticmethod
    def create[B: ControlInputBatch, D_u_: int, M_: int](
        *,
        standard_deviation: Array[Dims[D_u_]] | Float[JaxArray, "D_u"],
        control_dimension: D_u_ | None = None,
        rollout_count: M_,
        to_batch: ControlInputBatchCreator[B],
        key: PRNGKeyArray,
    ) -> "JaxGaussianSampler[B, D_u_, M_]":
        """Creates a sampler generating Gaussian noise around the specified control input
        sequence.
        """
        return JaxGaussianSampler(
            standard_deviation=jnp.asarray(standard_deviation),
            to_batch=to_batch,
            _control_dimension=(
                control_dimension
                if control_dimension is not None
                else cast(D_u_, standard_deviation.shape[0])
            ),
            _rollout_count=rollout_count,
            key=key,
        )

    def __post_init__(self) -> None:
        assert self.standard_deviation.shape[0] == self.control_dimension, (
            f"Expected standard deviation with shape ({self.control_dimension},), "
            f"but got {self.standard_deviation.shape}"
        )

    def sample(self, *, around: ControlInputSequence) -> BatchT:
        self.key, samples = sample(
            self.key,
            around=around.array,
            standard_deviation=self.standard_deviation,
            rollout_count=self.rollout_count,
        )

        return self.to_batch(array=samples)

    @property
    def control_dimension(self) -> D_u:
        return self._control_dimension

    @property
    def rollout_count(self) -> M:
        return self._rollout_count


@jax.jit(static_argnames=("rollout_count",))
@jaxtyped
def sample(
    key: PRNGKeyArray,
    *,
    around: Float[JaxArray, "T D_u"],
    standard_deviation: Float[JaxArray, "D_u"],
    rollout_count: int,
) -> tuple[PRNGKeyArray, Float[JaxArray, "T D_u M"]]:
    time_horizon, control_dimension = around.shape

    key, subkey = jrandom.split(key)
    samples = around[..., None] + standard_deviation[None, :, None] * jrandom.normal(
        subkey, shape=(time_horizon, control_dimension, rollout_count)
    )

    return key, samples
