from typing import Protocol, Sequence, Mapping, Literal, Type, Any, overload
from dataclasses import dataclass

type NotRequired = Literal[False]
type Required = Literal[True]
type IsRequired = NotRequired | Required


@dataclass(frozen=True)
class SimulationDataAccessor[T, RequiredT: IsRequired]:
    key: str
    required: RequiredT
    _type: Type[T]

    @staticmethod
    def create[T_](
        default_type: Type[T_], *, key: str
    ) -> "SimulationDataAccessor[T_, NotRequired]":
        return SimulationDataAccessor(key=key, required=False, _type=default_type)

    def require(self) -> "SimulationDataAccessor[T, Required]":
        return SimulationDataAccessor(key=self.key, required=True, _type=self._type)

    def assume[T_](self, new_type: Type[T_]) -> "SimulationDataAccessor[T_, RequiredT]":
        return SimulationDataAccessor(
            key=self.key, required=self.required, _type=new_type
        )


@dataclass(kw_only=True, frozen=True)
class SimulationData:
    data: Mapping[str, Any]

    @staticmethod
    def create(**kwargs: Any) -> "SimulationData":
        return SimulationData(data=kwargs)

    @overload
    def __call__[T](self, accessor: SimulationDataAccessor[T, Required]) -> T:
        """Retrieves the data for the specified accessor, ensuring it is present."""
        ...

    @overload
    def __call__[T](self, accessor: SimulationDataAccessor[T, NotRequired]) -> T | None:
        """Retrieves the data for the specified accessor, returning None if not present."""
        ...

    def __call__(self, accessor: SimulationDataAccessor) -> Any | None:
        assert accessor.key in self.data or not accessor.required, (
            f"Required simulation data '{accessor.key}' is missing."
        )

        return self.data.get(accessor.key)


class StateSequenceCreator[StateT, StateSequenceT](Protocol):
    def __call__(self, states: Sequence[StateT]) -> StateSequenceT:
        """Creates a state sequence from the provided sequence of states."""
        ...


class ObstacleStateSequencesCreator[
    ObstacleStatesForTimeStepT,
    ObstacleStateSequencesT,
](Protocol):
    def __call__(
        self, obstacle_states: Sequence[ObstacleStatesForTimeStepT]
    ) -> ObstacleStateSequencesT:
        """Creates obstacle state sequences from the provided sequence of obstacle states."""
        ...


class ObstacleStateObserver[ObstacleStatesForTimeStepT](Protocol):
    def observe(self, states: ObstacleStatesForTimeStepT) -> None:
        """Observes the specified obstacle states for a given time step."""
        ...
