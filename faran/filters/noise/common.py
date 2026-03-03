from typing import Any


class IdentityNoiseModel:
    @property
    def state(self) -> None:
        return None

    def __call__[NoiseT](
        self, *, noise: NoiseT, prediction: Any, observation: Any, state: None = None
    ) -> tuple[NoiseT, None]:
        return noise, state


class IdentityNoiseModelProvider:
    def __call__(self, *, observation_matrix: Any, noise: Any) -> IdentityNoiseModel:
        return IdentityNoiseModel()
