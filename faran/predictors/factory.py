from typing import Final

from faran.predictors.common import StaticPredictor, CurvilinearPredictor


class predictor:
    """Factory namespace for creating obstacle motion predictors."""

    class numpy:
        curvilinear: Final = CurvilinearPredictor.create
        static: Final = StaticPredictor.create

    class jax:
        curvilinear: Final = CurvilinearPredictor.create
        static: Final = StaticPredictor.create
