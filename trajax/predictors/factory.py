from typing import Final

from trajax.predictors.common import CurvilinearPredictor


class predictor:
    curvilinear: Final = CurvilinearPredictor.create
