from typing import Final

from trajax.predictors.common import ConstantVelocityPredictor


class predictor:
    constant_velocity: Final = ConstantVelocityPredictor.create
