from typing import Final, TypeAlias

from trajax.namespace import types as _types, classes as _classes
from trajax.factory import (
    mppi as _mppi,
    update as _update,
    padding as _padding,
    filters as _filters,
)
from trajax.costs import (
    costs as _costs,
    distance as _distance,
    boundary as _boundary,
    risk as _risk,
)
from trajax.models import model as _model
from trajax.obstacles import obstacles as _obstacles
from trajax.predictors import predictor as _predictor, propagator as _propagator
from trajax.samplers import sampler as _sampler
from trajax.trajectories import trajectory as _trajectory
from trajax.states import extract as _extract

types: TypeAlias = _types.numpy
classes: Final = _classes.numpy
mppi: Final = _mppi.numpy
update: Final = _update.numpy
padding: Final = _padding.numpy
filters: Final = _filters.numpy
costs: Final = _costs.numpy
distance: Final = _distance.numpy
boundary: Final = _boundary.numpy
risk: Final = _risk.numpy
model: Final = _model.numpy
obstacles: Final = _obstacles.numpy
predictor: Final = _predictor.numpy
propagator: Final = _propagator.numpy
sampler: Final = _sampler.numpy
trajectory: Final = _trajectory.numpy
extract: Final = _extract
