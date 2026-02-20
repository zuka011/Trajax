from typing import Final

from faran.samplers.gaussian import NumPyGaussianSampler, JaxGaussianSampler
from faran.samplers.halton import NumPyHaltonSplineSampler, JaxHaltonSplineSampler


class sampler:
    class numpy:
        gaussian: Final = NumPyGaussianSampler.create
        halton: Final = NumPyHaltonSplineSampler.create

    class jax:
        gaussian: Final = JaxGaussianSampler.create
        halton: Final = JaxHaltonSplineSampler.create
