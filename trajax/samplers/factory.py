from typing import Final

from trajax.samplers.gaussian import NumPyGaussianSampler, JaxGaussianSampler
from trajax.samplers.halton import NumPyHaltonSplineSampler, JaxHaltonSplineSampler


class sampler:
    class numpy:
        gaussian: Final = NumPyGaussianSampler.create
        halton: Final = NumPyHaltonSplineSampler.create

    class jax:
        gaussian: Final = JaxGaussianSampler.create
        halton: Final = JaxHaltonSplineSampler.create
