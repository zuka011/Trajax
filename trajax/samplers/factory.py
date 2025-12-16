from typing import Final

from trajax.samplers.gaussian import NumPyGaussianSampler, JaxGaussianSampler


class sampler:
    numpy: Final = NumPyGaussianSampler.create
    jax: Final = JaxGaussianSampler.create
