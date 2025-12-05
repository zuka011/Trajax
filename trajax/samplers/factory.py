from typing import Final

from trajax.gaussian import NumPyGaussianSampler, JaxGaussianSampler


class sampler:
    numpy: Final = NumPyGaussianSampler.create
    jax: Final = JaxGaussianSampler.create
