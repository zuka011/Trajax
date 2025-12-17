from typing import Final

from trajax.samplers.gaussian import NumPyGaussianSampler, JaxGaussianSampler


class sampler:
    class numpy:
        gaussian: Final = NumPyGaussianSampler.create

    class jax:
        gaussian: Final = JaxGaussianSampler.create
