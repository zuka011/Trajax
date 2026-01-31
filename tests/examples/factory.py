from typing import Final

from tests.examples.mppc import basic, accelerated


class reference:
    numpy: Final = basic.reference
    jax: Final = accelerated.reference


class obstacles:
    numpy: Final = basic.obstacles
    jax: Final = accelerated.obstacles


class sampling:
    numpy: Final = basic.NumPySamplingOptions
    jax: Final = accelerated.JaxSamplingOptions


class mpcc:
    numpy: Final = basic.configure
    jax: Final = accelerated.configure
