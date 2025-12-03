try:
    from beartype import beartype  # pyright: ignore[reportMissingImports]

    typechecker = beartype
except ImportError:
    typechecker = None


from jaxtyping import jaxtyped as jaxtyping_jaxtyped

jaxtyped = jaxtyping_jaxtyped(typechecker=typechecker)
