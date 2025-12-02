from typing import Annotated


clear_type = Annotated[..., "Forget about this variable's previous types."]
"""A helper variable that tells the type checker to forget the type of a variable.

This is useful in tests where we are reusing the same variables in different parameterized
test cases, and we want to avoid type checker errors about incompatible types.

Example:
    ```python
    from tests.dsl.type import clear_type

    ...

    T = 10 
    reveal_type(T)  # T is Literal[10]

    T = 2 # T is Literal[10] or Literal[2]

    T = clear_type
    T = 2 # T is now Literal[2]
    
    ...
    ```
"""
