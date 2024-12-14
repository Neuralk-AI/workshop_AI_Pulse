from functools import lru_cache
from typing import Optional

from workshop_AI_Pulse.vectorizer import (
    BaseVectorizer,
    VectorizerClasses,
    TextVectorizer,
    NumberVectorizer,
)


@lru_cache
def load_model(
    embedding_type: str, method: Optional[str] = None, merge: str = "*"
) -> BaseVectorizer:
    """
    Load an embedding model from its name.

    Args:
        - embedding_type : str ('text', 'number')
        - method : Optional[str] ('sinusoid', 'sigmoid', 'logarithmic')

    Returns:
        - BaseVectorizer
    """
    if embedding_type == "number":
        return NumberVectorizer(method=method, merge=merge)
    elif embedding_type == "text":
        return TextVectorizer()
    else:
        raise NotImplementedError(
            (
                f"{embedding_type}  is not a valid  embedding_type, "
                f"please choose one of {[element.value for element in VectorizerClasses]}."
            )
        )
