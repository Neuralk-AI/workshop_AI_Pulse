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
    Create embedding model from accessible list of classes.

    Parameters
    ----------
    embedding_type : str
        Modality to encode.
        Has to one of 'text', 'number'.
    method : Optional[str]
        Type of encoding scheme.
        Has to one of 'language_model', 'logarithmic', 'sigmoid', 'sinusoidal'.

    Returns
    -------
    BaseVectorizer
        Instance of embedding model

    Raises
    ------
    RuntimeError
        If embedding type is not supported it will raise a runtime error.
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
