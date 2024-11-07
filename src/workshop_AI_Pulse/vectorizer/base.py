from enum import Enum

import torch


class BaseVectorizer:
    """
    Base class to template common methods.
    """

    def __init__(self):
        # Initialize vectorizer configuration and resources
        pass

    def encode(self, input: list[int | float | str]) -> torch.Tensor:
        """
        Encodes input into embedding space.
        :param input: List of integers, floats, or strings.
        :return: A PyTorch tensor representing the input in embedding space.
        """
        pass


class VectorizerClasses(Enum):
    """
    Accesible embedding types.
    """

    TEXT = "text"
    NUMBER = "number"


class NumberVectorizerClasses(Enum):
    """
    Accesible number embedding types.
    """

    LOGARTIHMIC = "logarithmic"
    SIGMOID = "sigmoid"
    SINUSOIDAL = "sinusoidal"
