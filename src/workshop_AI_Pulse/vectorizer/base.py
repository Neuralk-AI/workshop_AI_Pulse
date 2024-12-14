from enum import Enum

import torch


class BaseVectorizer:
    """
    Base class  for all vectorizers.
    """

    def __init__(self):
        pass

    def encode(self, input: list[int | float | str]) -> torch.Tensor:
        """
        Use an embedding model to encode an input into an embedding space.

        Args:
            - input : list[int  |  float  |  str]
        Returns:
            - torch.Tensor
        """
        pass


class VectorizerTypes(Enum):
    """
    All possible modalities that can be encoded.
    """

    TEXT = "text"
    NUMBER = "number"
