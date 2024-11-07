import torch
import numpy as np
from sentence_transformers import SentenceTransformer

from workshop_AI_Pulse.vectorizer.base import BaseVectorizer


class TextVectorizer(BaseVectorizer):
    def __init__(self) -> None:
        super().__init__()
        self.sentence_transformer = SentenceTransformer("all-MiniLM-L6-v2")
        self.sentence_transformer.eval().to("cpu")

    def encode(self, input: list[int | float | str]) -> torch.Tensor:
        """
        Encodes contextual and numerical information of input

        Parameters
        ----------
        input : list[int  |  float  |  str]
            Input to be encoded.

        Returns
        -------
        torch.Tensor
        """
        input_as_string = [str(sentence).lower() for sentence in input]
        embedding_as_numpy_array = np.array(
            self.sentence_transformer.encode(input_as_string), dtype=np.float32
        )

        return torch.Tensor(embedding_as_numpy_array)
