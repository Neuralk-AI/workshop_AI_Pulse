from typing import List, Union
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from workshop_AI_Pulse.vectorizer.base import BaseVectorizer


class TextVectorizer(BaseVectorizer):
    """
    A vectorizer for textual input that uses a pre-trained SentenceTransformer model
    to generate embeddings.

    Attributes
    ----------
    model_name : str
        Name of the SentenceTransformer model to use.
    device : str
        Device for computation, e.g., 'cpu' or 'cuda'.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu") -> None:
        """
        Initialize the TextVectorizer with a pre-trained SentenceTransformer model.

        Parameters
        ----------
        model_name : str, optional
            Name of the pre-trained SentenceTransformer model (default: "all-MiniLM-L6-v2").
        device : str, optional
            Device for computation, e.g., 'cpu' or 'cuda' (default: "cpu").
        """
        super().__init__()
        self.model_name = model_name
        self.device = device
        self._sentence_transformer = None  # Lazy initialization

    @property
    def sentence_transformer(self) -> SentenceTransformer:
        """Lazy-load the SentenceTransformer model."""
        if self._sentence_transformer is None:
            self._sentence_transformer = SentenceTransformer(self.model_name)
            self._sentence_transformer.eval().to(self.device)
        return self._sentence_transformer

    def encode(self, inputs: List[Union[int, float, str]]) -> torch.Tensor:
        """
        Generate embeddings for a list of inputs.

        Parameters
        ----------
        inputs : List[Union[int, float, str]]
            A list of input values to be encoded.

        Returns
        -------
        torch.Tensor
            A tensor of shape (N, D), where N is the number of inputs and D is the embedding dimension.

        Raises
        ------
        ValueError
            If the input list is empty or contains unsupported types.
        """
        if not inputs:
            raise ValueError("Input list cannot be empty.")

        # Convert inputs to lowercase strings
        input_strings = [str(item).lower() for item in inputs]

        # Generate embeddings
        embeddings = self.sentence_transformer.encode(input_strings, convert_to_numpy=True)

        return torch.tensor(embeddings, dtype=torch.float32, device=self.device)
