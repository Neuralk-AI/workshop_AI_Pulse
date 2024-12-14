import torch

from workshop_AI_Pulse.vectorizer.base import BaseVectorizer
from workshop_AI_Pulse.vectorizer.text import TextVectorizer


class NumberVectorizer(BaseVectorizer):
    def __init__(
        self, method: str = "sigmoid", merge: str = "*", alpha: float = 1.0
    ) -> None:
        super().__init__()
        self.method = method
        self.merge = merge
        self.alpha = alpha
        self.text_vectorizer = TextVectorizer()

        self.embedding_size = (
            self.text_vectorizer.sentence_transformer.get_sentence_embedding_dimension()
        )
        self.init_method_params()

    def encode(
        self, input: list[int | float | str], threshold: int = None
    ) -> torch.Tensor:
        """
        Project the input into a latent space.

        Args:
            - input : list[int  |  float  |  str]

        Returns:
            - torch.Tensor
        """
        numbers_from_inputs = torch.Tensor(
            [self.extract_number(sentence) for sentence in input]
        )
        numerical_embeddings = self.encode_numbers(
            numbers_from_inputs, threshold=threshold
        )

        contextual_embeddings = []
        for i, inp in enumerate(input):
            inp = self.extract_text(inp)
            if inp is None:
                if self.merge == "*":
                    contextual_embeddings.append(
                        torch.ones(self.embedding_size).unsqueeze(0)
                    )
                elif self.merge == "+":
                    contextual_embeddings.append(
                        torch.zeros(self.embedding_size).unsqueeze(0)
                    )
            else:
                contextual_embeddings.append(self.text_vectorizer.encode([inp]))
        contextual_embeddings = torch.cat(contextual_embeddings, dim=0)

        if self.merge == "*":
            return numerical_embeddings * contextual_embeddings
        elif self.merge == "+":
            return self.alpha * numerical_embeddings + contextual_embeddings
        elif self.merge == "concat":
            return torch.cat([numerical_embeddings, contextual_embeddings], dim=-1)
        else:
            raise NotImplementedError("TODO")
        
    
    def extract_number(self, input_value: int | float | str) -> float:
        """
        Extract a number from the input. If no number is found, returns NaN.

        Args:
            - input_value : int | float | str
        Returns:
            - float
        """
        if isinstance(input_value, (int, float)):
            return float(input_value)
        if isinstance(input_value, str):
            for token in input_value.split():
                try:
                    return float(token)
                except ValueError:
                    continue
        return float("nan")


    def extract_text(input_value: int | float | str) -> str | None:
        """
        Extract the text from the input. If the input is a number or can be cast to one, returns None.

        Args:
            - input_value : int | float | str
        Returns:
            - str | None
        """
        if isinstance(input_value, (int, float)):
            return None
        try:
            _ = float(input_value)
            return None
        except ValueError:
            return input_value

    def encode_numbers(
        self, input: torch.Tensor, threshold: int = None
    ) -> torch.Tensor:
        """
        Sigmoid type embedding of logarithm of input. e.g. sigmoid(log(input + 1))

        Args:
            - x : torch.FloatTensor
            torch tensor of shape N with numbers as float or int.

        Returns:
            - torch.Tensor
        """
        if self.method == "sinusoid":
            return self.encode_sinusoid(input=input, threshold=threshold)
        elif self.method == "sigmoid":
            return self.encode_sigmoid(input=input, threshold=threshold)
        elif self.method == "logarithmic":
            return self.encode_logarithmic(input=input, threshold=threshold)
        else:
            raise NotImplementedError("TODO")
        
    def encode_sigmoid(self, input_tensor: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
        """
        Apply a sigmoid-like transformation to the logarithm of the input.

        Args:
            - input_tensor : torch.Tensor
            Tensor of shape (N,) with numeric values.
            - threshold : float, optional
            Value to shift the input by, default is 0.0.

        Returns:
            - torch.Tensor
            Transformed tensor of shape (N, 1).
        """
        input_tensor = torch.nan_to_num(input_tensor - threshold)
        return 2 - 1 / (2 + input_tensor.unsqueeze(-1))

    def encode_logarithmic(self, input_tensor: torch.Tensor, threshold: float = None) -> torch.Tensor:
        """
        Apply a shifted logarithmic transformation to the input.

        Args:
            - input_tensor : torch.Tensor
            Tensor of shape (N,) with numeric values.
            - threshold : float, optional
            Value to shift the input by. Default is the minimum value in the tensor.

        Returns:
            - torch.Tensor
            Transformed tensor of shape (N, 1).
        """
        if threshold is None:
            threshold = input_tensor.min().item()
        input_tensor = torch.nan_to_num(input_tensor - threshold)
        return torch.log(input_tensor.unsqueeze(-1) + 1 + torch.e)


    def encode_sinusoid(self, input_tensor: torch.Tensor, division_term: torch.Tensor) -> torch.Tensor:
        """
        Compute sinusoidal encoding as in the original Transformer paper.

        Args:
            - input_tensor : torch.Tensor
            Tensor of shape (N,) with numeric values.
            - division_term : torch.Tensor
            Precomputed division term for sinusoidal encoding.

        Returns:
            - torch.Tensor
            Sinusoidally encoded tensor.
        """
        # Expand input_tensor to shape (N, 1) and broadcast with division_term (D,)
        sinusoid_args = input_tensor.unsqueeze(-1) * division_term

        # Compute sinusoidal encodings
        sin_encodings = torch.sin(sinusoid_args[:, 0::2])  # Even indices
        cos_encodings = torch.cos(sinusoid_args[:, 1::2])  # Odd indices

        # Concatenate sin and cos encodings along the last dimension
        encoded = torch.empty_like(sinusoid_args)
        encoded[:, 0::2] = sin_encodings
        encoded[:, 1::2] = cos_encodings
        
        return encoded

    def init_method_params(self):
        """Initialize custom aprameters depending on the method chosen.

        Args:
            - method : str
            name of the encoding method
        """
        if self.method == "sinusoid":
            self._init_sinusoid()
        elif self.method == "sigmoid":
            self._init_sigmoid()
        elif self.method == "logarithmic":
            self._init_logarithmic()
        else:
            raise NotImplementedError("TODO")

    def _init_sinusoid(self):
        """Initialize the parameters for sinusoid encoding scheme."""
        # Compute coefficients for sinusoidal embedding
        log_inverse_10000 = -9.210340371976184
        arange_repeat_twice = (torch.arange(0, self.embedding_size) // 2 * 2).unsqueeze(
            0
        )
        log_division_term = log_inverse_10000 * (
            arange_repeat_twice / self.embedding_size
        )

        # Cache computation for sinusoidal embedding
        self.division_term = torch.exp(log_division_term)

    def _init_sigmoid(self):
        """Initialize the parameters for sigmoid encoding scheme."""
        pass

    def _init_logarithmic(self):
        """Initialize the parameters for logarithmic encoding scheme."""
        pass
