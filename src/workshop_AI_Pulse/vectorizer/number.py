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
        Encodes contextual and numerical information of input

        Parameters
        ----------
        input : list[int  |  float  |  str]
            Input to be encoded.

        Returns
        -------
        torch.Tensor
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

    def extract_number(self, input: int | float | str) -> float:
        """
        Extracts number from a sentence. i.e.
            '2 dollars' -> 2

        Parameters
        ----------
        input : int | float | str
            Sentence containing a number. If no number found returns -1.

        Returns
        -------
        float
        """
        if type(input) is str:
            for possible_number in input.split(" "):
                try:
                    return float(possible_number)
                except ValueError:
                    continue
        else:
            return float(input)

        return float("nan")

    def extract_text(self, input: int | float | str) -> float:
        """
        Extracts text from a sentence. i.e.
            '2 dollars' -> '2 dollars'
            'dollars' -> 'dollars'
            2 -> None
            '2' -> None

        Parameters
        ----------
        input : int | float | str
            Sentence containing a number. If no number found returns -1.

        Returns
        -------
        float
        """
        if isinstance(input, float) or isinstance(input, int):
            return None
        try:
            _ = float(input)
            return None
        except ValueError:
            return input

    def encode_numbers(
        self, input: torch.Tensor, threshold: int = None
    ) -> torch.Tensor:
        """
        Sigmoid type embedding of logarithm of input. e.g. sigmoid(log(input + 1))

        Parameters
        ----------
        x : torch.FloatTensor
            torch tensor of shape N with numbers as float or int.

        Returns
        -------
        torch.Tensor
        """
        if self.method == "sinusoid":
            return self.encode_sinusoid(input=input, threshold=threshold)
        elif self.method == "sigmoid":
            return self.encode_sigmoid(input=input, threshold=threshold)
        elif self.method == "logarithmic":
            return self.encode_logarithmic(input=input, threshold=threshold)
        else:
            raise NotImplementedError("TODO")

    def encode_sigmoid(
        self, input: torch.Tensor, threshold: int = None
    ) -> torch.Tensor:
        """
        Sigmoid type embedding of logarithm of input. e.g. sigmoid(log(input + 1))

        Parameters
        ----------
        x : torch.FloatTensor
            torch tensor of shape N with numbers as float or int.

        Returns
        -------
        torch.Tensor
        """
        # We shift the input to remove negative values
        threshold = threshold if threshold is not None else 0  # torch.min(input).item()
        input -= threshold
        input = torch.nan_to_num(input)
        extended_input = input.unsqueeze(1)

        return 2 - 1 / (2 + extended_input)

    def encode_logarithmic(
        self, input: torch.Tensor, threshold: int = None
    ) -> torch.Tensor:
        """
        Shifted logarithmical embedding. e.g. log(input + 1 + e)

        Parameters
        ----------
        x : torch.FloatTensor
            torch tensor of shape N with numbers as float or int.

        Returns
        -------
        torch.Tensor
        """
        # We shift the input to remove negative values
        threshold = threshold if threshold is not None else torch.min(input).item()
        input -= threshold
        input = torch.nan_to_num(input)
        extended_input = input.unsqueeze(1)

        return torch.log(extended_input + 1 + torch.e)

    def encode_sinusoid(
        self, input: torch.Tensor, threshold: int = None
    ) -> torch.Tensor:
        """
        Sinusoidal encoding from original transformers paper.

        Example:
        >>> x = torch.Tensor([0, 1, 2])
        >>> sinusoidal_encoding(x) # embedding size is explicitly assumed to be 2
        tensor([[ 0.8415,  0.5403],
                [ 0.9093, -0.4161],
                [ 0.1411, -0.9900]])

        Parameters
        ----------
        x : torch.FloatTensor
            torch tensor of shape N with numbers as float or int.

        Returns
        -------
        torch.Tensor
        """
        sinusoid_arguments = input.unsqueeze(1) + 1 * self.division_term
        print(sinusoid_arguments.shape)
        output = torch.empty_like(sinusoid_arguments)
        output[:, ::2] = torch.sin(sinusoid_arguments[:, ::2])
        output[:, 1::2] = torch.cos(sinusoid_arguments[:, 1::2])
        return output

    def init_method_params(self):
        """Initialize custom aprameters depending on the method chosen.

        Parameters
        ----------
        method : str
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
