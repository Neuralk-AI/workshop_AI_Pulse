# workshop_AI_Pulse: Hands-On Integration: Simulating Embedding Models for Structured Data

During this workshop we will see the importance of taking into consideration the internal structure of the object that we aim to vectorize.

Given the time constraints we have, we will focus on a smaller use-case: `number representation in strings`.
We will see :
- that current methods to represent strings containing numbers are limited 
- and how we can improve the representation quality of these strings.

# Investigating number representation in strings

This Python package is designed to create robust representations of numbers, whether they appear as standalone values or embedded within strings. This package is essential for anyone working with numerical data in natural language, as it improves the way numbers are understood and embedded in text, going beyond the limitations of traditional language model embeddings.


## Background and Motivation
In natural language processing (NLP), language models are widely used to represent words, sentences, and other text structures in vector space, capturing semantic meaning. However, these models struggle with numeric information, especially when numbers are embedded in sentences (e.g., “The stock rose by 5.7% today”). Most language models treat numbers as mere tokens, which ignores their quantitative meaning and fails to capture their relative magnitude or ordinal position.

Building better representations of numbers is critical in fields that rely on numerical information, such as finance, science, healthcare, and more. If numbers in text can be represented in a way that preserves their semantic and quantitative value, language models can perform better in tasks like numerical reasoning, quantitative question answering, and data analysis from text.

## Why Current Language Models Struggle with Numbers
Language models trained primarily on text data are not optimized to handle numbers in a meaningful way. For example, while a model might recognize that “1000” is a large number, it may not understand that it is exactly twice “500” or that “0.5” is a fraction. These nuances are often lost because the models do not inherently understand numerical concepts or relationships, leading to a lack of precision in tasks that depend on quantitative reasoning.

This package aims to address these limitations by developing vectorizers that treat numbers intelligently, enabling models to represent and distinguish numerical values accurately, even when they appear within text.




## About the Package
The package contains a set of vectorizers that transform numbers (and text containing numbers) into embeddings that capture both semantic and quantitative information. These vectorizers provide a foundation for improving number representation in embeddings and are easy to integrate with NLP models.


## Set up

```bash
git clone https://github.com/Neuralk-AI/workshop_AI_Pulse
cd workshop_AI_Pulse
conda create -n workshop_neuralk python=3.10
conda activate workshop_neuralk
pip install -r requirements.txt
```

## Vectorizer Classes
Each vectorizer class in this package inherits from a common base class, BaseVectorizer, which provides a template for encoding input in embedding space.

```python
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
```

The encode method accepts lists of integers, floats, or strings and returns a tensor representation of each element. This allows the package to handle mixed input types seamlessly, whether the data is purely numerical, purely textual, or a combination of both.


## Workshop: Using and Experimenting with Vectorizers
The objective of this workshop is to demonstrate the power and limitations of traditional embeddings in comparison to the custom vectorizers in this package.

### Experiment 1: Representing Standalone Numbers
This experiment shows how standalone numbers are embedded by both standard language model embeddings and our custom vectorizers.



### Experiment 2: Representing Numbers in Text
This experiment demonstrates the difference in handling numbers embedded in text. Standard embeddings often ignore or tokenize numbers without preserving their quantitative value.


### Experiment 3: Quantitative Reasoning
This experiment will show how embeddings from custom vectorizers can potentially improve model performance on quantitative reasoning tasks.
