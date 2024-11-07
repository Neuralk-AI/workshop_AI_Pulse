import torch
import numpy as np
from typing import Optional
import matplotlib.pyplot as plt


def pairwise_cosine_similarity_matrix(x1, x2):
    """
    Compute pairwise cosine similarity matrix for two sets of vectors.

    Parameters
    ----------
    x1 : N x D
    x2 : M x D

    Returns
    -------
    torch.Tensor N x M
    """
    eps = 1e-10
    norm_x1, norm_x2 = x1.norm(dim=-1).unsqueeze(1), x2.norm(dim=-1).unsqueeze(1)
    x1_scaled = x1 / torch.max(norm_x1, eps * torch.ones_like(norm_x1))
    x2_scaled = x2 / torch.max(norm_x2, eps * torch.ones_like(norm_x2))
    cosine_similarity_matrix = torch.mm(x1_scaled, x2_scaled.transpose(0, 1))
    return cosine_similarity_matrix


def plot_experiment_1(x, data1, data2, title1, title2):
    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.scatter(data1[:, 0], data1[:, 1], color="blue")
    for i, num in enumerate(x):
        plt.text(data1[i, 0], data1[i, 1], str(num), color="blue")
    plt.title(title1)

    plt.subplot(1, 2, 2)
    shift = np.min(data2[:, 0])
    # data2 = np.log(data2 - shift + 1)
    plt.scatter(data2[:, 0], data2[:, 1], color="green")
    for i, num in enumerate(x):
        plt.text(data2[i, 0], data2[i, 1], str(num), color="green")
    plt.title(title2)

    plt.show()


def plot_experiment_2(
    x,
    data1,
    data2,
    title1="Standard Embeddings for Text with Numbers (TSNE)",
    title2="Custom Embeddings for Text with Numbers (TSNE)",
):
    # Plot the results
    plt.figure(figsize=(12, 8))
    # plt.subplot(1, 2, 1)
    plt.scatter(data1[:, 0], data1[:, 1], color="blue")
    for i, text in enumerate(x):
        plt.text(data1[i, 0], data1[i, 1], text, color="blue")
    plt.title(title1)
    plt.show()

    # Plot the results
    plt.figure(figsize=(12, 8))
    # plt.subplot(1, 2, 1)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.scatter(data2[:5, 0], data2[:5, 1], color="green")
    plt.scatter(data2[5:9, 0], data2[5:9, 1], color="blue")
    plt.scatter(data2[9:, 0], data2[9:, 1], color="red")
    for i, text in enumerate(x):
        color = (
            "green"
            if i in [0, 1, 2, 3, 4]
            else ("blue" if i in [5, 6, 7, 8] else "red")
        )
        plt.text(data2[i, 0], data2[i, 1], text, color=color)
    plt.title(title2)


def plot_experiment_3(
    vectorizer,
    numbers: list,
    numbers_in_text: Optional[list] = None,
    title: str = "test",
    **kwargs
):
    # Generate numbers
    numbers = numbers.unsqueeze(1)  # Shape: (num_numbers, 1)
    num_numbers = numbers.size(0)

    # Compute embeddings
    data2encode = numbers if numbers_in_text is None else numbers_in_text
    embeddings = vectorizer.encode(
        data2encode, **kwargs
    )  # Shape: (num_numbers, embedding_dim)
    if embeddings.dim() > 2:
        embeddings = embeddings.squeeze()

    # Calculate cosine similarities for consecutive embeddings
    cosine_similarities = []
    distances = []
    for i in range(len(numbers) - 1):
        # Calculate cosine similarity between consecutive embeddings
        emb1 = embeddings[i].unsqueeze(0)
        emb2 = embeddings[i + 1].unsqueeze(0)
        similarity = torch.dist(emb1.detach(), emb2.detach(), p=2)
        cosine_similarities.append(similarity)

        # Calculate the distance between consecutive numbers
        distance = abs(numbers[i + 1] - numbers[i])
        distances.append(distance)

    # Convert lists to numpy arrays for plotting
    cosine_similarities = np.array(cosine_similarities)
    cosine_similarities = (
        cosine_similarities - np.median(cosine_similarities)
    ) / np.std(cosine_similarities)
    distances = np.array(distances)
    distances = (distances - np.median(distances)) / np.std(distances)

    # Plot the cosine similarity as a function of distance
    plt.figure(figsize=(10, 6))
    plt.plot(distances, cosine_similarities, marker="o", linestyle="-", color="b")
    plt.xlabel("Distance between numbers")
    plt.ylabel("L2 norm between embedding representations")
    plt.title(title)
    plt.grid(True)
    plt.xlim((-3, 3))
    plt.plot(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100), color="black")
    plt.ylim((-3, 3))
    plt.show()
