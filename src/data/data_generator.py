"""Sample dataset generator for testing and demonstration."""

import json
import random
from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)

SAMPLE_QA_PAIRS = [
    {
        "question": "What is machine learning?",
        "answer": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing computer programs that can access data and use it to learn for themselves.",
    },
    {
        "question": "Explain the difference between supervised and unsupervised learning.",
        "answer": "Supervised learning uses labeled data where the correct output is known, training the model to predict outcomes. Unsupervised learning works with unlabeled data, finding hidden patterns or structures without predefined categories. Supervised learning is used for classification and regression, while unsupervised learning is used for clustering and dimensionality reduction.",
    },
    {
        "question": "What is a neural network?",
        "answer": "A neural network is a computational model inspired by biological neural networks in animal brains. It consists of interconnected nodes (neurons) organized in layers that process information. Each connection has a weight that adjusts as learning proceeds, allowing the network to recognize patterns and make decisions.",
    },
    {
        "question": "What is overfitting in machine learning?",
        "answer": "Overfitting occurs when a model learns the training data too well, including noise and outliers, resulting in poor performance on new, unseen data. The model becomes too complex and captures patterns that don't generalize. Techniques like regularization, cross-validation, and using more training data can help prevent overfitting.",
    },
    {
        "question": "Explain gradient descent.",
        "answer": "Gradient descent is an optimization algorithm used to minimize the cost function in machine learning models. It iteratively adjusts model parameters in the direction of steepest descent of the cost function. The algorithm calculates gradients and updates parameters by moving in the opposite direction of the gradient, scaled by a learning rate.",
    },
    {
        "question": "What is transfer learning?",
        "answer": "Transfer learning is a technique where a model trained on one task is repurposed for a related task. Instead of training from scratch, the model leverages knowledge gained from the original task. This approach is particularly useful when you have limited data for the new task, as it can significantly reduce training time and improve performance.",
    },
    {
        "question": "What are hyperparameters?",
        "answer": "Hyperparameters are configuration settings used to control the learning process of a machine learning model. Unlike model parameters that are learned during training, hyperparameters are set before training begins. Examples include learning rate, batch size, number of layers, and regularization strength. Tuning hyperparameters is crucial for optimal model performance.",
    },
    {
        "question": "Explain precision and recall.",
        "answer": "Precision measures the accuracy of positive predictions - the ratio of true positives to all predicted positives. Recall measures the coverage of actual positives - the ratio of true positives to all actual positives. High precision means few false positives, while high recall means few false negatives. The F1 score combines both metrics into a single value.",
    },
    {
        "question": "What is a convolutional neural network?",
        "answer": "A Convolutional Neural Network (CNN) is a deep learning architecture designed for processing grid-like data, particularly images. CNNs use convolutional layers that apply filters to detect features like edges, textures, and patterns. They include pooling layers for dimensionality reduction and fully connected layers for classification, making them highly effective for computer vision tasks.",
    },
    {
        "question": "What is the purpose of activation functions?",
        "answer": "Activation functions introduce non-linearity into neural networks, enabling them to learn complex patterns. Without activation functions, neural networks would only learn linear relationships. Common activation functions include ReLU (Rectified Linear Unit), sigmoid, and tanh. They determine whether a neuron should be activated based on the weighted sum of inputs.",
    },
]


def generate_sample_dataset(
        output_dir: str = "data/processed", num_samples: int = 100, train_split: float = 0.8
) -> None:
    """
    Generate a sample Q&A dataset for demonstration and testing.

    Args:
        output_dir: Directory to save the generated dataset
        num_samples: Total number of samples to generate
        train_split: Fraction of data for training
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating {num_samples} sample Q&A pairs...")

    all_samples = []
    for _ in range(num_samples):
        qa_pair = random.choice(SAMPLE_QA_PAIRS)
        sample = {
            "text": f"<|user|>\n{qa_pair['question']}\n<|assistant|>\n{qa_pair['answer']}",
            "question": qa_pair["question"],
            "answer": qa_pair["answer"],
        }
        all_samples.append(sample)

    random.shuffle(all_samples)

    train_size = int(len(all_samples) * train_split)
    eval_size = (len(all_samples) - train_size) // 2

    train_data = all_samples[:train_size]
    eval_data = all_samples[train_size: train_size + eval_size]
    test_data = all_samples[train_size + eval_size:]

    train_file = output_path / "train.jsonl"
    eval_file = output_path / "eval.jsonl"
    test_file = output_path / "test.jsonl"

    with open(train_file, "w") as f:
        for sample in train_data:
            f.write(json.dumps(sample) + "\n")

    with open(eval_file, "w") as f:
        for sample in eval_data:
            f.write(json.dumps(sample) + "\n")

    with open(test_file, "w") as f:
        for sample in test_data:
            f.write(json.dumps(sample) + "\n")

    logger.info(f"Generated {len(train_data)} training samples -> {train_file}")
    logger.info(f"Generated {len(eval_data)} evaluation samples -> {eval_file}")
    logger.info(f"Generated {len(test_data)} test samples -> {test_file}")


if __name__ == "__main__":
    generate_sample_dataset()
