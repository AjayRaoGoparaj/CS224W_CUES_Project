import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for standalone plotting
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import torch
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
import os

# Create a directory to save all the plots
SAVE_DIR = "results"
os.makedirs(SAVE_DIR, exist_ok=True)

# Sample training_logs for demonstration
training_logs = {
    "GCN": [0.6, 0.45, 0.38, 0.34, 0.32],
    "GraphSAGE": [0.7, 0.5, 0.4, 0.35, 0.3],
    "GAT": [0.65, 0.5, 0.42, 0.35, 0.33],
    "GCN_metrics": {"accuracy": 0.85, "precision": 0.83, "recall": 0.84, "f1_score": 0.84},
    "GraphSAGE_metrics": {"accuracy": 0.87, "precision": 0.86, "recall": 0.85, "f1_score": 0.86},
    "GAT_metrics": {"accuracy": 0.88, "precision": 0.87, "recall": 0.86, "f1_score": 0.87},
}

# Visualization: Loss vs Epoch for each model
def visualize_loss(training_logs):
    """Visualize training loss over epochs for different models."""
    plt.figure(figsize=(8, 6))
    for model_name, losses in training_logs.items():
        if "_metrics" not in model_name:
            plt.plot(range(1, len(losses) + 1), losses, label=model_name)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Across Models")
    plt.legend()
    plt.savefig(f"{SAVE_DIR}/training_loss.png")  # Save the plot
    plt.show()

# Visualization: Metric Comparison
def visualize_metrics(training_logs):
    """Visualize metric comparisons (accuracy, precision, recall, F1 score) across models."""
    metrics = ["accuracy", "precision", "recall", "f1_score"]
    model_names = [model.replace("_metrics", "") for model in training_logs if "_metrics" in model]

    for metric in metrics:
        values = [training_logs[model + "_metrics"][metric] for model in model_names]
        plt.figure(figsize=(8, 6))
        plt.bar(model_names, values, alpha=0.7, color='skyblue')
        plt.title(f"{metric.capitalize()} Comparison Across Models")
        plt.xlabel("Models")
        plt.ylabel(f"{metric.capitalize()}")
        plt.ylim(0, 1)
        plt.savefig(f"{SAVE_DIR}/{metric}_comparison.png")  # Save the plot
        plt.show()

# Visualization: Confusion Matrix (Example Prediction)
def visualize_confusion_matrix(y_true, y_pred, model_name):
    """Visualize the confusion matrix for predictions."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_true))
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix for {model_name}")
    plt.savefig(f"{SAVE_DIR}/confusion_matrix_{model_name}.png")  # Save the plot
    plt.show()

# Visualization: Graph Structure (Example Dataset)
def visualize_graph_structure(edge_index, num_nodes=None, node_features=None):
    """Visualize the structure of a graph."""
    data = Data(edge_index=edge_index)
    if node_features is not None:
        data.x = node_features
    if num_nodes is None:
        num_nodes = edge_index.max().item() + 1
    data.num_nodes = num_nodes

    G = to_networkx(data, to_undirected=True)
    plt.figure(figsize=(8, 8))
    nx.draw(G, with_labels=True, node_color="skyblue", node_size=700, font_size=10)
    plt.title("Graph Structure Visualization")
    plt.savefig(f"{SAVE_DIR}/graph_structure.png")  # Save the plot
    plt.show()

# Visualization: Dataset Feature Distribution
def visualize_feature_distribution(features, feature_names=None):
    """Visualize the distribution of features."""
    features = features.numpy()
    num_features = features.shape[1]

    plt.figure(figsize=(12, 6))
    for i in range(num_features):
        sns.histplot(features[:, i], kde=True, label=feature_names[i] if feature_names else f"Feature {i+1}")
    plt.title("Feature Value Distribution Across Nodes")
    plt.xlabel("Feature Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(f"{SAVE_DIR}/feature_distribution.png")  # Save the plot
    plt.show()

# Visualization: Adjacency Matrix
def visualize_adjacency_matrix(adjacency_matrix):
    """Visualize the adjacency matrix of the graph."""
    plt.figure(figsize=(6, 6))
    sns.heatmap(adjacency_matrix.numpy(), annot=True, fmt=".0f", cmap="Blues", cbar=False)
    plt.title("Adjacency Matrix Visualization")
    plt.xlabel("Nodes")
    plt.ylabel("Nodes")
    plt.savefig(f"{SAVE_DIR}/adjacency_matrix.png")  # Save the plot
    plt.show()

# Main Visualization Function
def main_visualize():
    """Main function to execute all visualizations."""
    print("Starting Visualization...\n" + "-" * 50)

    # Loss and Metrics Visualizations
    visualize_loss(training_logs)
    visualize_metrics(training_logs)

    # Example Confusion Matrix
    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 0, 0]
    visualize_confusion_matrix(y_true, y_pred, "Example_Model")

    # Graph Structure Visualization
    edge_index = torch.tensor([[0, 1, 2, 0], [1, 2, 0, 2]])  # Example edge index
    visualize_graph_structure(edge_index)

    # Feature Distribution Visualization
    features = torch.tensor([[1.0, 0.5], [2.0, 1.5], [1.5, 0.75], [2.5, 1.25]])  # Example features
    feature_names = ["Influence Score", "Diversity Metric"]
    visualize_feature_distribution(features, feature_names)

    # Adjacency Matrix Visualization
    adjacency_matrix = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0], [0, 1, 1, 1], [0, 0, 1, 1]])  # Example adjacency matrix
    visualize_adjacency_matrix(adjacency_matrix)

if __name__ == "__main__":
    main_visualize()
