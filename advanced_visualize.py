import torch
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import numpy as np
from torch_geometric.utils import to_networkx
from ogb.nodeproppred import NodePropPredDataset
import os

# Create directory to save graphs
SAVE_DIR = "/content/CUES_Project/results"
os.makedirs(SAVE_DIR, exist_ok=True)

# Helper Functions
def visualize_feature_distributions(features, save_path):
    """Visualize the distributions of specific latent node features."""
    print("Visualizing Feature Distributions...")
    plt.figure(figsize=(8, 6))
    sns.kdeplot(features[:, 0], label="Feature A (Latent Dimension: Influence in Network)", fill=True, alpha=0.7)
    sns.kdeplot(features[:, 1], label="Feature B (Latent Dimension: Diversity of Connections)", fill=True, alpha=0.7)
    plt.title("Feature Embedding Distributions for Node Attributes")
    plt.xlabel("Normalized Feature Value (Latent Dimension)")
    plt.ylabel("Density of Node Representation")
    plt.legend()
    plt.savefig(f"{save_path}/feature_distributions.png")
    plt.show()

def visualize_embeddings(features, method, save_path):
    """Visualize embeddings using PCA or t-SNE."""
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    if method == "PCA":
        reducer = PCA(n_components=2)
        print("Performing PCA...")
    elif method == "t-SNE":
        reducer = TSNE(n_components=2, perplexity=30, n_iter=300)
        print("Performing t-SNE...")

    reduced_features = reducer.fit_transform(scaled_features)
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c='blue', alpha=0.5)
    plt.title(f"{method} Visualization of Node Features")
    plt.xlabel(f"{method} Latent Dimension 1")
    plt.ylabel(f"{method} Latent Dimension 2")
    plt.savefig(f"{save_path}/{method.lower()}_embeddings.png")
    plt.show()

def calculate_centrality_measures(graph):
    """Calculate and print centrality measures for the graph."""
    G = to_networkx(graph, to_undirected=True)
    if not nx.is_connected(G):
        print("Graph is not connected. Using the largest connected component.")
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()

    degree_centrality = nx.degree_centrality(G)
    eigenvector_centrality = nx.eigenvector_centrality_numpy(G)
    closeness_centrality = nx.closeness_centrality(G)

    print("\nTop 5 Nodes by Centrality Measures:")
    print("Degree:", sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5])
    print("Eigenvector:", sorted(eigenvector_centrality.items(), key=lambda x: x[1], reverse=True)[:5])
    print("Closeness:", sorted(closeness_centrality.items(), key=lambda x: x[1], reverse=True)[:5])

def generate_heatmap(features, save_path):
    """Generate and save a heatmap of feature correlations."""
    print("Generating Heatmap of Feature Correlations...")
    correlation_matrix = np.corrcoef(features.T)
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=False, cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.savefig(f"{save_path}/feature_correlation_heatmap.png")
    plt.show()

# Main Execution
if __name__ == "__main__":
    print("Loading ogbn-arxiv dataset...")
    dataset = NodePropPredDataset(name="ogbn-arxiv")
    graph_data, labels = dataset[0]

    # Debug prints
    print("Graph Data Keys:", graph_data.keys())
    print("Node Feature Shape:", graph_data['node_feat'].shape)

    # Visualizing Feature Distributions
    visualize_feature_distributions(graph_data['node_feat'], SAVE_DIR)

    # Visualizing Embeddings
    visualize_embeddings(graph_data['node_feat'], method="PCA", save_path=SAVE_DIR)
    visualize_embeddings(graph_data['node_feat'], method="t-SNE", save_path=SAVE_DIR)

    # Graph Metrics
    print("\nCalculating Centrality Measures...")
    calculate_centrality_measures(graph_data)

    # Heatmap of Feature Correlations
    generate_heatmap(graph_data['node_feat'], SAVE_DIR)
