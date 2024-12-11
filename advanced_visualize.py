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

# Definitions
# Feature A and Feature B:
# These represent node features extracted from the dataset (e.g., node attributes like paper embeddings in a citation graph).
# Feature Distribution:
# This shows the frequency of feature values across nodes, helping us understand the range and overlap between features.

# Helper Functions
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

def visualize_feature_distributions(features):
    """Visualize the distributions of features (Feature A and Feature B)."""
    print("Visualizing Feature Distributions...")
    sns.kdeplot(features[:, 0], label="Feature A (e.g., Embedding Dimension 1)", fill=True)
    sns.kdeplot(features[:, 1], label="Feature B (e.g., Embedding Dimension 2)", fill=True)
    plt.title("Feature Distributions")
    plt.xlabel("Feature Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

def visualize_embeddings(features, method="PCA"):
    """Visualize embeddings using PCA or t-SNE."""
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    if method == "PCA":
        reducer = PCA(n_components=2)
        print("Performing PCA...")
    elif method == "t-SNE":
        reducer = TSNE(n_components=2, perplexity=30, max_iter=300)
        print("Performing t-SNE...")

    reduced_features = reducer.fit_transform(scaled_features)
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c='blue', alpha=0.5)
    plt.title(f"{method} Visualization of Node Features")
    plt.xlabel(f"{method} Component 1")
    plt.ylabel(f"{method} Component 2")
    plt.show()

# Load the dataset and unpack it
print("Loading ogbn-arxiv dataset...")
dataset = NodePropPredDataset(name="ogbn-arxiv")
graph_data, labels = dataset[0]

# Debug prints
print("Graph Data Keys:", graph_data.keys())
print("Node Feature Shape:", graph_data['node_feat'].shape)

# Visualizing Feature Distributions
visualize_feature_distributions(graph_data['node_feat'])

# Visualizing Embeddings
visualize_embeddings(graph_data['node_feat'], method="PCA")
visualize_embeddings(graph_data['node_feat'], method="t-SNE")

# Graph Metrics
print("\nCalculating Centrality Measures...")
calculate_centrality_measures(graph_data)

# Heatmap of Feature Correlations
print("Generating Heatmap of Feature Correlations...")
correlation_matrix = np.corrcoef(graph_data['node_feat'].T)
sns.heatmap(correlation_matrix, annot=False, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()
