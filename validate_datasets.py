import pandas as pd
from ogb.linkproppred import LinkPropPredDataset
from ogb.nodeproppred import NodePropPredDataset

# Function to validate the facebook_combined dataset
def validate_facebook_combined():
    # Update the correct path for facebook_combined.txt
    facebook_path = "data/facebook_combined/facebook_combined.txt"
    try:
        # Load the Facebook dataset
        facebook_data = pd.read_csv(facebook_path, sep=" ", header=None, names=["source", "target"])
        print("Facebook Dataset Loaded Successfully!")
        print(f"Number of edges: {len(facebook_data)}")
        print(f"Sample Data:\n{facebook_data.head()}")
    except FileNotFoundError as e:
        print(f"Error: {e}")

# Function to validate ogbl-ddi dataset
def validate_ogbl_ddi():
    try:
        dataset_ddi = LinkPropPredDataset(name='ogbl-ddi')
        data_ddi = dataset_ddi[0]
        print("ogbl-ddi Dataset Loaded Successfully!")
        print(f"Number of nodes: {data_ddi['num_nodes']}")
        print(f"Number of edges: {data_ddi['edge_index'].shape[1]}")
        print(f"Edge Index Sample:\n{data_ddi['edge_index'][:, :5]}")
    except Exception as e:
        print(f"Error validating ogbl-ddi: {e}")

# Function to validate ogbn-arxiv dataset
def validate_ogbn_arxiv():
    try:
        dataset_arxiv = NodePropPredDataset(name='ogbn-arxiv')
        data_arxiv = dataset_arxiv[0]
        print("ogbn-arxiv Dataset Loaded Successfully!")
        print(f"Number of nodes: {data_arxiv[0]['num_nodes']}")
        print(f"Number of edges: {data_arxiv[0]['edge_index'].shape[1]}")
        print(f"Node Feature Shape: {data_arxiv[0]['node_feat'].shape}")
        print(f"Labels Shape: {data_arxiv[1].shape}")
    except Exception as e:
        print(f"Error validating ogbn-arxiv: {e}")

if __name__ == "__main__":
    print("Validating Datasets...\n" + "-"*50)
    validate_facebook_combined()
    print("\n" + "-"*50)
    validate_ogbl_ddi()
    print("\n" + "-"*50)
    validate_ogbn_arxiv()
