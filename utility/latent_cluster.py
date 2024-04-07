import numpy as np
import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def embeddings(model, dataloader, hidden_dim = 512, verbose = False):
    model.eval() #Evaluation Mode
    features = torch.tensor([])
    iteration_idx = 1

    print("Starting Feature Extraction")
    with torch.no_grad():
        for images, labels in dataloader:
            batch_features = model(images).reshape([len(labels), hidden_dim])
            features = torch.cat((features, batch_features), dim=0)
            if verbose:
                print(f"Iteration {iteration_idx} completed")
                iteration_idx+=1

    print("Features Extracted. Performing PCA...")
    return features.numpy()

def clustering(model, dataset, dataloader, hidden_dim = 512, verbose = False):
    # Perform PCA with 2 components
    data = embeddings(model, dataloader, hidden_dim, verbose)
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(data)
    print("Finished Performing PCA. Onto Plotting")

    # Extract unique labels and sort together with data for legend order
    labels = [x[1] for x in dataset]
    unique_labels, counts = np.unique(labels, return_counts=True)
    sorted_indices = np.argsort(counts)[::-1]  # Sort by count descending
    unique_labels = unique_labels[sorted_indices]

    # Separate data by label for plotting with legend
    data_by_label = {label: [] for label in unique_labels}
    for i, label in enumerate(labels):
        data_by_label[label].append(principal_components[i])

    # Create the plot
    colors = plt.cm.get_cmap('tab10')(np.arange(len(unique_labels)) % len(plt.cm.get_cmap('tab10').colors))  # Generate colors for legend

    for label, data_points in data_by_label.items():
        x, y = zip(*data_points)  # Unpack data points
        plt.scatter(x, y, label=label, color=colors[list(data_by_label.keys()).index(label)])

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA Visualization (2D)")
    plt.legend()
    plt.show()