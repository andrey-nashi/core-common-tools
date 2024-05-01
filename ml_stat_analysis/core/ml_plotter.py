import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plot_pca_3d(features: np.ndarray, labels: np.ndarray, path_output: str = None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(features[:, 0], features[:, 1], features[:, 2], c=labels, cmap="viridis")

    ax.set_xlabel("pca_f_1")
    ax.set_ylabel("pca_f_2")
    ax.set_zlabel("pca_f_3")
    ax.set_title("3D PCA Plot")

    legend = fig.colorbar(scatter, ax=ax, orientation="vertical", shrink=0.5)
    legend.set_label("labels")

    # -----------------------------
    if path_output is not None:
        plt.savefig(path_output, format="png", dpi=100)
    else: plt.show()

def plot_pca_2d(features: np.ndarray, labels: np.ndarray, path_output: str = None):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(features[:, 0], features[:, 1], c=labels, cmap='viridis', edgecolor='k', alpha=0.7)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('2D PCA Plot')
    plt.colorbar(scatter)

    # -----------------------------
    if path_output is not None:
        plt.savefig(path_output, format="png", dpi=100)
    else: plt.show()


def plot_feature_label_distribution(data: dict, feature_name: str = "Feature", path_output: str = None):
    rows = []
    for category, counts in data.items():
        for label, count in counts.items():
            rows.extend([(category, label)] * count)

    df = pd.DataFrame(rows, columns=[feature_name, "Label"])

    sns.set(style="darkgrid")
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(x=feature_name, hue="Label", data=df)
    ax.set_title(f"Count of {feature_name} by Label")
    ax.set_xlabel(feature_name)
    ax.set_ylabel("Count")

    if path_output is not None:
        plt.savefig(path_output, format="png", dpi=100)
    else: plt.show()

def plot_feature_histogram(data, feature_name: str = "Feature", path_output: str = None):
    categories = [str(i) for i in range(len(data))]
    values = list(data.values())

    plt.figure(figsize=(8, 5))
    plt.bar(categories, values, color='skyblue')
    plt.legend(categories, list(data.keys()))
    plt.xlabel(feature_name)
    plt.ylabel("Count")
    plt.title(f"Histogram of {feature_name}")

    if path_output is not None:
        plt.savefig(path_output, format="png", dpi=100)
    else: plt.show()

def save_confusion_matrix(cnf_matrix, path_image):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cnf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=np.arange(1, 9), yticklabels=np.arange(1, 9))
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.savefig(path_image, format="png", dpi=50)