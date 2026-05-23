import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

def run_exploratory_pca(X_train, y_train):
    """
    Task 2.7: Runs PCA on the scaled training data, prints loadings, and generates plots.
    """
    print("\n--- Task 2.7: Exploratory PCA ---")
    
    # Run PCA on all components
    pca = PCA()
    X_pca = pca.fit_transform(X_train)
    
    # 1. Plot the Scree Plot (Cumulative Explained Variance)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
             np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--')
    plt.title('PCA Scree Plot')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.grid(True)
    plt.savefig('pca_scree_plot.png')
    plt.close()
    
    # 2. Inspect the Loadings (Which features matter most for PC1?)
    loadings = pd.DataFrame(pca.components_.T, columns=[f'PC{i+1}' for i in range(pca.n_components_)], index=X_train.columns)
    print("Top features contributing to PC1:")
    print(loadings['PC1'].abs().sort_values(ascending=False).head(4))
    
    # 3. 2D Scatter Plot
    plt.figure(figsize=(8, 6))
    # We plot the first two columns of X_pca (PC1 and PC2)
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y_train, alpha=0.6, palette="coolwarm")
    plt.title('2D PCA Projection (Colored by Revenue)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.savefig('pca_2d_scatter.png')
    plt.close()
    
    print("PCA complete! Saved 'pca_scree_plot.png' and 'pca_2d_scatter.png' to your folder.")