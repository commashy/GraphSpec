import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle

def load_results(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def plot_cosine_similarity_distributions(cos_sim1, cos_sim2, labels=('Model 1', 'Model 2'), bins=100, save_path=None):
    """
    Plots the distributions of cosine similarities for two models using histograms and KDE.
    
    Parameters:
    - cos_sim1: List or array of cosine similarities for the first model.
    - cos_sim2: List or array of cosine similarities for the second model.
    - labels: Tuple containing the labels for the two models.
    - bins: Number of bins for the histogram.
    - save_path: Path to save the plot image. If None, the plot is not saved.
    """
    # Set up the matplotlib figure
    plt.figure(figsize=(10, 6))
    
    # Plot histograms with normalized KDE
    sns.histplot(cos_sim1, color="skyblue", label=labels[0], bins=bins, kde=True, stat="density", linewidth=0)
    sns.histplot(cos_sim2, color="red", label=labels[1], bins=bins, kde=True, stat="density", linewidth=0)
    
    # Plot customization
    plt.title('Cosine Similarity Distribution')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Density')
    plt.legend(title='Model')
    
    # Show the plot
    plt.tight_layout()
    plt.show()
    
    # Save the plot if a save_path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")

graphspec_results = load_results('test_results.pickle')
seq2ms_results = load_results('test_results2.pickle')

# Extracting cosine similarities assuming each entry in the list has a 'cosine_similarity' key
cos_sim1 = [entry['cosine_similarity'] for entry in graphspec_results]
cos_sim2 = [entry['cosine_similarity'] for entry in seq2ms_results]

# Call the function
plot_cosine_similarity_distributions(cos_sim1, cos_sim2, labels=('GraphSpec Model', 'Seq2MS Model'), save_path='cosine_similarity_distribution.png')
