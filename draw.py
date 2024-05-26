import numpy as np
import matplotlib.pyplot as plt
import math
import pickle
import random

def sparse(x, y, th=0.02):
    x = np.asarray(x, dtype='float32')
    y = np.asarray(y, dtype='float32')
    y /= np.max(y)
    return x[y > th], y[y > th] * 100

def plot_spectra(input_spectra, graphspec_output, seq2ms_output, save_path=None):
    precision = 0.1
    low = 0
    dim = 20000
    imz = np.arange(0, dim, dtype='int32') * precision + low
    
    # Process spectra with sparse function
    input_mzs, input_its = sparse(imz, input_spectra)
    graphspec_mzs, graphspec_its = sparse(imz, graphspec_output)
    seq2ms_mzs, seq2ms_its = sparse(imz, seq2ms_output)
    
    # Create a mirror plot
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(30, 7), gridspec_kw={'height_ratios': [1, 1]})
    fig.subplots_adjust(hspace=.0)  # Eliminate space between subplots

    # Plotting the input spectrum on top
    axs[0].stem(input_mzs, input_its, linefmt="C0-", markerfmt=" ", basefmt=" ")

    # Plotting the GraphSpec and Seq2MS outputs below
    axs[1].stem(graphspec_mzs, graphspec_its, linefmt="C1-", markerfmt=" ", basefmt=" ", label='GraphSpec Output')
    axs[1].stem(seq2ms_mzs, seq2ms_its, linefmt="C2-", markerfmt=" ", basefmt=" ", label='Seq2MS Output')
    # axs[1].set_xlabel('m/z')
    axs[1].invert_yaxis()  # Also invert this axis to align it like a mirror with the top plot

    # Increase tick precision on the bottom graph
    axs[1].xaxis.set_minor_locator(plt.MultipleLocator(50))  # Set minor ticks every 50
    axs[1].xaxis.set_major_locator(plt.MultipleLocator(250))  # Set major ticks every 250

    axs[0].yaxis.set_minor_locator(plt.MultipleLocator(5))  # Set minor ticks every 5
    axs[0].yaxis.set_major_locator(plt.MultipleLocator(25))  # Set major ticks every 20
    axs[0].set_yticklabels([f"{int(tick)}%" for tick in axs[0].get_yticks()])  # Set y-axis tick labels as percentages
    
    axs[1].yaxis.set_minor_locator(plt.MultipleLocator(5))  # Set minor ticks every 5
    axs[1].yaxis.set_major_locator(plt.MultipleLocator(25))  # Set major ticks every 20
    axs[1].set_yticklabels([f"{int(tick)}%" for tick in axs[1].get_yticks()])  # Set y-axis tick labels as percentages

    # Ensure both subplots use the same x-axis range
    min_mz = min(0, min(input_mzs), min(graphspec_mzs), min(seq2ms_mzs))
    max_mz = max(2000, max(input_mzs.max(), graphspec_mzs.max(), seq2ms_mzs.max()))
    axs[0].set_xlim(min_mz, max_mz)
    axs[1].set_xlim(min_mz, max_mz)
    axs[0].set_ylim(0, 100)
    axs[1].set_ylim(100, 0)

    axs[0].spines[['right', 'top']].set_visible(False)  # Hide the right and top spines
    axs[1].spines[['right', 'top']].set_visible(False)  # Hide the right and top spines

    # remove vertical gap between subplots
    plt.subplots_adjust(hspace=.0)

    # Show the plot
    plt.show()

    # Save the figure
    if save_path:
        fig.savefig(save_path, format='png', dpi=300, transparent=True)  # Save as PNG with high dpi for better quality
        print(f"Plot saved as {save_path}")

def load_comparison_results(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

comparison_results = load_comparison_results('comparison_results.pickle')

# Select a random entry from the results, ensure at least one entry exists
if comparison_results:
    selected_entry = random.choice(comparison_results)  # Select a random entry

# Example usage
input_spectra = selected_entry['input_spectra']
graphspec_output = selected_entry['graphspec_output']
seq2ms_output = selected_entry['seq2ms_output']

plot_spectra(input_spectra, graphspec_output, seq2ms_output, save_path='output_plot.png')

print("Original Sequence:", selected_entry['original_sequence'])
print("GraphSpec Cosine Similarity:", selected_entry['graphspec_cosine_similarity'])
print("Seq2MS Cosine Similarity:", selected_entry['seq2ms_cosine_similarity'])
