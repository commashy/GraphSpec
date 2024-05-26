import pickle
import random
import numpy as np

def load_comparison_results(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def save_to_text_file(data, filename):
    with open(filename, 'w') as file:
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                file.write(f"{key}: {np.array2string(value, precision=2, separator=', ', threshold=np.inf)}\n")
            else:
                file.write(f"{key}: {value}\n")

# Load the comparison results
comparison_results = load_comparison_results('comparison_results.pickle')

# Select a random entry from the results, ensure at least one entry exists
if comparison_results:
    selected_entry = random.choice(comparison_results)  # Select a random entry
    # Save the selected entry to a text file
    save_to_text_file(selected_entry, 'selected_entry.txt')
    print("Random data has been saved to 'selected_entry.txt'.")
else:
    print("No entries found in the comparison results.")
