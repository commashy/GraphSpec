import pickle
import numpy as np
from tqdm import tqdm

def load_results(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def create_index(results):
    """ Create an index for quick lookup using `seq_ptms` as the key. """
    index = {}
    print("Indexing results...")
    for res in tqdm(results, desc="Building Index"):
        key = res['seq_ptms'][0]  # Directly use `seq_ptms` as the key
        if key not in index:
            index[key] = []
        index[key].append(res)
    return index

def filter_and_compare(graphspec_index, seq2ms_index, similarity_threshold1, similarity_threshold2, difference_threshold):
    comparison_results = []
    print("Comparing results...")
    # Progress bar for GraphSpec index traversal
    progress_bar = tqdm(total=len(graphspec_index), desc="Filtering and Comparing")
    
    for key, gs_results in graphspec_index.items():
        if key in seq2ms_index:
            for gs_result in gs_results:
                if gs_result['cosine_similarity'] > similarity_threshold1:
                    for seq2ms_result in seq2ms_index[key]:
                        similarity_difference = abs(gs_result['cosine_similarity'] - seq2ms_result['cosine_similarity'])
                        if similarity_difference > difference_threshold and seq2ms_result['cosine_similarity'] > similarity_threshold2 and seq2ms_result['cosine_similarity'] < 0.76:
                            comparison_results.append({
                                'original_sequence': key,  # Now, key is `seq_ptms`
                                'input_spectra': gs_result['input'],
                                'graphspec_output': gs_result['prediction'],
                                'graphspec_cosine_similarity': gs_result['cosine_similarity'],
                                'seq2ms_output': seq2ms_result['prediction'],
                                'seq2ms_cosine_similarity': seq2ms_result['cosine_similarity'],
                                'similarity_difference': similarity_difference
                            })
        progress_bar.update(1)
    progress_bar.close()
    return comparison_results

# Load the test results
graphspec_results = load_results('test_results.pickle')
seq2ms_results = load_results('test_results2.pickle')
print(f"Loaded {len(graphspec_results)} GraphSpec results and {len(seq2ms_results)} Seq2MS results.")
print(graphspec_results[0])
print(seq2ms_results[0])

# Create indices
graphspec_index = create_index(graphspec_results)
seq2ms_index = create_index(seq2ms_results)
print("Indexing complete.")

# Set the thresholds
similarity_threshold1 = 0.9
similarity_threshold2 = 0.74
difference_threshold = 0.15

# Perform the filtering and comparison
final_results = filter_and_compare(graphspec_index, seq2ms_index, similarity_threshold1, similarity_threshold2, difference_threshold)

# Optionally, save the final results to a file
with open('comparison_results.pickle', 'wb') as f:
    pickle.dump(final_results, f)

print(f"Found {len(final_results)} entries matching the criteria.")
