from utils import *
from AmorProt import AmorProtV2
from collections import Counter

# def debug_encoding_for_peptides(data):
#     """
#     Function to debug and print the encoding for all peptides.

#     :param data: The dataset containing peptide information.
#     """
#     lengths = [len(peptide_data['Sequence']) for peptide_data in data]
#     length_counts = Counter(lengths)

#     for length, count in length_counts.items():
#         print(f"{length}: {count}")

# def debug_encoding_for_peptides(data, num_peptides=5):
#     """
#     Function to debug and print the encoding for a few peptides.

#     :param data: The dataset containing peptide information.
#     :param num_peptides: Number of peptides to process for debugging.
#     """
#     for i in range(num_peptides):
#         peptide_data = data[i]
#         embed = asnp32(embed_maxquant_with_global_features(peptide_data, fixedaugment=False, key='N', havelong=True))
#         print(f"Peptide {i+1}: {peptide_data['Sequence']}")
#         print(f"Encoded Features: \n{embed}\n")

# def save_encoded_features_to_file(data, file_path="encoded_features.txt", num_peptides=5):
#     with open(file_path, 'w') as f:
#         for i in range(num_peptides):
#             peptide_data = data[i]
#             embed = asnp32(embed_maxquant_with_global_features(peptide_data, fixedaugment=False, key='N', havelong=True))
#             f.write(f"Peptide {i+1}: {peptide_data['Sequence']}\n")
#             f.write("Encoded Features:\n")

#             # Iterate over each row in the encoded features array
#             for row in embed:
#                 # Convert each element in the row to a string and join them with a comma
#                 row_str = ', '.join([f"{elem:.4f}" if isinstance(elem, float) else str(elem) for elem in row])
#                 f.write(row_str + "\n")
#             f.write("\n")

# # Assuming you have a function like readmgf that reads the data and returns a list of peptide data
# print('Loading data for debugging...')
# debug_data = readmgf('/home/johaa/swinms/dataset/NIST.mgf')  # Update the path to your MGF file
# debug_encoding_for_peptides(debug_data)

ap = AmorProtV2(maccs=True, ecfp4=False, ecfp6=False, rdkit=False, graph=False)
with open('/data/jerry/ptm/linked_data.pickle', 'rb') as f:
    linked_data = pickle.load(f)
print('linked data len:', len(linked_data))

random.Random(4).shuffle(linked_data)
first_data = linked_data[0]
print('first_data:', first_data)

# embedding = combined_embedding_function(first_data, ap)
# print('embedding:', embedding)


# mod_seq = '_K(ac)K(bi)K(bu)K(cr)K(di)K(fo)K(gl)K(hy)K(ma)K(me)K(pr)K(su)K(tr)M(ox)R(ci)R(di)R(me)P(hy)Y(ni)Y(ph)_'

# # Extract sequence and PTMs
# seq_ptms = find_mod2(mod_seq)

# # print(seq_ptms)

# # Adjust the sequence length to MAX_PEPTIDE_LENGTH and prepare seq and ptms for fingerprinting
# seq_ptms += [['X', '']] * (MAX_PEPTIDE_LENGTH - len(seq_ptms))  # Pad if needed
# seq_ptms = seq_ptms[:MAX_PEPTIDE_LENGTH]  # Truncate if needed

# # # Unzip the sequence and PTMs into separate lists
# seq, ptms = zip(*seq_ptms)

# charge = 0
# NCE = 0.25

# fp1 = ap.fingerprint(seq, ptms, charge, NCE)

# print(fp1.shape)

# fp_list = []
# for i in range(len(fp1)):
#     fp_list.append(fp1[i])
#     print(f"{i+1}th row of fp1:", fp1[i])

# for i, row in enumerate(fp1):
#     non_zero_cols = [j for j, val in enumerate(row) if val != 0]
#     print(f"{i+1}th row of fp1 has non-zero values at columns: ", non_zero_cols)


# similarity = ap.calculate_similarity(row_8_fp1, row_8_fp2)
# print("Similarity:", similarity)