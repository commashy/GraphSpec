import math
from pyteomics import mgf, mass
import argparse
import pandas as pd
import pickle
import random
import numpy as np

import torch
import torch.nn as nn

from Bio.SeqUtils.ProtParam import ProteinAnalysis

from AmorProt import AmorProtV2

encoding_dimension = 12
SPECTRA_DIMENSION = 20000
BIN_SIZE = 0.1
MAX_PEPTIDE_LENGTH = 50
MAX_MZ = 2000


def parse_spectra(sps):
    db = []

    for sp in sps:
        param = sp['params']

        c = int(str(param['charge'][0])[0])
        if 'seq' in param:
            pep = param['seq']
        else:
            pep = param['title']

        leave = False
        for i in range(len(pep)):
          if pep[i] not in charMap.keys():
            leave = True
        if leave == True:
#           print('have mod, skipped')
          continue

        if 'pepmass' in param:
            mass = param['pepmass'][0]
        else:
            mass = float(param['parent'])
        mz = sp['m/z array']
        it = sp['intensity array']

        db.append({'Sequence': pep, 'Charge': c,'Modified sequence': pep,'Modifications': '',
                   'Mass': mass, 'mz': mz, 'it': it, 'len': len(pep)})

    return db


def readmgf(fn):
    file = open(fn, "r")
    data = mgf.read(file, convert_arrays=1, read_charges=False,
                    dtype='float32', use_index=False)

    codes = parse_spectra(data)
    return codes
    
Alist = list('ACDEFGHIKLMNPQRSTVWY*')

charMap = {}
for i, a in enumerate(Alist):
    charMap[a] = i + 1


def find_mod(row):
    row = row.strip('_')
    seq = row#[1:-1]
    pos = 0
    poslist = []
    modlist = []
    ismod = False
    mod = ""
    for i in range(len(seq)):
       
        if seq[i] == ')':
            ismod = False
            pos -= 1
            poslist.append(pos)
            modlist.append(mod)
            mod = ""
        if ismod == True:
            mod += seq[i]
        else:
            pos += 1
#             print(pos)
        if seq[i] == '(':
            ismod = True
            pos -= 1
    return modlist, poslist
 
mono = {"G": 57.021464, "A": 71.037114, "S": 87.032029, "P": 97.052764, "V": 99.068414, "T": 101.04768,
        "C": 160.03019, "L": 113.08406, "I": 113.08406, "D": 115.02694, "Q": 128.05858, "K": 128.09496,
        "E": 129.04259, "M": 131.04048, "m": 147.0354, "H": 137.05891, "F": 147.06441, "R": 156.10111,
        "Y": 163.06333, "N": 114.04293, "W": 186.07931, "O": 147.03538}

ave_mass = {"A": 71.0788, "R": 156.1875, "N": 114.1038, "D": 115.0886, "C": 160.1598, "E": 129.1155,
            "Q": 128.1307, "G": 57.0519, "H": 137.1411, "I": 113.1594, "L": 113.1594, "K": 128.1741,
            "M": 131.1926, "F": 147.1766, "P": 97.1167, "S": 87.0782, "T": 101.1051,
            "W": 186.2132, "Y": 163.1760, "V": 99.1326}
# Amino acid : [C H N O S P]
atoms = { "A": [1,2,0,0,0,0], "R":[4,9,3,0,0,0], "N":[2,3,1,1,0,0], "D":[2,2,0,2,0,0], "C":[1,2,0,0,1,0], "Q":[3,5,1,1,0,0], "E":[3,4,0,2,0,0], "G":[0,0,0,0,0,0],
          "H": [4,4,2,0,0,0], "I":[4,8,0,0,0,0], "L":[4,8,0,0,0,0], "K":[4,9,1,0,0,0], "M":[3,6,0,0,1,0], "F":[7,6,0,0,0,0], "P":[3,4,0,0,0,0], "S":[1,2,0,1,0,0],
          "T": [2,4,0,1,0,0], "W":[9,7,1,0,0,0], "Y":[7,6,0,1,0,0], "V":[3,6,0,0,0,0] 
}

# C H N O S P
mods = {"ox" : [0,0,0,1,0,0], 
        "ph" : [0,1,0,3,0,1], 
        "cam" : [2,3,1,1,0,0] , "ac": [2,2,0,1,0,0], "me": [1,2,0,0,0,0], "hy": [0,0,0,1,0,0], "gly": [4,6,2,2,0,0],
        "bi" : [10,14,2,2,1,0], "cr": [4,4,0,1,0,0], "di": [2,4,0,0,0,0], "ma": [3,2,0,3,0,0], "ni": [0,-1,1,2,0,0],
        "bu" : [4,6,0,1,0,0], "fo": [1,0,0,1,0,0], "glu": [5,6,0,3,0,0], "hyb": [4,6,0,2,0,0], "pr": [3,4,0,1,0,0],
        "su" : [4,4,0,3,0,0], "tr": [3,6,0,0,0,0], "ci": [0,-1,-1,1,0,0]}
        
mass_weight = np.array([0.06,0.005,0.07,0.08,0.16,0.155])

physical_properties = {
    "R": {"Hydropathy": 1, "Charge": 1, "pKa_NH2": 9.09, "pKa_COOH": 2.18, "pK_R": 13.2, "Solubility": 71.8},
    "N": {"Hydropathy": 1, "Charge": 0, "pKa_NH2": 8.8, "pKa_COOH": 2.02, "pK_R": None, "Solubility": 2.4},
    "D": {"Hydropathy": 1, "Charge": -1, "pKa_NH2": 9.6, "pKa_COOH": 1.88, "pK_R": 3.65, "Solubility": 0.42},
    "E": {"Hydropathy": 1, "Charge": -1, "pKa_NH2": 9.67, "pKa_COOH": 2.19, "pK_R": 4.25, "Solubility": 0.72},
    "Q": {"Hydropathy": 1, "Charge": 0, "pKa_NH2": 9.13, "pKa_COOH": 2.17, "pK_R": None, "Solubility": 2.6},
    "K": {"Hydropathy": 1, "Charge": 1, "pKa_NH2": 8.9, "pKa_COOH": 2.2, "pK_R": 10.28, "Solubility": 1000},  # Assuming high solubility for 'freely'
    "S": {"Hydropathy": 1, "Charge": 0, "pKa_NH2": 9.15, "pKa_COOH": 2.21, "pK_R": None, "Solubility": 36.2},
    "T": {"Hydropathy": 1, "Charge": 0, "pKa_NH2": 9.12, "pKa_COOH": 2.15, "pK_R": None, "Solubility": 1000},  # Assuming high solubility for 'freely'
    "C": {"Hydropathy": 2, "Charge": 0, "pKa_NH2": 10.78, "pKa_COOH": 1.71, "pK_R": 8.33, "Solubility": 1000},  # Assuming high solubility for 'freely'
    "H": {"Hydropathy": 2, "Charge": 1, "pKa_NH2": 8.97, "pKa_COOH": 1.78, "pK_R": 6, "Solubility": 4.19},
    "M": {"Hydropathy": 2, "Charge": 0, "pKa_NH2": 9.21, "pKa_COOH": 2.28, "pK_R": None, "Solubility": 5.14},
    "A": {"Hydropathy": 0, "Charge": 0, "pKa_NH2": 9.87, "pKa_COOH": 2.35, "pK_R": None, "Solubility": 15.8},
    "V": {"Hydropathy": 0, "Charge": 0, "pKa_NH2": 9.72, "pKa_COOH": 2.29, "pK_R": None, "Solubility": 5.6},
    "G": {"Hydropathy": 0, "Charge": 0, "pKa_NH2": 9.6, "pKa_COOH": 2.34, "pK_R": None, "Solubility": 22.5},
    "I": {"Hydropathy": 0, "Charge": 0, "pKa_NH2": 9.76, "pKa_COOH": 2.32, "pK_R": None, "Solubility": 3.36},
    "L": {"Hydropathy": 0, "Charge": 0, "pKa_NH2": 9.6, "pKa_COOH": 2.36, "pK_R": None, "Solubility": 2.37},
    "F": {"Hydropathy": 0, "Charge": 0, "pKa_NH2": 9.24, "pKa_COOH": 2.58, "pK_R": None, "Solubility": 2.7},
    "P": {"Hydropathy": 0, "Charge": 0, "pKa_NH2": 10.6, "pKa_COOH": 1.99, "pK_R": None, "Solubility": 1.54},
    "W": {"Hydropathy": 0, "Charge": 0, "pKa_NH2": 9.39, "pKa_COOH": 2.38, "pK_R": None, "Solubility": 1.06},
    "Y": {"Hydropathy": 0, "Charge": 0, "pKa_NH2": 9.11, "pKa_COOH": 2.2, "pK_R": 10.1, "Solubility": 0.038},
}

# coding: utf-8
import re
import csv
import numpy as np
import pandas as pd
from pyteomics import mass
from Bio.SeqUtils.ProtParam import ProteinAnalysis

"""
Parent sequence
* m/z x 1: user mass from pyteomics libray

code example:
mass.calculate_mass(sequence = peptide , ion_type = 'M', charge = int(peptideLi[1]))

* composition x 20: use ProteinAnalysis from Bio.SeqUtils.ProtParam

code example:
analysed_seq = ProteinAnalysis(peptide)
featureLi.extend(list(analysed_seq.count_amino_acids().values()))

* chemical property x 7 (total: 9): use method chemical_pro from ProteinAnalysis, pephelicity, pephydrophobicity and pepbasicity
* isoelectric point 
* instability index
* aromaticity
* helicity
* hydrophobicity
* basicity
* secondary_structure_fraction ([Helix, Turn, Sheet])

Fragment ion
* m/z x 1 
* chemical property x 7

"""

# helicity
def pephelicity (peptide) :
    helicity = 0
    for word in peptide:
        if ( word == 'A'):
            helicity += 1.24; 
        elif ( word == 'B'):
            helicity += 0.92; 
        elif ( word == 'C'):
            helicity += 0.79; 
        elif ( word == 'D'):
            helicity += 0.89; 
        elif ( word == 'E'):
            helicity += 0.85; 
        elif ( word == 'F'):
            helicity += 1.26; 
        elif ( word == 'G'):
            helicity += 1.15; 
        elif ( word == 'H'):
            helicity += 0.97; 
        elif ( word == 'I'):
            helicity += 1.29; 
        elif ( word == 'K'):
            helicity += 0.88; 
        elif ( word == 'L'):
            helicity += 1.28; 
        elif ( word == 'M'):
            helicity += 1.22; 
        elif ( word == 'N'):
            helicity += 0.94; 
        elif ( word == 'P'):
            helicity += 0.57; 
        elif ( word == 'Q'):
            helicity += 0.96; 
        elif ( word == 'R'):
            helicity += 0.95; 
        elif ( word == 'S'):
            helicity += 1; 
        elif ( word == 'T'):
            helicity += 1.09; 
        elif ( word == 'V'):
            helicity += 1.27; 
        elif ( word == 'W'):
            helicity += 1.07; 
        elif ( word == 'X'):
            helicity += 1.29; 
        elif ( word == 'Y'):
            helicity += 1.11; 
        elif ( word == 'Z'):
            helicity += 0.91; 
    helicity = helicity / len(peptide)
    return helicity

# hydrophobicity
def pephydrophobicity(peptide) :
    hydrophobicity = 0
    for word in peptide:
        if ( word == 'A'):
            hydrophobicity += 0.16; 
        elif ( word == 'B'):
            hydrophobicity += -3.14; 
        elif ( word == 'C'):
            hydrophobicity += 2.5; 
        elif ( word == 'D'):
            hydrophobicity += -2.49; 
        elif ( word == 'E'):
            hydrophobicity += -1.5; 
        elif ( word == 'F'):
            hydrophobicity += 5; 
        elif ( word == 'G'):
            hydrophobicity += -3.31; 
        elif ( word == 'H'):
            hydrophobicity += -4.63; 
        elif ( word == 'I'):
            hydrophobicity += 4.41; 
        elif ( word == 'K'):
            hydrophobicity += -5; 
        elif ( word == 'L'):
            hydrophobicity += 4.76; 
        elif ( word == 'M'):
            hydrophobicity += 3.23; 
        elif ( word == 'N'):
            hydrophobicity += -3.79; 
        elif ( word == 'P'):
            hydrophobicity += -4.92; 
        elif ( word == 'Q'):
            hydrophobicity += -2.76; 
        elif ( word == 'R'):
            hydrophobicity += -2.77; 
        elif ( word == 'S'):
            hydrophobicity += -2.85; 
        elif ( word == 'T'):
            hydrophobicity += -1.08; 
        elif ( word == 'V'):
            hydrophobicity += 3.02; 
        elif ( word == 'W'):
            hydrophobicity += 4.88; 
        elif ( word == 'X'):
            hydrophobicity += 4.59; 
        elif ( word == 'Y'):
            hydrophobicity += 2; 
        elif ( word == 'Z'):
            hydrophobicity += -2.13; 
        hydrophobicity = hydrophobicity / len(peptide)
    return hydrophobicity

# basicity
def pepbasicity (peptide) :
    basicity = 0
    for word in peptide:
        if ( word == 'A'):
            basicity += 206.4; 
        elif ( word == 'B'):
            basicity += 210.7; 
        elif ( word == 'C'):
            basicity += 206.2; 
        elif ( word == 'D'):
            basicity += 208.6; 
        elif ( word == 'E'):
            basicity += 215.6; 
        elif ( word == 'F'):
            basicity += 212.1; 
        elif ( word == 'G'):
            basicity += 202.7; 
        elif ( word == 'H'):
            basicity += 223.7; 
        elif ( word == 'I'):
            basicity += 210.8; 
        elif ( word == 'K'):
            basicity += 221.8; 
        elif ( word == 'L'):
            basicity += 209.6; 
        elif ( word == 'M'):
            basicity += 213.3; 
        elif ( word == 'N'):
            basicity += 212.8; 
        elif ( word == 'P'):
            basicity += 214.4; 
        elif ( word == 'Q'):
            basicity += 214.2; 
        elif ( word == 'R'):
            basicity += 237; 
        elif ( word == 'S'):
            basicity += 207.6; 
        elif ( word == 'T'):
            basicity += 211.7; 
        elif ( word == 'V'):
            basicity += 208.7; 
        elif ( word == 'W'):
            basicity += 216.1; 
        elif ( word == 'X'):
            basicity += 210.2; 
        elif ( word == 'Y'):
            basicity += 213.1; 
        elif ( word == 'Z'):
            basicity += 214.9; 
    basicity = basicity / len(peptide)
    return basicity

# chemical property into list
def chemical_pro(Li,sequence) :
    analysed_seq = ProteinAnalysis(sequence)
    #Li.extend(list(analysed_seq.count_amino_acids().values()))
    #Li.extend(list(analysed_seq.get_amino_acids_percent().values()))
    Li.append(analysed_seq.isoelectric_point())
    Li.append(analysed_seq.instability_index())
    Li.append(analysed_seq.aromaticity())
    #Li.append(analysed_seq.secondary_structure_fraction())
    for x in analysed_seq.secondary_structure_fraction() :
        Li.append(x)
    Li.append(pephydrophobicity(sequence))
    Li.append(pephelicity(sequence))
    Li.append(pepbasicity(sequence))
    
def chemical_pro_cle(Li,sequence) :
    analysed_seq = ProteinAnalysis(sequence)
    Li.extend(list(analysed_seq.count_amino_acids().values()))
    #Li.extend(list(analysed_seq.get_amino_acids_percent().values()))
    Li.append(analysed_seq.isoelectric_point())
    Li.append(analysed_seq.instability_index())
    Li.append(analysed_seq.aromaticity())
    Li.append(pephydrophobicity(sequence))
    Li.append(pephelicity(sequence))
    Li.append(pepbasicity(sequence))


def asnp(x): return np.asarray(x)
def asnp32(x): return np.asarray(x, dtype='float16')

def spectrum2vector(mz_list, it_list, bin_size, charge):
    
    it_list = it_list / np.max(it_list)

    vector = np.zeros(SPECTRA_DIMENSION, dtype='float32')

    mz_list = np.asarray(mz_list)

    indexes = np.floor(mz_list / bin_size)
    indexes = np.around(indexes).astype('int32')
    indexes = np.clip(indexes,0,SPECTRA_DIMENSION-1)

    for i, index in enumerate(indexes):
        vector[index] += it_list[i]


    return vector

def embed(sp, charge, mass_scale=MAX_MZ):
    encoding = np.zeros((MAX_PEPTIDE_LENGTH + 2, encoding_dimension), dtype='float32')

    pep = sp
    for i in range(len(pep)):
        if i >= MAX_PEPTIDE_LENGTH:
          break
        encoding[i][:6] = atoms[pep[i]] * mass_weight

    encoding[-1][charge] = 1

    return encoding    

def embed_maxquant(sp, mass_scale=MAX_MZ, augment=True, fixedaugment = False, key=None, havelong=False):
    encoding = np.zeros((MAX_PEPTIDE_LENGTH + 2, encoding_dimension), dtype='float16')

    pep = sp['Sequence']
    charge = int(sp['Charge'])
    if augment:
        pos_to_mod = np.random.randint(0, len(pep))
        roll = np.random.uniform(0,1)
    for i in range(len(pep)):
        if fixedaugment and pep[i] == key:
          encoding[i][6:12] = atoms[pep[i]] * mass_weight
          continue
        if i >= MAX_PEPTIDE_LENGTH:
          encoding[-2][:6] += atoms[pep[i]] * mass_weight
          continue
        if augment and roll <= 0.1:
          if i == pos_to_mod:
            encoding[i][6:12] = atoms[pep[i]] * mass_weight
            continue
        encoding[i][:6] = atoms[pep[i]] * mass_weight
    encoding[-1][charge] = 1    
    encoding[-1][-1] = sp['NCE'] / 100 if 'NCE' in sp else 0.25    

    #add modification
    modlist, poslist = find_mod(sp['Modified sequence'])
    for i in range(len(poslist)):
      if modlist[i] == "gl":
        modstring = sp['Modifications'].split(',')
        if "Glutaryl [K]" in modstring:
          modlist[i] = "glu"
        else:
          modlist[i] = "gly"
      if modlist[i] == "hy":
        modstring = sp['Modifications'].split(',')
        if "Hydroxyisobutyryl [K]" in modstring:
          modlist[i] = "hyb"
        else:
          modlist[i] = "ox"
      #encoding[poslist[i]][:6] += mods[modlist[i]] * mass_weight # prevent double count if modified glycine
      encoding[poslist[i]][6:12] += mods[modlist[i]] * mass_weight
    return encoding

def embed_pdeep(sp, mass_scale=MAX_MZ):
    encoding = np.zeros((MAX_PEPTIDE_LENGTH + 2, encoding_dimension), dtype='float16')

    pep = sp['peptide']
    charge = int(sp['charge'])
    for i in range(len(pep)):
        if i >= MAX_PEPTIDE_LENGTH:
          break
        encoding[i][:6] = atoms[pep[i]] * mass_weight
    encoding[-1][charge] = 1       

    #add modification
    allmods = sp['modification'].split(';')
    for i in allmods:
        modstring = i.split(',')[1][:2]
        modstring = modstring.lower()
        pos = int(i.split(',')[0])
        if modstring == "gl":
          modstring = i.split(',')[1][:2].lower()
        if modstring == "hy":
          modstring = i.split(',')[1][:2].lower()
        encoding[pos][:6] += mods[modstring] * mass_weight
        encoding[pos][6:12] += mods[modstring] * mass_weight
    return encoding

def embed_maxquant_with_physical_properties(sp, mass_scale=MAX_MZ, augment=True, fixedaugment=False, key=None, havelong=False):
    encoding = np.zeros((MAX_PEPTIDE_LENGTH + 2, encoding_dimension), dtype='float16')

    pep = sp['Sequence']
    charge = int(sp['Charge'])
    if augment:
        pos_to_mod = np.random.randint(0, len(pep))
        roll = np.random.uniform(0,1)
    
    for i in range(len(pep)):
        if fixedaugment and pep[i] == key:
          encoding[i][6:12] = atoms[pep[i]] * mass_weight
          continue
        if i >= MAX_PEPTIDE_LENGTH:
          encoding[-2][:6] += atoms[pep[i]] * mass_weight
          continue
        if augment and roll <= 0.1:
          if i == pos_to_mod:
            encoding[i][6:12] = atoms[pep[i]] * mass_weight
            continue
        encoding[i][:6] = atoms[pep[i]] * mass_weight
        
        # Adding physical properties
        amino_acid_sequence = pep[i]  # Single amino acid sequence for feature calculation
        temp_features = []
        chemical_pro(temp_features, amino_acid_sequence)  # This function will append the new features to temp_features

        # Add the new features to the encoding array
        encoding[i][12:12+len(temp_features)] = temp_features  # Adjust slice [18:25] based on the actual indices you want to use


    encoding[-1][charge] = 1    
    encoding[-1][-1] = sp['NCE'] / 100 if 'NCE' in sp else 0.25    

    # Mod
    modlist, poslist = find_mod(sp['Modified sequence'])
    for i in range(len(poslist)):
      if modlist[i] == "gl":
        modstring = sp['Modifications'].split(',')
        if "Glutaryl [K]" in modstring:
          modlist[i] = "glu"
        else:
          modlist[i] = "gly"
      if modlist[i] == "hy":
        modstring = sp['Modifications'].split(',')
        if "Hydroxyisobutyryl [K]" in modstring:
          modlist[i] = "hyb"
        else:
          modlist[i] = "ox"
      #encoding[poslist[i]][:6] += mods[modlist[i]] * mass_weight # prevent double count if modified glycine
      encoding[poslist[i]][6:12] += mods[modlist[i]] * mass_weight
    
    return encoding

def embed_maxquant_with_global_features(sp, mass_scale=MAX_MZ, augment=True, fixedaugment=False, key=None, havelong=False):
    # Assume global_features_count is the number of global features calculated by chemical_pro
    global_features_count = 9  # For example, adjust based on actual count from chemical_pro
    local_features_count = 12  # The count for local features (atoms * mass_weight)
    total_features_count = local_features_count + global_features_count  # Total features per position

    # Update encoding_dimension to include both local and global features for each position
    encoding_dimension = total_features_count
    encoding = np.zeros((MAX_PEPTIDE_LENGTH + 2, encoding_dimension), dtype='float16')

    pep = sp['Sequence']
    charge = int(sp['Charge'])

    # Calculate global physicochemical features for the entire sequence
    global_features = []
    chemical_pro(global_features, pep)  # Assuming chemical_pro appends features to global_features list

    if augment:
        pos_to_mod = np.random.randint(0, len(pep))
        roll = np.random.uniform(0,1)
    
    for i in range(len(pep)):
        if fixedaugment and pep[i] == key:
            encoding[i][6:12] = atoms[pep[i]] * mass_weight
            continue
        if i >= MAX_PEPTIDE_LENGTH:
            encoding[-2][:6] += atoms[pep[i]] * mass_weight
            continue
        if augment and roll <= 0.1:
            if i == pos_to_mod:
                encoding[i][6:12] = atoms[pep[i]] * mass_weight
                continue
        encoding[i][:6] = atoms[pep[i]] * mass_weight
        
        # Repeat global features for each residue position
        encoding[i][local_features_count:total_features_count] = global_features

    encoding[-1][charge] = 1    
    encoding[-1][-1] = sp['NCE'] / 100 if 'NCE' in sp else 0.25    

    # Mod
    modlist, poslist = find_mod(sp['Modified sequence'])
    for i in range(len(poslist)):
      if modlist[i] == "gl":
        modstring = sp['Modifications'].split(',')
        if "Glutaryl [K]" in modstring:
          modlist[i] = "glu"
        else:
          modlist[i] = "gly"
      if modlist[i] == "hy":
        modstring = sp['Modifications'].split(',')
        if "Hydroxyisobutyryl [K]" in modstring:
          modlist[i] = "hyb"
        else:
          modlist[i] = "ox"
      #encoding[poslist[i]][:6] += mods[modlist[i]] * mass_weight # prevent double count if modified glycine
      encoding[poslist[i]][6:12] += mods[modlist[i]] * mass_weight
    
    return encoding
    
def embed_amorprot(sp, ap):
    pep = sp['Sequence']
    charge = int(sp['Charge']) if 'Charge' in sp else 0
    NCE = sp['NCE'] / 100 if 'NCE' in sp else 0.25

    # Truncate or pad the sequence to MAX_PEPTIDE_LENGTH
    if len(pep) > MAX_PEPTIDE_LENGTH:
        pep = pep[:MAX_PEPTIDE_LENGTH]  # Truncate the sequence
    elif len(pep) < MAX_PEPTIDE_LENGTH:
        pep = pep.ljust(MAX_PEPTIDE_LENGTH, 'X')  # Pad the sequence with 'X' (or another padding character)

    # Generate the fingerprint using AmorProt
    fp = ap.fingerprint(pep, charge, NCE)

    # Add the charge and NCE to the fingerprint
    # fp_extended = np.append(fp, [charge, NCE])

    return fp

def find_mod2(sequence):
    # Initialize the 2D list with empty sublists for each amino acid
    result = [["" for _ in range(2)] for _ in sequence]
    
    # Remove leading and trailing underscores (or any other preprocessing)
    seq = sequence.strip('_')
    
    i = 0
    pos = 0  # Position in the result list
    while i < len(seq):
        if seq[i] == '(':
            # Find the end of the PTM
            end = seq.find(')', i) + 1
            ptm = seq[i:end]  # Extract the PTM, including parentheses
            # Append the PTM to the corresponding amino acid in the result
            result[pos-1][1] = ptm
            i = end
        else:
            # Add the amino acid to the first column of the current row in the result
            result[pos][0] = seq[i]
            pos += 1
            i += 1

    # Trim the list to the actual sequence length (remove empty entries if any)
    result = [r for r in result if r[0] != ""]
    
    return result

def embed_amorprot2(sp, ap):
    modified_seq = sp['Modified sequence']
    charge = int(sp['Charge']) if 'Charge' in sp else 0
    NCE = sp['NCE'] / 100 if 'NCE' in sp else 0.25

    # Extract sequence and PTMs
    seq_ptms = find_mod2(modified_seq)

    # Process each modification
    for i, (aa, mod) in enumerate(seq_ptms):
        if mod == '(gl)':
            modstring = sp['Modifications'].split(',')
            if "Glutaryl [K]" in modstring:
                seq_ptms[i][1] = '(glu)'  # Change the modification to 'glu'
            else:
                seq_ptms[i][1] = '(gly)'  # Change the modification to 'gly'

    # Adjust the sequence length to MAX_PEPTIDE_LENGTH and prepare seq and ptms for fingerprinting
    seq_ptms += [['X', '']] * (MAX_PEPTIDE_LENGTH - len(seq_ptms))  # Pad if needed
    seq_ptms = seq_ptms[:MAX_PEPTIDE_LENGTH]  # Truncate if needed
    
    # Unzip the sequence and PTMs into separate lists
    seq, ptms = zip(*seq_ptms)

    # Generate the fingerprint using AmorProt, passing both sequence and PTMs
    fp = ap.fingerprint(seq, ptms, charge, NCE)

    return fp

def combined_embedding_function(sp, ap, mass_scale=MAX_MZ, augment=True, fixedaugment=False, key='N', havelong=True):
    encoding = np.zeros((MAX_PEPTIDE_LENGTH + 2, 12), dtype='float16')

    # First, generate embeddings using embed_maxquant
    pep = sp['Sequence']
    charge = int(sp['Charge'])
    if augment:
        pos_to_mod = np.random.randint(0, len(pep))
        roll = np.random.uniform(0,1)
    for i in range(len(pep)):
        if fixedaugment and pep[i] == key:
          encoding[i][6:12] = atoms[pep[i]] * mass_weight
          continue
        if i >= MAX_PEPTIDE_LENGTH:
          encoding[-2][:6] += atoms[pep[i]] * mass_weight
          continue
        if augment and roll <= 0.1:
          if i == pos_to_mod:
            encoding[i][6:12] = atoms[pep[i]] * mass_weight
            continue
        encoding[i][:6] = atoms[pep[i]] * mass_weight
    encoding[-1][charge] = 1    
    encoding[-1][-1] = sp['NCE'] / 100 if 'NCE' in sp else 0.25    

    #add modification
    modlist, poslist = find_mod(sp['Modified sequence'])
    for i in range(len(poslist)):
      if modlist[i] == "gl":
        modstring = sp['Modifications'].split(',')
        if "Glutaryl [K]" in modstring:
          modlist[i] = "glu"
        else:
          modlist[i] = "gly"
      if modlist[i] == "hy":
        modstring = sp['Modifications'].split(',')
        if "Hydroxyisobutyryl [K]" in modstring:
          modlist[i] = "hyb"
        else:
          modlist[i] = "ox"
      #encoding[poslist[i]][:6] += mods[modlist[i]] * mass_weight # prevent double count if modified glycine
      encoding[poslist[i]][6:12] += mods[modlist[i]] * mass_weight
    
    modified_seq = sp['Modified sequence']
    # Extract sequence and PTMs
    seq_ptms = find_mod2(modified_seq)

    # Process each modification
    for i, (aa, mod) in enumerate(seq_ptms):
        if mod == '(gl)':
            modstring = sp['Modifications'].split(',')
            if "Glutaryl [K]" in modstring:
                seq_ptms[i][1] = '(glu)'  # Change the modification to 'glu'
            else:
                seq_ptms[i][1] = '(gly)'  # Change the modification to 'gly'

    # Adjust the sequence length to MAX_PEPTIDE_LENGTH and prepare seq and ptms for fingerprinting
    seq_ptms += [['X', '']] * ((MAX_PEPTIDE_LENGTH+2) - len(seq_ptms))  # Pad if needed
    seq_ptms = seq_ptms[:MAX_PEPTIDE_LENGTH+2]  # Truncate if needed
    
    # Unzip the sequence and PTMs into separate lists
    seq, ptms = zip(*seq_ptms)

    # Generate the fingerprint using AmorProt, passing both sequence and PTMs
    fp = ap.fingerprint2(seq, ptms)

    combined_embedding = np.concatenate((fp, encoding), axis=1)
    
    return combined_embedding


def embed_graph(sp, ap):
    modified_seq = sp['Modified sequence'][0]
    charge = int(sp['Charge']) if 'Charge' in sp else 0
    NCE = sp['NCE'] / 100 if 'NCE' in sp else 0.25

    # Extract sequence and PTMs
    seq_ptms = find_mod2(modified_seq)

    # Process each modification
    for i, (aa, mod) in enumerate(seq_ptms):
        if mod == '(gl)':
            modstring = sp['Modifications'][0].split(',')
            if "Glutaryl [K]" in modstring:
                seq_ptms[i][1] = '(glu)'  # Change the modification to 'glu'
            else:
                seq_ptms[i][1] = '(gly)'  # Change the modification to 'gly'

    # Adjust the sequence length to MAX_PEPTIDE_LENGTH and prepare seq and ptms for fingerprinting
    seq_ptms += [['X', '']] * (MAX_PEPTIDE_LENGTH - len(seq_ptms))  # Pad if needed
    if len(seq_ptms) > MAX_PEPTIDE_LENGTH:
        seq_ptms = seq_ptms[:MAX_PEPTIDE_LENGTH]
    
    # Unzip the sequence and PTMs into separate lists
    seq, ptms = zip(*seq_ptms)
    pepgraph = ap.graph_gen(seq, ptms)

    return pepgraph, charge, NCE

def prep_graph(sp):
    modified_seq = sp['Modified sequence']
    charge = int(sp['Charge']) if 'Charge' in sp else 0
    NCE = sp['NCE'] / 100 if 'NCE' in sp else 0.25

    # Extract sequence and PTMs
    seq_ptms = find_mod2(modified_seq)
    length = len(seq_ptms)

    # Process each modification
    for i, (aa, mod) in enumerate(seq_ptms):
        if mod == '(gl)':
            modstring = sp['Modifications'][0].split(',')
            if "Glutaryl [K]" in modstring:
                seq_ptms[i][1] = '(glu)'  # Change the modification to 'glu'
            else:
                seq_ptms[i][1] = '(gly)'  # Change the modification to 'gly'

    # Adjust the sequence length to MAX_PEPTIDE_LENGTH and prepare seq and ptms for fingerprinting
    # seq_ptms += [['X', '']] * (MAX_PEPTIDE_LENGTH - len(seq_ptms))  # Pad if needed

    # if len(seq_ptms) > MAX_PEPTIDE_LENGTH:
    #     seq_ptms = seq_ptms[:MAX_PEPTIDE_LENGTH]

    return seq_ptms, charge, NCE, length

def prep_graph_test(sp):
    modified_seq = sp['Modified sequence']
    charge = int(sp['Charge']) if 'Charge' in sp else 0
    NCE = sp['NCE'] / 100 if 'NCE' in sp else 0.25

    # Extract sequence and PTMs
    seq_ptms = find_mod2(modified_seq)
    length = len(seq_ptms)

    # Process each modification
    for i, (aa, mod) in enumerate(seq_ptms):
        if mod == '(gl)':
            modstring = sp['Modifications'][0].split(',')
            if "Glutaryl [K]" in modstring:
                seq_ptms[i][1] = '(glu)'  # Change the modification to 'glu'
            else:
                seq_ptms[i][1] = '(gly)'  # Change the modification to 'gly'

    # Adjust the sequence length to MAX_PEPTIDE_LENGTH and prepare seq and ptms for fingerprinting
    # seq_ptms += [['X', '']] * (MAX_PEPTIDE_LENGTH - len(seq_ptms))  # Pad if needed

    # if len(seq_ptms) > MAX_PEPTIDE_LENGTH:
    #     seq_ptms = seq_ptms[:MAX_PEPTIDE_LENGTH]

    return seq_ptms, charge, NCE, length, modified_seq

def embed_maxquant_test(sp, mass_scale=MAX_MZ, augment=True, fixedaugment = False, key=None, havelong=False):
    encoding = np.zeros((MAX_PEPTIDE_LENGTH + 2, encoding_dimension), dtype='float16')

    pep = sp['Sequence']
    charge = int(sp['Charge'])
    modified_seq = sp['Modified sequence']
    if augment:
        pos_to_mod = np.random.randint(0, len(pep))
        roll = np.random.uniform(0,1)
    for i in range(len(pep)):
        if fixedaugment and pep[i] == key:
          encoding[i][6:12] = atoms[pep[i]] * mass_weight
          continue
        if i >= MAX_PEPTIDE_LENGTH:
          encoding[-2][:6] += atoms[pep[i]] * mass_weight
          continue
        if augment and roll <= 0.1:
          if i == pos_to_mod:
            encoding[i][6:12] = atoms[pep[i]] * mass_weight
            continue
        encoding[i][:6] = atoms[pep[i]] * mass_weight
    encoding[-1][charge] = 1    
    encoding[-1][-1] = sp['NCE'] / 100 if 'NCE' in sp else 0.25    

    #add modification
    modlist, poslist = find_mod(sp['Modified sequence'])
    for i in range(len(poslist)):
      if modlist[i] == "gl":
        modstring = sp['Modifications'].split(',')
        if "Glutaryl [K]" in modstring:
          modlist[i] = "glu"
        else:
          modlist[i] = "gly"
      if modlist[i] == "hy":
        modstring = sp['Modifications'].split(',')
        if "Hydroxyisobutyryl [K]" in modstring:
          modlist[i] = "hyb"
        else:
          modlist[i] = "ox"
      #encoding[poslist[i]][:6] += mods[modlist[i]] * mass_weight # prevent double count if modified glycine
      encoding[poslist[i]][6:12] += mods[modlist[i]] * mass_weight
    return encoding, modified_seq

def flatten_seq_ptms(list_of_lists):
    """Flatten a list of lists of [amino_acid, modification] pairs into a single list."""
    # This will flatten a list of sequences into one large sequence
    return [pair for sublist in list_of_lists for pair in sublist]


def embed_graph2(seq_ptms, ap):
    # Flatten seq_ptms into one large list of [amino_acid, modification] pairs
    flattened_seq_ptms = flatten_seq_ptms(seq_ptms)

    # Now, you can unzip the sequence and PTMs into separate lists
    seq, ptms = zip(*flattened_seq_ptms)

    # Generate the peptide graph
    pepgraph = ap.graph_gen(seq, ptms)

    return pepgraph

# def embed_amorprot2(sp, ap):
#     modified_seq = sp['Modified sequence']
#     charge = int(sp['Charge']) if 'Charge' in sp else 0
#     NCE = sp['NCE'] / 100 if 'NCE' in sp else 0.25

#     # Use find_mod_with_2d_output to get sequence and PTMs
#     seq, ptms = find_mod_with_2d_output(modified_seq)
    
#     # Adjust the sequence length to MAX_PEPTIDE_LENGTH
#     if len(seq) > MAX_PEPTIDE_LENGTH:
#         seq = seq[:MAX_PEPTIDE_LENGTH]  # Truncate the sequence
#         ptms = ptms[:MAX_PEPTIDE_LENGTH]  # Truncate the PTMs accordingly
#     elif len(seq) < MAX_PEPTIDE_LENGTH:
#         seq = seq.ljust(MAX_PEPTIDE_LENGTH, 'X')  # Pad the sequence
#         ptms = ptms.ljust(MAX_PEPTIDE_LENGTH, '_')  # Pad the PTMs with placeholders
    
#     # Generate the fingerprint using AmorProt, passing both sequence and PTMs
#     fp = ap.fingerprint(seq, ptms, charge, NCE)

#     return fp
import random

def partition_data_by_specific_ptm(data, ptm_description, inclusion_percentage):
    """
    Partition the data based on a specific PTM described in the 'Modifications' field,
    including only a specified percentage of sequences with that PTM in the training set.

    Parameters:
    - data: The dataset (list of dictionaries).
    - ptm_description: Description of the PTM to filter by (e.g., 'Butyryl (K)').
    - inclusion_percentage: Percentage of sequences with the specified PTM to include in the training set.

    Returns:
    - train_set: The training set with the specified percentage of PTM sequences included.
    - validation_set: The validation set with the excluded PTM sequences.
    """
    
    # Filter sequences by the specified PTM
    ptm_sequences = [seq for seq in data if ptm_description in seq.get('Modifications', '')]
    non_ptm_sequences = [seq for seq in data if ptm_description not in seq.get('Modifications', '')]

    # Shuffle and split the PTM sequences based on the specified inclusion percentage
    random.Random(777).shuffle(ptm_sequences)
    split_index = int(len(ptm_sequences) * inclusion_percentage / 100)
    ptm_train = ptm_sequences[:split_index]
    ptm_validation = ptm_sequences[split_index:]

    # Combine the PTM training sequences with the non-PTM sequences for the training set
    train_set = non_ptm_sequences + ptm_train
    # train_set = ptm_train

    validation_set = ptm_validation  # The validation set consists of the excluded PTM sequences

    return train_set, validation_set


def spectrum2vector_maxquant(mz_list, it_list, bin_size, charge):
    it_list = it_list.strip('[]\n\t')
    mz_list = mz_list.strip('[]\n\t')
    it_list =  [float(idx) for idx in it_list.split(';')]
    mz_list = [float(idx) for idx in mz_list.split(';')]
    it_list = it_list / np.max(it_list)

    vector = np.zeros(SPECTRA_DIMENSION, dtype='float16')

    mz_list = np.asarray(mz_list)
    mz_list[mz_list>1999.8] = 1999.8

    indexes = np.floor(mz_list / bin_size)
    indexes = np.around(indexes).astype('int16')

    for i, index in enumerate(indexes):
        vector[index] += it_list[i]

    # if normalize
    #vector = np.sqrt(vector)

    return vector

def spectral_angle(true, pred):
    # cos_sim =  tf.keras.losses.CosineSimilarity(axis=1)
    cos_sim = torch.nn.CosineSimilarity(dim=1)
    
    product = cos_sim(true,pred)
    # arccos = tf.math.acos(product)
    arccos = torch.acos(product)
    return 2 * arccos / np.pi


# def masked_spectral_distance(pred, true):
#     epsilon = 1e-07
#     pred_masked = ((true + 1) * pred) / (true + 1 + epsilon)
#     true_masked = ((true + 1) * true) / (true + 1 + epsilon)
#     pred_norm = torch.nn.functional.normalize(pred_masked, p=2, dim=-1)
#     true_norm = torch.nn.functional.normalize(true_masked, p=2, dim=-1)
#     product = torch.sum(pred_norm * true_norm, dim=-1)
#     # product_clamped = torch.clamp(product, -1.0, 1.0)
#     # arccos = torch.acos(product_clamped)
#     arccos = torch.acos(product)
#     loss = 2 * arccos / np.pi
#     return torch.mean(loss)  # or torch.sum(loss) if you prefer

def masked_spectral_distance(true, pred):
    epsilon = torch.finfo(torch.float32).eps
    pred_masked = ((true + 1) * pred) / (true + 1 + epsilon)
    true_masked = ((true + 1) * true) / (true + 1 + epsilon)
    pred_norm = torch.nn.functional.normalize(pred_masked, dim=-1)
    true_norm = torch.nn.functional.normalize(true_masked, dim=-1)
    product = torch.sum(pred_norm * true_norm, dim=1)
    product_clamped = torch.clamp(product, -1.0, 1.0)  # Clamp to avoid out-of-range for acos
    arccos = torch.acos(product_clamped)
    loss = 2 * arccos / np.pi
    return loss.mean()
    

def write_msp(out, sps, peps):

    def f2(x): return "{0:.2f}".format(x)

    def f4(x): return "{0:.4f}".format(x)

    def sparse(x, y, th=0.02): #0.02
        x = np.asarray(x, dtype='float32')
        y = np.asarray(y, dtype='float32')
        y /= np.max(y)
        return x[y > th], y[y > th]

    def write_one(f, sp, pep):
        precision = 0.1
        low = 0
        dim = 20000
        #pep['Mass'] = 1000
        #sp[min(math.ceil(float(pep['Mass']) * int(pep['Charge']) / precision), len(sp)):] = 0
        imz = np.arange(0, dim, dtype='int32') * precision + low  # more acurate
        mzs, its = sparse(imz, sp)
        
        seq = pep['Sequence']
        charge = pep['Charge']
        mass = pep['Mass']
        protein = pep['Protein']
        modlist, poslist = find_mod(pep['Modified sequence'])
        for i in range(len(poslist)):
            for mod in pep['Modification'].split(','):
                if mod[:2].lower() == modlist[i]:
                    modlist[i] = mod.split(' ')[0]
                    break

        modstr = str(len(poslist)) + ''.join([f'({p},{seq[p]},{m})' for m, p in zip(modlist, poslist)]) if poslist != [] else '0'
        head = (f"Name: {seq}/{charge}_{modstr}\n"
                f"Comment: Charge={charge} Parent={mass/charge} Mods={modstr} Protein={protein}\n"
                f"Num peaks: {len(mzs)}\n")
        peaks = [f"{f2(mz)}\t{f4(it * 1000)}" for mz, it in zip(mzs, its)]

        f.write(head + '\n'.join(peaks) + '\n\n')

    for i in range(len(peps)):
      write_one(out, sps[i], peps.iloc[i])