#!/usr/bin/env python
# coding: utf-8

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys

import torch
import torch_geometric

from torch_geometric.utils import smiles as smiles_utils
from torch_geometric.data import Data
from torch_geometric.data import Batch

MAX_PEPTIDE_LENGTH = 50

class AmorProtV2:
    def __init__(self, maccs=True, ecfp4=True, ecfp6=True, rdkit=True, graph=True, A=10, W=10, R=0.85):
        self.AA_dict = {
            'G': 'NCC(=O)O', 'A': 'N[C@@]([H])(C)C(=O)O', 
            'R': 'N[C@@]([H])(CCCNC(=N)N)C(=O)O', 'N': 'N[C@@]([H])(CC(=O)N)C(=O)O', 
            'D': 'N[C@@]([H])(CC(=O)O)C(=O)O', 'C': 'N[C@@]([H])(CS)C(=O)O', 
            'E': 'N[C@@]([H])(CCC(=O)O)C(=O)O', 'Q': 'N[C@@]([H])(CCC(=O)N)C(=O)O', 
            'H': 'N[C@@]([H])(CC1=CN=C-N1)C(=O)O', 'I': 'N[C@@]([H])(C(CC)C)C(=O)O', 
            'L': 'N[C@@]([H])(CC(C)C)C(=O)O', 'K': 'N[C@@]([H])(CCCCN)C(=O)O', 
            'M': 'N[C@@]([H])(CCSC)C(=O)O', 'F': 'N[C@@]([H])(Cc1ccccc1)C(=O)O', 
            'P': 'N1[C@@]([H])(CCC1)C(=O)O', 'S': 'N[C@@]([H])(CO)C(=O)O', 
            'T': 'N[C@@]([H])(C(O)C)C(=O)O', 'W': 'N[C@@]([H])(CC(=CN2)C1=C2C=CC=C1)C(=O)O', 
            'Y': 'N[C@@]([H])(Cc1ccc(O)cc1)C(=O)O', 'V': 'N[C@@]([H])(C(C)C)C(=O)O',
            'K(ac)': 'CC(=O)NCCCC[C@@H](C(=O)O)N', 'K(bi)': 'C1[C@H]2[C@@H]([C@@H](S1)CCCCC(=O)N[C@@H](CCCCN)C(=O)O)NC(=O)N2',
            'K(bu)': 'CCCC(=O)NCCCC[C@@H](C(=O)O)N', 'K(cr)': 'C/C=C/C(=O)NCCCC[C@@H](C(=O)O)N',
            'K(di)': 'CN(C)CCCC[C@@H](C(=O)O)N', 'K(fo)': 'C(CCNC=O)C[C@@H](C(=O)O)N',
            'K(glu)': 'C(CCNC(=O)CCCC(=O)O)C[C@@H](C(=O)O)N', 'K(gly)': 'C(CCNC(=O)CNC(=O)CN)C[C@@H](C(=O)O)N',
            'K(hy)': 'CC(C)(C(=O)NCCCC[C@@H](C(=O)O)N)O', 'K(ma)': 'C1C(=O)N(C1=O)CCCC[C@@H](C(=O)O)N', 
            'K(me)': 'CNCCCC[C@@H](C(=O)O)N', 'K(pr)': 'N[C@@H](CCCCNC(CC)=O)C(O)=O', 
            'K(su)': 'C(CCNC(=O)CCC(=O)O)C[C@@H](C(=O)O)N', 'K(tr)': 'C[N+](C)(C)CCCC[C@@H](C(=O)O)N', 
            'M(ox)': 'C[S@](=O)CC[C@@H](C(=O)O)N',
            'R(ci)': 'C(C[C@@H](C(=O)O)N)CNC(=O)N', 'R(di)': 'CN(C)C(=NCCC[C@@H](C(=O)O)N)N',
            'R(me)': 'C(NC(=N)NC)CC[C@@H](C(=O)O)N', 'P(hy)': 'C1[C@H](CN[C@@H]1C(=O)O)O',
            'Y(ni)': 'C1=CC(=C(C=C1C[C@@H](C(=O)O)N)[N+](=O)[O-])O', 'Y(ph)': 'C1=CC(=CC=C1C[C@@H](C(=O)O)N)OP(=O)(O)O'
        }

        self.maccs = maccs
        self.ecfp4 = ecfp4
        self.ecfp6 = ecfp6
        self.rdkit = rdkit
        self.graph = graph
        self.A = A
        self.W = W
        self.R = R

        self.maccs_dim = 167
        self.ecfp_dim = 1024  # Both ecfp4 and ecfp6 have the same dimension
        self.rdkit_dim = 2048

        if self.graph:
            self.graph_dict = {}  # Initialize the graph dictionary here

        self.initialize_fingerprints()

    def initialize_fingerprints(self):
        if self.maccs:
            self.maccs_dict = {aa: MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(smiles)) for aa, smiles in self.AA_dict.items()}
        if self.ecfp4:
            self.ecfp4_dict = {aa: AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), 2, nBits=1024) for aa, smiles in self.AA_dict.items()}
        if self.ecfp6:
            self.ecfp6_dict = {aa: AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), 3, nBits=1024) for aa, smiles in self.AA_dict.items()}
        if self.rdkit:
            self.rdkit_dict = {aa: AllChem.RDKFingerprint(Chem.MolFromSmiles(smiles)) for aa, smiles in self.AA_dict.items()}
        if self.graph:
            for aa, smiles in self.AA_dict.items():
                graph = smiles_utils.from_smiles(smiles, with_hydrogen=True)
                graph = Data(x=graph.x.float(), edge_index=graph.edge_index, edge_attr=graph.edge_attr)
                self.graph_dict[aa] = graph

    def T(self, fp, p, A=10, W=10, R=0.85):
        return (((np.sin(p/W))/A)+R) * np.array(fp)

    def fingerprint(self, seq, ptms, charge, NCE):
        pos = np.arange(1, len(seq) + 1) / len(seq)
        profp = np.zeros((MAX_PEPTIDE_LENGTH, self.get_total_dim() + 2))

        for i, (aa, ptm) in enumerate(zip(seq, ptms)):
            if i >= MAX_PEPTIDE_LENGTH or aa == 'X':
                continue

            aa_key = f"{aa}{ptm}" if ptm else aa  # Use AA+PTM as key if PTM is present
    
            temp = []
            # Check and append features based on the modified/unmodified AA
            if self.maccs and aa_key in self.maccs_dict:
                temp.append(self.T(self.maccs_dict[aa_key], pos[i], A=self.A, W=self.W, R=self.R))
            if self.ecfp4 and aa_key in self.ecfp4_dict:
                temp.append(self.T(self.ecfp4_dict[aa_key], pos[i], A=self.A, W=self.W, R=self.R))
            if self.ecfp6 and aa_key in self.ecfp6_dict:
                temp.append(self.T(self.ecfp6_dict[aa_key], pos[i], A=self.A, W=self.W, R=self.R))
            if self.rdkit and aa_key in self.rdkit_dict:
                temp.append(self.T(self.rdkit_dict[aa_key], pos[i], A=self.A, W=self.W, R=self.R))

            # Only process if temp is not empty (i.e., AA or AA with PTM was found in the dictionary)
            if temp:
                aa_fingerprint = np.concatenate(temp)  # This would be a 1D array for this AA
                # print('aa_fingerprint', aa_fingerprint.shape)
                # print('profp', profp[i, :-2].shape)
                profp[i, :-2] = aa_fingerprint
                profp[i, -2:] = [charge, NCE]  # Add charge and NCE at the end of this AA's fingerprint

        return profp


    def fingerprint2(self, seq, ptms):
        pos = np.arange(1, len(seq) + 1) / len(seq)
        profp = np.zeros((MAX_PEPTIDE_LENGTH+2, self.get_total_dim()))

        for i, (aa, ptm) in enumerate(zip(seq, ptms)):
            if i >= MAX_PEPTIDE_LENGTH+2 or aa == 'X':
                continue

            aa_key = f"{aa}{ptm}" if ptm else aa  # Use AA+PTM as key if PTM is present
    
            temp = []
            # Check and append features based on the modified/unmodified AA
            if self.maccs and aa_key in self.maccs_dict:
                temp.append(self.T(self.maccs_dict[aa_key], pos[i], A=self.A, W=self.W, R=self.R))
            if self.ecfp4 and aa_key in self.ecfp4_dict:
                temp.append(self.T(self.ecfp4_dict[aa_key], pos[i], A=self.A, W=self.W, R=self.R))
            if self.ecfp6 and aa_key in self.ecfp6_dict:
                temp.append(self.T(self.ecfp6_dict[aa_key], pos[i], A=self.A, W=self.W, R=self.R))
            if self.rdkit and aa_key in self.rdkit_dict:
                temp.append(self.T(self.rdkit_dict[aa_key], pos[i], A=self.A, W=self.W, R=self.R))

            # Only process if temp is not empty (i.e., AA or AA with PTM was found in the dictionary)
            if temp:
                aa_fingerprint = np.concatenate(temp)  # This would be a 1D array for this AA
                # print('aa_fingerprint', aa_fingerprint.shape)
                # print('profp', profp[i, :-2].shape)
                profp[i] = aa_fingerprint

        return profp


    def graph_gen(self, seq, ptms):
        # graphs = []

        # for i, (aa, ptm) in enumerate(zip(seq, ptms)):
        #     if i >= MAX_PEPTIDE_LENGTH or aa == 'X':
        #         continue

        #     aa_key = f"{aa}{ptm}" if ptm else aa  # Use AA+PTM as key if PTM is present
        #     if aa_key in self.graph_dict:
        #         # Directly use the graph representation
        #         graphs.append(self.graph_dict[aa_key])

        # return graphs
        # MAX_NODES = 30  # Example maximum number of nodes, adjust as needed
        # FEATURE_DIM = 9  # Example feature dimension, adjust as needed
        graphs = []

        for aa, ptm in zip(seq, ptms):
            if aa == 'X':  # Skip padding symbols
                continue
            aa_key = f"{aa}{ptm}" if ptm else aa
            if aa_key in self.graph_dict:
                graph = self.graph_dict[aa_key]

                graphs.append(graph)

        # If the sequence is shorter than MAX_PEPTIDE_LENGTH, pad the list with empty graphs
        # while len(graphs) < MAX_PEPTIDE_LENGTH:
        #     graphs.append(Data(x=torch.zeros((MAX_NODES, FEATURE_DIM), dtype=torch.float), edge_index=torch.empty((2, 0), dtype=torch.int), edge_attr=torch.empty((0,), dtype=torch.float)))

        # Convert the list of graphs into a batched graph object
        graphs = Batch.from_data_list(graphs)

        return graphs

    def get_total_dim(self):
        # Calculate the total dimension based on which fingerprints are enabled
        total_dim = 0
        if self.maccs:
            total_dim += self.maccs_dim
        if self.ecfp4:
            total_dim += self.ecfp_dim
        if self.ecfp6:
            total_dim += self.ecfp_dim
        if self.rdkit:
            total_dim += self.rdkit_dim
        return total_dim
