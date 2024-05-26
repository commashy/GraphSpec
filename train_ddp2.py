#!/usr/bin/python -u
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Only the first GPU will be visible

import sys
import torch
torch.backends.cudnn.benchmark = True
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
from torch.utils.data.distributed import DistributedSampler
from torch.nn.functional import cosine_similarity
from torch.optim.lr_scheduler import LambdaLR

from torch.profiler import profile, record_function, ProfilerActivity

import math
from tqdm import tqdm
from utils import *
from model import *
import h5py

from multiprocessing import Pool
import numpy as np

from AmorProt import AmorProtV2

import wandb
import os
import random
import pickle

import socket

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


# Specify the checkpoint directory
checkpoint_dir = "checkpoints_finetune"  # This will create a "checkpoints" directory in the current working directory
os.makedirs(checkpoint_dir, exist_ok=True)

def setup(rank, world_size, free_port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(free_port)  # Use the free port found
    # os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    # Initialize wandb on the main process
    if rank == 0:
        # wandb.init(project="FYT_physical_properties",
        wandb.init(project="FYT_finetune2_01p",
                   config={
                       "learning_rate": 0.00001,
                       "architecture": "seq2ms with 1280 channels",
                       "dataset": "FULL",
                       "epochs": 30,
                   })

def cleanup():
    dist.destroy_process_group()

# Custom Dataset class for PyTorch
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.length = len(self.data)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        seq_ptms, charge, NCE, length = prep_graph(self.data[index])
        spectra = spectrum2vector(self.data[index]['mz'], self.data[index]['it'], BIN_SIZE, self.data[index]['Charge'])
        return seq_ptms, charge, NCE, length, spectra

def custom_collate_fn(batch):
    """
    Collate function to process a batch of data.
    
    Args:
        batch: A list of tuples, each containing (seq_ptms, charge, NCE, spectra) from the dataset.
    
    Returns:
        A tuple of tensors: (seq_ptms_batch, charges, NCEs, spectra_batch) ready for model input.
    """
    
    # Initialize lists to store batched data
    seq_ptms_batch = []
    charges = []
    NCEs = []
    lengths = []
    spectra_batch = []

    # Process each item in the batch
    for seq_ptms, charge, NCE, length, spectra in batch:  
        # Append to batch lists
        seq_ptms_batch.append(seq_ptms)
        charges.append(charge)
        NCEs.append(NCE)
        lengths.append(length)
        # Convert numpy.ndarray to tensor before appending to spectra_batch
        spectra_tensor = torch.tensor(spectra, dtype=torch.float32)  # Use torch.from_numpy(spectra) if spectra is a numpy array
        spectra_batch.append(spectra_tensor)
    
    # Convert lists to tensors
    # For seq_ptms, we'll keep it as a list of lists for now, assuming further processing is done later
    charges = torch.tensor(charges, dtype=torch.float32)
    NCEs = torch.tensor(NCEs, dtype=torch.float32)
    lengths = torch.tensor(lengths)
    spectra_batch = torch.stack(spectra_batch)  # Assuming spectra is already a tensor

    return seq_ptms_batch, charges, NCEs, lengths, spectra_batch


def save_checkpoint(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)

# Function to evaluate the model
def evaluate(model, data_loader, device, criterion=masked_spectral_distance, ap=None):
    model.eval()
    total_loss = 0
    total_cosine_similarity = 0
    with torch.no_grad():
        for i, (seq_ptms, charge, NCE, length, spectra) in enumerate(data_loader):
            spectra = spectra.to(device)
            
            # Convert seq_lists, charge_list, and NCE_list to tensors
            charge = charge.to(device=device, dtype=torch.float32).detach().requires_grad_(True)
            NCE = NCE.to(device=device, dtype=torch.float32).detach().requires_grad_(True)
            length = length.to(device)
            graphs = embed_graph2(seq_ptms, ap)
            graphs = graphs.to(device)

            outputs = model(graphs, charge, NCE, length)
            
            loss = criterion(outputs, spectra)
            total_loss += loss.item()
            # Compute cosine similarity
            cos_sim = cosine_similarity(outputs, spectra).mean()
            total_cosine_similarity += cos_sim.item()
    
    avg_loss = total_loss / len(data_loader)
    avg_cosine_similarity = total_cosine_similarity / len(data_loader)
    return avg_loss, avg_cosine_similarity


def train(rank, world_size, free_port):
    setup(rank, world_size, free_port)

    # Constants
    batch_size = 64
    num_epochs = 30
    log_interval = 10

    # Check if GPU is available and set the device accordingly
    device = torch.device("cuda:{}".format(rank))

    # Model, Optimizer, Criterion
    if rank == 0:
        print("Building model.....")
    model = GraphSpectraModel().to(device)
    if rank == 0:
        print("Model building done!")
    # ddp_model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    # optimizer = optim.Adam(ddp_model.parameters(), lr=0.0001)
    optimizer = optim.Adam(model.parameters(), lr=0.00005)
    # optimizer = optim.AdamW(ddp_model.parameters(), lr=0.0001, weight_decay=0.05)
    criterion = masked_spectral_distance

    checkpoint = torch.load('/home/johaa/swinms/checkpointsgraph/checkpoint_epoch_30.pth.tar')

    # If originally saved with DDP, adjust the state dict
    base_state_dict = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(base_state_dict)
    # optimizer.load_state_dict(checkpoint['optimizer'])

    # After loading, wrap with DDP
    ddp_model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    # ddp_model.load_state_dict(checkpoint['state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer'])

    if rank == 0:
        print('Loading training data...')

    with open('/data/jerry/ptm/linked_data.pickle', 'rb') as f:
        linked_data = pickle.load(f)

    # ptm_description = 'Succinyl (K)'
    import os
    ptm_description = os.getenv('PTM_DESCRIPTION', 'Succinyl (K)')

    inclusion_percentage = 0.1


    linked_train, linked_test = partition_data_by_specific_ptm(linked_data, ptm_description, inclusion_percentage)

    print('Training set size:', len(linked_train))
    print('Validation set size:', len(linked_test))

    # random.Random(4).shuffle(linked_data)
    # split = int(len(linked_data)*0.8)
    # linked_train = linked_data[:split]
    # linked_test = linked_data[split:]

    # Massive = readmgf("/home/johaa/swinms/dataset/MassIVE.mgf")
    # if rank == 0:
    #     print('Loaded training data 1/4')

    # PT = readmgf("/home/johaa/swinms/dataset/ProteomeTools.mgf")
    # if rank == 0:
    #     print('Loaded training data 2/4')

    # NIST = readmgf("/home/johaa/swinms/dataset/NIST.mgf")
    # if rank == 0:
    #     print('Loaded training data 3/4')

    # NISTsyn = readmgf("/home/johaa/swinms/dataset/NIST_Synthetic.mgf")
    # if rank == 0:
    #     print('Loaded training data 4/4')

    # train_data = Massive + PT + NIST + NISTsyn + linked_train
    # train_data = Massive + PT + NIST + NISTsyn
    train_data = linked_train
    if rank == 0:
        print('Concatenated training data')

    train_dataset = CustomDataset(train_data)
    if rank == 0:
        print('Created training dataset')
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, pin_memory=True, collate_fn=custom_collate_fn)
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    if rank == 0:
        print('Training dataset size:', len(train_data))


    # test_path = 'test_esm.h5'
    test_data = readmgf('/home/johaa/swinms/dataset/hcd_testingset.mgf')
    # test_data = test_data + linked_test
    test_data = test_data
    test_dataset = CustomDataset(test_data)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler, pin_memory=True, collate_fn=custom_collate_fn)

    test_data2 = linked_test
    test_dataset2 = CustomDataset(test_data2)
    test_sampler2 = DistributedSampler(test_dataset2, num_replicas=world_size, rank=rank, shuffle=False)
    test_loader2 = DataLoader(test_dataset2, batch_size=batch_size, sampler=test_sampler2, pin_memory=True, collate_fn=custom_collate_fn)
    if rank == 0:
        print('Testing dataset size:', len(test_dataset))

    # Learning rate scheduler function
    def lr_lambda(epoch):
        # if epoch < 10:
        #     return (epoch + 1) / 5  # Linear warm up for the first 5 epochs
        if epoch < 10:
            return 1.0  # Constant learning rate for the next 5 epochs
        else:
            cosine_decay = 0.5 * (1 + math.cos(math.pi * (epoch - 10) / (num_epochs - 10)))
            return cosine_decay

    # return math.exp(-0.05 * (epoch - 10))

    scheduler = LambdaLR(optimizer, lr_lambda)

    # Training Loop
    ap = AmorProtV2(maccs=False, ecfp4=False, ecfp6=False, rdkit=False, graph=True)

    best_loss = float('inf')
    for epoch in range(num_epochs):
        ddp_model.train()
        train_sampler.set_epoch(epoch)
        running_loss = 0.0
        running_cosine_similarity = 0.0  # Assuming you have a way to calculate this in your loop

        # Wrap your data loader with tqdm for a progress bar
        train_loader_with_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for i, (seq_ptms, charge, NCE, length, spectra) in enumerate(train_loader_with_progress):
            spectra = spectra.to(device)
            
            # Convert seq_lists, charge_list, and NCE_list to tensors
            charge = charge.to(device=device, dtype=torch.float32).detach().requires_grad_(True)
            NCE = NCE.to(device=device, dtype=torch.float32).detach().requires_grad_(True)
            length = length.to(device)
            graphs = embed_graph2(seq_ptms, ap)
            graphs = graphs.to(device)

            optimizer.zero_grad()
            outputs = ddp_model(graphs, charge, NCE, length)
            loss = criterion(outputs, spectra)
            loss.backward()
            optimizer.step()

            # Accumulate the loss and cosine similarity
            running_loss += loss.item()
            current_cosine_similarity = cosine_similarity(outputs, spectra).mean()
            running_cosine_similarity += current_cosine_similarity.item()

            if rank == 0:
                # Update the progress bar with the current loss
                wandb.log({"loss": loss.item(), "cosine_similarity": current_cosine_similarity.item()})

            # Update the progress bar with the current loss
            train_loader_with_progress.set_postfix({'Loss': loss.item()})

        scheduler.step()

        # Calculate training and validation loss
        avg_train_loss = running_loss / len(train_loader)
        avg_train_cosine_similarity = running_cosine_similarity / len(train_loader)
        val_loss, val_cosine_similarity = evaluate(ddp_model, test_loader, device, criterion, ap)
        val_loss2, val_cosine_similarity2 = evaluate(ddp_model, test_loader2, device, criterion, ap)

        # Evaluate model and print results (only by rank 0 to avoid duplicate logs)
        if rank == 0:
            wandb.log({"val_loss": val_loss, "val_cosine_similarity": val_cosine_similarity, "avg_train_loss": avg_train_loss, "avg_train_cosine_similarity": avg_train_cosine_similarity, "val_loss_ptm": val_loss2, "val_cosine_similarity_ptm": val_cosine_similarity2, "learning_rate": scheduler.get_last_lr()[0]})

            if val_loss < best_loss:
                best_loss = val_loss
                checkpoint_path = os.path.join(checkpoint_dir, f"graph_checkpoint_epoch_{epoch+1}.pth.tar")
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_loss': best_loss,
                    'optimizer': optimizer.state_dict(),
                }, filename=checkpoint_path)

            print(f'Epoch {epoch+1}/{num_epochs}')
            print(f'Training Loss: {avg_train_loss:.4f}')
            print(f'Training Cosine Similarity: {avg_train_cosine_similarity:.4f}')
            print(f'Validation Loss: {val_loss:.4f}')
            print(f'Validation Cosine Similarity: {val_cosine_similarity:.4f}')
            print(f'Validation Loss_ptm: {val_loss2:.4f}')
            print(f'Validation Cosine Similarity_ptm: {val_cosine_similarity2:.4f}')
            print(f'Learning Rate: {scheduler.get_last_lr()[0]:.6f}')

    cleanup()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    free_port = find_free_port()
    torch.multiprocessing.spawn(train, args=(world_size,free_port,), nprocs=world_size, join=True)
