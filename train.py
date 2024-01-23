import os
from sacred import Experiment
from sacred.commands import print_config, save_config
from sacred.observers import FileStorageObserver
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from dataset.dataset import MelDataset
from model import NoteTransformer
from tqdm import tqdm

ex = Experiment("train_transcriber")

def patch_trg(trg):
    gold = trg[:, 1:].contiguous().view(-1)
    trg = trg[:, :-1]
    return trg, gold
    

@ex.config
def config():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


@ex.automain
def train(logdir, device, n_layers, checkpoint_interval, batch_size,
          learning_rate, learning_rate_decay_steps,
          clip_gradient_norm, epochs, data_path):
    ex.observers.append(FileStorageObserver.create(logdir))
    sw = SummaryWriter(logdir)

    logdir = Path(logdir)
    print_config(ex.current_run)
    save_config(ex.current_run.config, logdir / "config.json")

    data_path = Path(data_path)
    train_data = MelDataset(data_path / "mel",
                            data_path / "note",
                            data_path / "train.json",
                            device=device)

    valid_data = MelDataset(data_path / "mel",
                            data_path / "note",
                            data_path / "valid.json",
                            device=device)

    train_loader = DataLoader(train_data, batch_size, shuffle=True, drop_last=False,
                              collate_fn=train_data.collate_fn)
    eval_loader = DataLoader(valid_data, 1, shuffle=False, drop_last=False,
                             collate_fn=valid_data.collate_fn)

    model = NoteTransformer(kernel_size=9,
                            d_model=256,
                            d_inner=512,
                            n_layers=n_layers)
    model = model.to(device)
    model.train()
    
    for e in range(epochs):
        for x in tqdm(train_loader):
            mel = x["mel"].to(device)
            pitch = x["pitch"].to(device)
            start = x["start"].to(device)
            end = x["end"].to(device)

            pitch_i, pitch_o = patch_trg(pitch)
            start_i, start_o = patch_trg(start)
            end_i, end_o = patch_trg(end)

            pitch_p, start_p, end_p = model(mel, pitch_i, start_i, end_i)
            # print(pitch_p.size())
            # print(start_p.size())
            # print(end_p.size())
            # _ = input()
        _ = input()

            
            

            
