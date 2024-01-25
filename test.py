import os
from sacred import Experiment
from pathlib import Path
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset.dataset import MelDataset
from dataset.constants import *
from model import NoteTransformer
from utils import *

ex = Experiment("text_transcription")

@ex.config
def cfg():
    ckpt_id = "00045000"


@ex.automain
def test(logdir, device, data_path, n_layers, ckpt_id):
    logdir = Path(logdir)
    print_config(ex.current_run)

    data_path = Path(data_path)
    test_data = MelDataset(data_path / "mel",
                           data_path / "note",
                           data_path / "test.json",
                           device=device)

    model = NoteTransformer(kernel_size=9,
                            d_model=256,
                            d_inner=512,
                            n_layers=n_layers).to(device)
    ckpt_path = logdir / "ckpt" / ckpt_id
    ckpt_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt_dict["model"])

    loader = DataLoader(dataset, 1, shuffle=False, drop_last=False)

    for batch in tqdm(loader):
        with torch.no_grad():
            for i, x in tqdm(enumerate(loader)):
                mel = x["mel"].to(device)
                pitch = x["pitch"].to(device)
                start = x["start"].to(device)
                end = x["end"].to(device)

                pitch_p, start_p, end_p = model.predict(mel)

                if i < 5:
                    pred_list = get_list(pitch_p, start_p, end_p)
                    gt_list = get_list(pitch, start, end)

                    fig_pred = plot_midi(*pred_list)
                    fig.savefig("pred_%d.png" % i)
                    fig_gt = plot_midi(*gt_list)
                    fig.savefig("gt_%d.png" % i)
    
