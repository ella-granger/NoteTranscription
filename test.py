import os
from sacred import Experiment
from sacred.commands import print_config
from pathlib import Path
import numpy as np
import torch
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
    ckpt_id = "00060000"


@ex.automain
def test(logdir, device, data_path, n_layers, ckpt_id,
        checkpoint_interval, batch_size, learning_rate, warmup_steps,
        clip_gradient_norm, epochs, output_interval, summary_interval,
         val_interval, loss_norm, time_loss_alpha, train_mode):
    logdir = Path(logdir)
    print_config(ex.current_run)

    data_path = Path(data_path)
    test_data = MelDataset(data_path / "mel",
                           data_path / "note",
                           data_path / "test.json",
                           train_mode,
                           device=device)

    model = NoteTransformer(kernel_size=9,
                            d_model=256,
                            d_inner=512,
                            n_layers=n_layers,
                            train_mode=train_mode).to(device)
    ckpt_path = logdir / "ckpt" / ckpt_id
    ckpt_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt_dict["model"])
    model = model.double()

    loader = DataLoader(test_data, 1, shuffle=False, drop_last=False)

    with torch.no_grad():
        for i, x in tqdm(enumerate(loader)):
            mel = x["mel"].to(device).double()
            pitch = x["pitch"].to(device)
            start = None
            dur = None
            start_t = None
            end = None
            if "S" in train_mode:
                start = x["start"].to(device)
                dur = x["dur"].to(device)
            if "T" in train_mode:
                start_t = x["start_t"].to(device)
                end = x["end"].to(device)

            result = model.predict(mel)

            if train_mode == "S":
                pitch_p, start_p, dur_p = result
            elif train_mode == "T":
                pitch_p, start_t_p, end_p = result
            else:
                pitch_p, start_p, dur_p, start_t_p, end_p = result

            if i < 5:
                if "S" in train_mode:
                    pred_list = get_list_s(pitch_p, start_p, dur_p)
                    gt_list = get_list_s(pitch, start, dur)
                    fig_pred = plot_score(*pred_list)
                    fig_pred.savefig("pred/pred_score_%d.png" % i)
                    fig_gt = plot_score(*gt_list)
                    fig_gt.savefig("pred/gt_score_%d.png" % i)

                if "T" in train_mode:
                    pred_list = get_list_t(pitch_p, start_t_p, end_p)
                    gt_list = get_list_t(pitch, start_t, end)
                    
                    fig_pred = plot_midi(*pred_list)
                    fig_pred.savefig("pred/pred_trans_%d.png" % i)
                    fig_gt = plot_midi(*gt_list)
                    fig_gt.savefig("pred/gt_trans_%d.png" % i)
            else:
                break
    
