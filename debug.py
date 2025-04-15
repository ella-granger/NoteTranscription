import os
from sacred import Experiment
from sacred.commands import print_config, save_config
from sacred.observers import FileStorageObserver
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.beta import Beta
import torchaudio
import torch_optimizer as optim

import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from dataset.dataset import MelDataset
from model import NoteTransformer, get_trg_mask
from transformer.Optim import ScheduledOptim
from mir_metrics import cal_mir_metrics
from tqdm import tqdm
from dataset.constants import *
from utils import *
# import torch.multiprocessing as mp
import time
# mp.set_start_method("fork", force=True)


ex = Experiment("debug_transcriber")


def patch_trg(trg):
    gold = trg[:, 1:].contiguous()
    trg = trg[:, :-1]
    return trg, gold


def masked_bce_loss(pred, gt, mask):
    batch_sum = 0
    for i in range(gt.size(0)):
        p = pred[i][mask[i]]
        g = gt[i][mask[i]]
        batch_sum += F.binary_cross_entropy(p, g, reduce="sum")
    return batch_sum

def masked_sig_log(pred, gt, mask):
    batch_sum = 0
    for i in range(gt.size(0)):
        mu = pred[i, :, 0][mask[i]]
        sigma = pred[i, :, 1][mask[i]]
        # print("mu:", mu)
        # print("sigma:", sigma)
        dist = build_sigmoid_logistics(mu, sigma)
        tar = gt[i, :][mask[i]]
        # print("target:", tar)
        nll = torch.sum(-dist.log_prob(tar))
        batch_sum += nll
    return batch_sum


@ex.config
def config():
    # default settings, will be changed by configuration json
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    prob_model = "gaussian"
    output_interval = 5
    summary_interval = 20
    val_interval = 1000
    checkpoint_interval = 5000
    warmup_steps = 8000
    seg_len = 320
    mix_k = 0.000001
    epsilon = 0.1
    loss_norm = 1
    time_lambda = 3
    enable_encoder = True
    scheduled_sampling = False
    scst = False


@ex.automain
def debug(logdir, device, n_layers, checkpoint_interval, batch_size,
          learning_rate, warmup_steps, mix_k, ss_epsilon, total_steps,
          clip_gradient_norm, epochs, data_path, scst, scst_step,
          output_interval, summary_interval, val_interval,
          loss_norm, enable_encoder, scheduled_sampling_step,
          scheduled_sampling, prob_model, seg_len, time_lambda):
    print("DEBUG START")
    ex.observers.append(FileStorageObserver.create(logdir))
    sw = SummaryWriter(logdir)
    
    logdir = Path(logdir)
    print_config(ex.current_run)
    save_config(ex.current_run.config, logdir / "config.json")

    data_path = Path(data_path)
    valid_data = MelDataset(data_path / "mel",
                            data_path / "note",
                            data_path / "valid.json",
                            seg_len=seg_len,
                            device=device)

    eval_loader = DataLoader(valid_data, 1, shuffle=False, drop_last=False,
                             collate_fn=valid_data.collate_fn, num_workers=16, pin_memory=True)

    model = NoteTransformer(kernel_size=9,
                            d_model=256,
                            d_inner=512,
                            n_layers=n_layers,
                            seg_len=seg_len,
                            enable_encoder=enable_encoder,
                            prob_model=prob_model).to(device)


    ckpt_path = logdir / "ckpt" / "cur"
    ckpt_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt_dict["model"])
    model.eval()

    for i, x in enumerate(eval_loader):
        
        mel = x["mel"].to(device)
        pitch = x["pitch"].to(device)
        voice = x["voice"].to(device)
        start = x["start"].to(device)
        dur = x["dur"].to(device)

        pitch_i, pitch_o = patch_trg(pitch)
        voice_i, voice_o = patch_trg(voice)
        start_i, start_o = patch_trg(start)
        dur_i, dur_o = patch_trg(dur)
        
        print(pitch_o)
        print(voice_o)
        print(start_o)
        print(dur_o)

        with torch.no_grad():
            teacher_forcing = model(mel, pitch_i, start_i, dur_i, voice_i)
            pitch_p, start_p, dur_p, voice_p = teacher_forcing
            pitch_loss = F.cross_entropy(torch.permute(pitch_p, (0, 2, 1)), pitch_o, ignore_index=PAD_IDX, reduction="sum")
            seq_mask = (pitch_o != PAD_IDX) * (pitch_o != 0)
            voice_loss = masked_bce_loss(voice_p, voice_o, seq_mask)
            start_loss = masked_sig_log(start_p, start_o, seq_mask)
            dur_loss = masked_sig_log(dur_p, dur_o, seq_mask)

            print(pitch_loss.item())
            print(voice_loss.item())
            print(start_loss.item())
            print(dur_loss.item())
            
            greedy, greedy_p = model.sample(mel, "greedy")
            gen, ll, gen_p = model.sample(mel, "sample")
            print(ll.item())
        
        fig, axs = plt.subplots(2, 2)
        pitch_gt = torch.zeros(pitch_o.size(1), teacher_forcing[0].size(-1))
        pitch_gt[torch.arange(pitch_o.size(1)), pitch_o[0]] = 1
        im = axs[0, 0].imshow(pitch_gt, aspect="auto", origin="lower", interpolation="none")
        plt.colorbar(im, ax=axs[0, 0])
        axs[0,0].title.set_text("GT")

        im = axs[0, 1].imshow(F.softmax(teacher_forcing[0], dim=-1).detach().cpu().numpy()[0], aspect="auto", origin="lower", interpolation="none")
        plt.colorbar(im, ax=axs[0, 1])
        axs[0, 1].title.set_text("TF")
        
        im = axs[1, 0].imshow(greedy_p[0].detach().cpu().numpy(), aspect="auto", origin="lower", interpolation="none")
        plt.colorbar(im, ax=axs[1, 0])
        axs[1, 0].title.set_text("GR")
        
        im = axs[1, 1].imshow(gen_p[0].detach().cpu().numpy(), aspect="auto", origin="lower", interpolation="none")
        plt.colorbar(im, ax=axs[1, 1])
        axs[1, 1].title.set_text("SP")
        
        plt.tight_layout()
        plt.savefig("pitch.png")

        plt.clf()
        colors = ["red", "yellow", "green", "blue", "black"]
        fig, axs = plt.subplots(2, 2)
        for j, (s, e, voice) in enumerate(zip(start_o[0].detach().cpu(),
                                              dur_o[0].detach().cpu(),
                                              voice_o[0].detach().cpu())):
            if sum(voice) == 0:
                axs[0,0].hlines(j, s, e, "black", linewidths=3, alpha=0.4)
            else:
                for k, v in enumerate(voice):
                    if v > 0.5:
                        axs[0,0].hlines(j, s, e, colors[k], linewidths=3, alpha=0.4)
        axs[0,0].title.set_text("GT")

        for j, (s, e, voice) in enumerate(zip(F.sigmoid(teacher_forcing[1][0, :, 0]).detach().cpu(),
                                              F.sigmoid(teacher_forcing[2][0, :, 0]).detach().cpu(),
                                              teacher_forcing[3][0].detach().cpu())):
            if sum(voice) == 0:
                axs[0,1].hlines(j, s, e, "black", linewidths=3, alpha=0.4)
            else:
                for k, v in enumerate(voice):
                    if v > 0.5:
                        axs[0,1].hlines(j, s, e, colors[k], linewidths=3, alpha=0.4)
        axs[0,1].title.set_text("TF")

        for j, (s, e, voice) in enumerate(zip(F.sigmoid(greedy_p[2][:, 0]).detach().cpu(),
                                              F.sigmoid(greedy_p[3][:, 0]).detach().cpu(),
                                              greedy_p[1].detach().cpu())):
            if sum(voice) == 0:
                axs[1,0].hlines(j, s, e, "black", linewidths=3, alpha=0.4)
            else:
                for k, v in enumerate(voice):
                    if v > 0.5:
                        axs[1,0].hlines(j, s, e, colors[k], linewidths=3, alpha=0.4)
        axs[1,0].title.set_text("GR")

        for j, (s, e, voice) in enumerate(zip(F.sigmoid(gen_p[2][:, 0]).detach().cpu(),
                                              F.sigmoid(gen_p[3][:, 0]).detach().cpu(),
                                              gen_p[1].detach().cpu())):
            if sum(voice) == 0:
                axs[1,1].hlines(j, s, e, "black", linewidths=3, alpha=0.4)
            else:
                for k, v in enumerate(voice):
                    if v > 0.5:
                        axs[1,1].hlines(j, s, e, colors[k], linewidths=3, alpha=0.4)
        axs[1,1].title.set_text("SP")
        
        plt.tight_layout()
        plt.savefig("time&voice.png")
        _ = input()
