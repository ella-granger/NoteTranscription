import os
from sacred import Experiment
from sacred.commands import print_config, save_config
from sacred.observers import FileStorageObserver
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from dataset.dataset import MelDataset
from model import NoteTransformer
from transformer.Optim import ScheduledOptim
from tqdm import tqdm
from dataset.constants import *
from utils import *

ex = Experiment("train_transcriber")

def patch_trg(trg):
    gold = trg[:, 1:].contiguous()
    trg = trg[:, :-1]
    return trg, gold


def masked_l1(pred, gt, mask):
    diff = pred - gt.unsqueeze(-1)
    loss = torch.sum(torch.abs(diff[mask.unsqueeze(-1)]))
    return loss


def masked_l2(pred, gt, mask):
    diff = pred - gt.unsqueeze(-1)
    loss = torch.sum(diff[mask.unsqueeze(-1)] ** 2)
    return loss


@ex.config
def config():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_interval = 5
    summary_interval = 20
    val_interval = 1000
    checkpoint_interval = 5000
    warmup_steps = 8000
    train_mode = "TS"


@ex.automain
def train(logdir, device, n_layers, checkpoint_interval, batch_size,
          learning_rate, warmup_steps,
          clip_gradient_norm, epochs, data_path,
          output_interval, summary_interval, val_interval,
          loss_norm, time_loss_alpha, train_mode):
    ex.observers.append(FileStorageObserver.create(logdir))
    sw = SummaryWriter(logdir)

    logdir = Path(logdir)
    print_config(ex.current_run)
    save_config(ex.current_run.config, logdir / "config.json")

    data_path = Path(data_path)
    train_data = MelDataset(data_path / "mel",
                            data_path / "note",
                            data_path / "train.json",
                            train_mode,
                            device=device)

    valid_data = MelDataset(data_path / "mel",
                            data_path / "note",
                            data_path / "valid.json",
                            train_mode,
                            device=device)

    train_loader = DataLoader(train_data, batch_size, shuffle=True, drop_last=False,
                              collate_fn=train_data.collate_fn)
    eval_loader = DataLoader(valid_data, 1, shuffle=False, drop_last=False,
                             collate_fn=valid_data.collate_fn)

    model = NoteTransformer(kernel_size=9,
                            d_model=256,
                            d_inner=512,
                            n_layers=n_layers,
                            train_mode=train_mode)
    model = model.to(device)

    optimizer = ScheduledOptim(
        optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09),
        2.0, 256, warmup_steps)

    
    model.train()
    step = 0
    for e in range(epochs):
        itr = tqdm(train_loader)
        for x in itr:
            mel = x["mel"].to(device)
            pitch = x["pitch"].to(device)

            start_i = None
            dur_i = None
            start_t_i = None
            end_i = None
            if "S" in train_mode:
                start = x["start"].to(device)
                dur = x["dur"].to(device)
            if "T" in train_mode:
                start_t = x["start_t"].to(device)
                end = x["end"].to(device)

            pitch_i, pitch_o = patch_trg(pitch)
            if "S" in train_mode:
                start_i, start_o = patch_trg(start)
                dur_i, dur_o = patch_trg(dur)
            if "T" in train_mode:
                start_t_i, start_t_o = patch_trg(start_t)
                end_i, end_o = patch_trg(end)

            optimizer.zero_grad()
            result = model(mel, pitch_i, start_i, dur_i, start_t_i, end_i)

            if train_mode == "S":
                pitch_p, start_p, dur_p = result
            elif train_mode == "T":
                pitch_p, start_t_p, end_p = result
            else:
                pitch_p, start_t_p, end_p, start_p, dur_p = result

            start_loss = 0
            dur_loss = 0
            start_t_loss = 0
            end_loss = 0
            pitch_loss = F.cross_entropy(torch.permute(pitch_p, (0, 2, 1)), pitch_o, ignore_index=PAD_IDX, reduction='sum')
            if "S" in train_mode:
                # print(start_p.size())
                # print(dur_p.size())
                # print(start_o)
                # print(dur_o)
                # _ = input()
                start_loss = F.cross_entropy(torch.permute(start_p, (0, 2, 1)), start_o, ignore_index=MAX_START+1, reduction='sum')
                dur_loss = F.cross_entropy(torch.permute(dur_p, (0, 2, 1)), dur_o, ignore_index=0, reduction='sum')
            if "T" in train_mode:
                seq_mask = (pitch_i != PAD_IDX) * (pitch_i != 0)
                start_t_loss = time_loss_alpha * masked_l1(start_t_p, start_t_o, seq_mask)
                end_loss = time_loss_alpha * masked_l1(end_p, end_o, seq_mask)
            
            loss = pitch_loss + start_loss + dur_loss + start_t_loss + end_loss
            loss.backward()
            optimizer.step_and_update_lr()

            if step % output_interval == 0:
                itr.set_description("pitch: %.2f" % (pitch_loss.item()))

            if step % summary_interval == 0:
                sw.add_scalar("training/loss", loss.item(), step)
                sw.add_scalar("training/pitch_loss", pitch_loss.item(), step)
                if "S" in train_mode:
                    sw.add_scalar("training/start_loss", start_loss.item(), step)
                    sw.add_scalar("training/dur_loss", dur_loss.item(), step)
                if "T" in train_mode:
                    sw.add_scalar("training/start_t_loss", start_t_loss.item(), step)
                    sw.add_scalar("training/end_loss", end_loss.item(), step)
                sw.add_scalar("training/lr", optimizer._optimizer.param_groups[0]["lr"], step)

            if step % checkpoint_interval == 0: # and step > 0:
                checkpoint_path = logdir / "ckpt"
                checkpoint_path.mkdir(exist_ok=True)
                checkpoint_path = checkpoint_path / ("%08d" % step)
                obj = {"model": model.state_dict(),
                       "optim": optimizer._optimizer.state_dict(),
                       "steps": step,
                       "epoch": e}
                torch.save(obj, str(checkpoint_path))

            if step % val_interval == 0:
                model.eval()
                with torch.no_grad():
                    total_loss = 0
                    total_pitch_loss = 0
                    total_start_loss = 0
                    total_dur_loss = 0
                    total_start_t_loss = 0
                    total_end_loss = 0
                    total_T = 0
                    total_start_T = 0
                    total_dur_T = 0
                    total_C = 0
                    total_count = 0
                    for i, batch in tqdm(enumerate(eval_loader)):
                        mel = batch["mel"].to(device)
                        pitch = batch["pitch"].to(device)

                        start_i = None
                        dur_i = None
                        start_t_i = None
                        end_i = None
                        if "S" in train_mode:
                            start = batch["start"].to(device)
                            dur = batch["dur"].to(device)
                        if "T" in train_mode:
                            start_t = batch["start_t"].to(device)
                            end = batch["end"].to(device)

                        pitch_i, pitch_o = patch_trg(pitch)
                        if "S" in train_mode:
                            start_i, start_o = patch_trg(start)
                            dur_i, dur_o = patch_trg(dur)
                        if "T" in train_mode:
                            start_t_i, start_t_o = patch_trg(start_t)
                            end_i, end_o = patch_trg(end)

                        result = model(mel, pitch_i, start_i, dur_i, start_t_i, end_i)
                        if train_mode == "S":
                            pitch_p, start_p, dur_p = result
                        elif train_mode == "T":
                            pitch_p, start_t_p, end_p = result
                        else:
                            pitch_p, start_t_p, end_p, start_p, dur_p = result

                        start_loss = 0
                        dur_loss = 0
                        start_t_loss = 0
                        end_loss = 0
                        pitch_loss = F.cross_entropy(torch.permute(pitch_p, (0, 2, 1)), pitch_o, ignore_index=PAD_IDX, reduction='sum')
                        if "S" in train_mode:
                            start_loss = F.cross_entropy(torch.permute(start_p, (0, 2, 1)), start_o, ignore_index=MAX_START+1, reduction='sum')
                            dur_loss = F.cross_entropy(torch.permute(dur_p, (0, 2, 1)), dur_o, ignore_index=0, reduction='sum')
                        if "T" in train_mode:
                            seq_mask = (pitch_i != PAD_IDX) * (pitch_i != 0)

                            start_t_loss = time_loss_alpha * masked_l1(start_t_p, start_t_o, seq_mask)
                            end_loss = time_loss_alpha * masked_l1(end_p, end_o, seq_mask)
            
                        loss = pitch_loss + start_loss + dur_loss + start_t_loss + end_loss

                        if i < 1:

                            if "S" in train_mode:
                                pred_list = get_list_s(pitch_p, start_p, dur_p)
                                gt_list = get_list_s(pitch_o, start_o, dur_o)
                                
                                pred_list = [(n, s, e) for n, s, e in zip(*pred_list)]
                                gt_list = [(n, s, e) for n, s, e in zip(*gt_list)]

                                sw.add_text("gt/s_%d" % i, str(gt_list), step)
                                sw.add_text("pred/s_%d" % i, str(pred_list), step)

                                sw.add_figure("gt/s_%d" % i, plot_score(pitch_o, start_o, dur_o), step)
                                sw.add_figure("pred/s_%d" % i, plot_score(pitch_p, start_p, dur_p), step)
                            if "T" in train_mode:
                                pred_list = get_list_t(pitch_p, start_t_p, end_p)
                                gt_list = get_list_t(pitch_o, start_t_o, end_o)

                                pred_list = [(n, s, e) for n, s, e in zip(*pred_list)]
                                gt_list = [(n, s, e) for n, s, e in zip(*gt_list)]

                                sw.add_text("gt/t_%d" % i, str(gt_list), step)
                                sw.add_text("pred/t_%d" % i, str(pred_list), step)
                                
                                sw.add_figure("gt/t_%d" % i, plot_midi(pitch_o, start_t_o, end_o), step)
                                sw.add_figure("pred/t_%d" % i, plot_midi(pitch_p, start_t_p, end_p), step)

                        pitch_pred = torch.argmax(pitch_p, dim=-1)
                        if "S" in train_mode:
                            start_pred = torch.argmax(start_p, dim=-1)
                            dur_pred = torch.argmax(dur_p, dim=-1)
                            total_start_T += torch.sum(start_pred == start_o).item()
                            total_dur_T += torch.sum(dur_pred == dur_o).item()
                        total_T += torch.sum(pitch_pred == pitch_o).item()
                        total_C += pitch_pred.size(1)
                        total_loss += loss.item()
                        total_pitch_loss += pitch_loss.item()
                        if "S" in train_mode:
                            total_start_loss += start_loss.item()
                            total_dur_loss += dur_loss.item()
                        if "T" in train_mode:
                            total_start_t_loss += start_t_loss.item()
                            total_end_loss += end_loss.item()
                        total_count += 1

                eval_loss = total_loss / total_count
                eval_pitch_loss = total_pitch_loss / total_count
                sw.add_scalar("eval/loss", eval_loss, step)
                sw.add_scalar("eval/pitch_loss", eval_pitch_loss, step)
                if "S" in train_mode:
                    eval_start_loss = total_start_loss / total_count
                    eval_dur_loss = total_dur_loss / total_count
                    sw.add_scalar("eval/start_loss", eval_start_loss, step)
                    sw.add_scalar("eval/dur_loss", eval_dur_loss, step)
                if "T" in train_mode:
                    eval_start_t_loss = total_start_t_loss / total_count
                    eval_end_loss = total_end_loss / total_count
                    sw.add_scalar("eval/start_t_loss", eval_start_t_loss, step)
                    sw.add_scalar("eval/end_loss", eval_end_loss, step)
                sw.add_scalar("eval/pitch_prec", total_T / total_C, step)
                if "S" in train_mode:
                    sw.add_scalar("eval/start_prec", total_start_T / total_C, step)
                    sw.add_scalar("eval/dur_prec", total_dur_T / total_C, step)
                print(eval_loss, total_T / total_C)
                model.train()
                
            step += 1 
