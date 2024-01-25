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


def plot_midi(pitch, start, end):
    pitch, start, end = get_list(pitch, start, end)

    fig, ax = plt.subplots(figsize=(10, 4))
    for n, s, e in zip(pitch, start, end):
        if s <= e:
            ax.hlines(n, s, e, linewidths=3)
    fig.canvas.draw()
    plt.close()
    return fig


def get_list(pitch, start, end):
    pitch = pitch.detach().cpu().numpy()[0]
    start = start.detach().cpu().numpy()[0]
    end = end.detach().cpu().numpy()[0]

    if len(pitch.shape) > 1:
        pitch = np.argmax(pitch, axis=1, keepdims=False)
        start = start.reshape(-1)
        end = end.reshape(-1)

    return pitch.tolist(), start.tolist(), end.tolist()


@ex.config
def config():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_interval = 5
    summary_interval = 20
    val_interval = 1000
    checkpoint_interval = 5000


@ex.automain
def train(logdir, device, n_layers, checkpoint_interval, batch_size,
          learning_rate, learning_rate_decay_steps,
          clip_gradient_norm, epochs, data_path,
          output_interval, summary_interval, val_interval,
          loss_norm, time_loss_alpha):
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

    optimizer = ScheduledOptim(
        optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09),
        2.0, 256, 8000)

    
    model.train()
    step = 0
    for e in range(epochs):
        itr = tqdm(train_loader)
        for x in itr:
            mel = x["mel"].to(device)
            pitch = x["pitch"].to(device)
            start = x["start"].to(device)
            end = x["end"].to(device)

            pitch_i, pitch_o = patch_trg(pitch)
            start_i, start_o = patch_trg(start)
            end_i, end_o = patch_trg(end)

            optimizer.zero_grad()
            pitch_p, start_p, end_p = model(mel, pitch_i, start_i, end_i)

            pitch_loss = F.cross_entropy(torch.permute(pitch_p, (0, 2, 1)), pitch_o, ignore_index=PAD_IDX, reduction='sum')
            seq_mask = (pitch_i != PAD_IDX)
            if loss_norm == 1:
                start_loss = time_loss_alpha * masked_l1(start_p, start_o, seq_mask)
                end_loss = time_loss_alpha * masked_l1(end_p, end_o, seq_mask)
            else:
                start_loss = time_loss_alpha * masked_l2(start_p, start_o, seq_mask)
                end_loss = time_loss_alpha * masked_l2(end_p, end_o, seq_mask)
            loss = pitch_loss + start_loss + end_loss
            loss.backward()
            optimizer.step_and_update_lr()

            if step % output_interval == 0:
                itr.set_description("pitch: %.2f, start: %.4f, end: %.4f" % (pitch_loss.item(), start_loss.item(), end_loss.item()))

            if step % summary_interval == 0:
                sw.add_scalar("training/loss", loss.item(), step)
                sw.add_scalar("training/pitch_loss", pitch_loss.item(), step)
                sw.add_scalar("training/start_loss", start_loss.item(), step)
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
                    total_end_loss = 0
                    total_T = 0
                    total_C = 0
                    total_count = 0
                    for i, batch in tqdm(enumerate(eval_loader)):
                        mel = batch["mel"].to(device)
                        pitch = batch["pitch"].to(device)
                        start = batch["start"].to(device)
                        end = batch["end"].to(device)

                        pitch_i, pitch_o = patch_trg(pitch)
                        start_i, start_o = patch_trg(start)
                        end_i, end_o = patch_trg(end)

                        pitch_p, start_p, end_p = model(mel, pitch_i, start_i, end_i)

                        pitch_loss = F.cross_entropy(torch.permute(pitch_p, (0, 2, 1)), pitch_o, ignore_index=PAD_IDX, reduction='sum')
                        seq_mask = (pitch_i != PAD_IDX)
                        if loss_norm == 1:
                            start_loss = time_loss_alpha * masked_l1(start_p, start_o, seq_mask)
                            end_loss = time_loss_alpha * masked_l1(end_p, end_o, seq_mask)
                        else:
                            start_loss = time_loss_alpha * masked_l2(start_p, start_o, seq_mask)
                            end_loss = time_loss_alpha * masked_l2(end_p, end_o, seq_mask)
                        loss = pitch_loss + start_loss + end_loss

                        if i < 1:
                            sw.add_figure("gt/%d" % i, plot_midi(pitch_o, start_o, end_o), step)
                            sw.add_figure("pred/%d" % i, plot_midi(pitch_p, start_p, end_p), step)

                            pred_list = get_list(pitch_p, start_p, end_p)
                            gt_list = get_list(pitch_o, start_o, end_o)

                            pred_list = [(n, s, e) for n, s, e in zip(*pred_list)]
                            gt_list = [(n, s, e) for n, s, e in zip(*gt_list)]

                            sw.add_text("gt/%d" % i, str(gt_list), step)
                            sw.add_text("pred/%d" % i, str(pred_list), step)

                        pitch_pred = torch.argmax(pitch_p, dim=-1)
                        total_T += torch.sum(pitch_pred == pitch_o).item()
                        total_C += pitch_pred.size(1)
                        total_loss += loss.item()
                        total_pitch_loss += pitch_loss.item()
                        total_start_loss += start_loss.item()
                        total_end_loss += end_loss.item()
                        total_count += 1

                eval_loss = total_loss / total_count
                eval_pitch_loss = total_pitch_loss / total_count
                eval_start_loss = total_start_loss / total_count
                eval_end_loss = total_end_loss / total_count
                sw.add_scalar("eval/loss", eval_loss, step)
                sw.add_scalar("eval/pitch_loss", eval_pitch_loss, step)
                sw.add_scalar("eval/start_loss", eval_start_loss, step)
                sw.add_scalar("eval/end_loss", eval_end_loss, step)
                sw.add_scalar("eval/pitch_prec", total_T / total_C, step)
                print(eval_loss, eval_pitch_loss, eval_start_loss, eval_end_loss, total_T / total_C)
                model.train()
                
            step += 1

            
            

            
