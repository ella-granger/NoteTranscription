import os
from sacred import Experiment
from sacred.commands import print_config, save_config
from sacred.observers import FileStorageObserver
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import optim
import torch.nn.functional as F
from torch.distributions.normal import Normal
import torchaudio
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


def masked_normal_nll(pred, gt, mask):
    batch_sum = 0
    for i in range(gt.size(0)):
        mu = pred[i, :, 0][mask[i]]
        var = pred[i, :, 1][mask[i]]
        tar = gt[i, :][mask[i]]
        # print(mask[i])
        # print(mu)
        # print(var)
        # print(tar)
        batch = F.gaussian_nll_loss(mu, tar, var, reduction="sum")
        # print(batch)
        # print(F.gaussian_nll_loss(mu, tar, var, reduction="mean"))
        # print(F.gaussian_nll_loss(mu, tar, var, reduction="none"))
        # _ = input()
        batch_sum += batch

    return batch_sum


def masked_l2(pred, gt, mask):
    diff = pred - gt.unsqueeze(-1)
    loss = torch.sum(diff[mask.unsqueeze(-1)] ** 2)
    return loss


def get_mix_t(step, k, epsilon):
    # return 1
    if step < 160000:
        t = 1
    else:
        t = max(epsilon, 1 - k * (step - 160000))
    return t


@ex.config
def config():
    # default settings, will be changed by configuration json
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_interval = 5
    summary_interval = 20
    val_interval = 1000
    checkpoint_interval = 5000
    warmup_steps = 8000
    train_mode = "TS"
    mix_k = 0.000003
    epsilon = 0.1
    enable_encoder = True
    scheduled_sampling = False


@ex.automain
def train(logdir, device, n_layers, checkpoint_interval, batch_size,
          learning_rate, warmup_steps, mix_k, epsilon,
          clip_gradient_norm, epochs, data_path,
          output_interval, summary_interval, val_interval,
          loss_norm, time_loss_alpha, train_mode, enable_encoder,
          scheduled_sampling):
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
                              collate_fn=train_data.collate_fn, num_workers=8)
    eval_loader = DataLoader(valid_data, 1, shuffle=False, drop_last=False,
                             collate_fn=valid_data.collate_fn, num_workers=8)

    model = NoteTransformer(kernel_size=9,
                            d_model=256,
                            d_inner=512,
                            n_layers=n_layers,
                            train_mode=train_mode,
                            enable_encoder=enable_encoder)
    total_params = sum(p.numel() for p in model.parameters())
    print(total_params)
    # exit()
    model = model.to(device)
    
    optimizer = ScheduledOptim(
        optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09),
        2.0, 256, warmup_steps)

    step = 0
    max_pitch_prec = 0
    
    ckpt_path = logdir / "ckpt" / "cur"
    if ckpt_path.exists():
        ckpt_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt_dict["model"])
        step = ckpt_dict["step"]
        max_pitch_prec = ckpt_dict["max_pitch_prec"]
        optimizer._optimizer.load_state_dict(ckpt_dict["optim"])
        optimizer.n_steps = step
    
    model.train()
    torch.autograd.set_detect_anomaly(True)
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

            if not scheduled_sampling:
                optimizer.zero_grad()
            result = model(mel, pitch_i, start_i, dur_i, start_t_i, end_i)

            if scheduled_sampling:
                if train_mode == "S":
                    pitch_p, start_p, dur_p = result
                    start_t_p = None
                    end_p = None
                    pitch_p = pitch_p.detach()
                    start_p = start_p.detach()
                    dur_p = dur_p.detach()
                elif train_mode == "T":
                    pitch_p, start_t_p, end_p = result
                    start_p = None
                    dur_p = None
                    pitch_p = pitch_p.detach()
                    start_t_p = start_t_p.detach()
                    end_p = end_p.detach()
                else:
                    pitch_p, start_t_p, end_p, start_p, dur_p = result
                    pitch_p = pitch_p.detach()
                    start_t_p = start_t_p.detach()
                    end_p = end_p.detach()
                    start_p = start_p.detach()
                    dur_p = dur_p.detach()

                optimizer.zero_grad()
                t = get_mix_t(step, mix_k, epsilon)
                result = model.forward_mix(mel, t,
                                           pitch_p, start_p, dur_p, start_t_p, end_p,
                                           pitch_i, start_i, dur_i, start_t_i, end_i)

            if train_mode == "S":
                pitch_p, start_p, dur_p = result
                start_t_p = None
                end_p = None
            elif train_mode == "T":
                pitch_p, start_t_p, end_p = result
                start_p = None
                dur_p = None
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
                start_t_loss = time_loss_alpha * masked_normal_nll(start_t_p, start_t_o, seq_mask)
                end_loss = time_loss_alpha * masked_normal_nll(end_p, end_o, seq_mask)
            
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
                # sw.add_scalar("training/mix_t", t, step)

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
                        # print(batch["fid"], batch["begin_time"], batch["end_time"])
                        fid = batch["fid"][0]
                        begin_time = batch["begin_time"][0]
                        end_time = batch["end_time"][0]

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

                        # result, (enc_attn, dec_self_attn, dec_enc_attn) = model(mel, pitch_i, start_i, dur_i, start_t_i, end_i, return_attns=True)
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

                            start_t_loss = time_loss_alpha * masked_normal_nll(start_t_p, start_t_o, seq_mask)
                            end_loss = time_loss_alpha * masked_normal_nll(end_p, end_o, seq_mask)
            
                        loss = pitch_loss + start_loss + dur_loss + start_t_loss + end_loss

                        if i < 1:
                            b = begin_time
                            e = end_time
                            if data_path.stem == "WebChorale":
                                # WebChorale
                                audio_path = Path("/storageNVME/huiran/WebChoraleDataset/OneSong")
                            else:
                                # BachChorale
                                audio_path = Path("/storageNVME/huiran/BachChorale/BachChorale")
                            audio_f = audio_path / ("%s.flac" % fid)
                            wav, sr = torchaudio.load(audio_f)
                            print(sr)
                            b = int(b * sr)
                            e = int(e * sr)
                            wav = wav.mean(dim=0)
                            wav = wav[b:e]
                            if len(wav) > 0:
                                sw.add_audio("%d" % i, wav, step, sr)
                                sw.add_text("info_%d" % i, "%s:%.3f-%.3f" % (fid, begin_time, end_time), step)
                            sw.add_figure("spec_%d" % i, plot_spec(mel[0].detach().cpu()), step)

                            # for a_i, attn in enumerate(enc_attn):
                            #     sw.add_figure("Attn/enc_%d" % a_i, plot_attn(attn[0].detach().cpu()), step)
                            # for a_i, attn in enumerate(dec_self_attn):
                            #     sw.add_figure("Attn/dec_self_%d" % a_i, plot_attn(attn[0].detach().cpu()), step)
                            # for a_i, attn in enumerate(dec_enc_attn):
                            #     sw.add_figure("Attn/dec_enc_%d" % a_i, plot_attn(attn[0].detach().cpu()), step)
                            if "S" in train_mode:
                                pred_list = get_list_s(pitch_p, start_p, dur_p)
                                gt_list = get_list_s(pitch_o, start_o, dur_o)
                                
                                pred_list = [(n, s, e) for n, s, e in zip(*pred_list)]
                                gt_list = [(n, s, e) for n, s, e in zip(*gt_list)]

                                sw.add_text("s/%d/gt" % i, str(gt_list), step)
                                sw.add_text("s/%d_pred" % i, str(pred_list), step)

                                sw.add_figure("gt/s_%d" % i, plot_score(pitch_o, start_o, dur_o), step)
                                sw.add_figure("pred/s_%d" % i, plot_score(pitch_p, start_p, dur_p), step)
                            if "T" in train_mode:
                                pred_list = get_list_t(pitch_p, start_t_p, end_p)
                                gt_list = get_list_t(pitch_o, start_t_o, end_o)

                                pred_list = [(n, s, e) for n, s, e in zip(*pred_list)]
                                gt_list = [(n, s, e) for n, s, e in zip(*gt_list)]

                                sw.add_text("t/%d/gt" % i, str(gt_list), step)
                                sw.add_text("t/%d/pred" % i, str(pred_list), step)
                                
                                sw.add_figure("gt/t_%d" % i, plot_midi(pitch_o, start_t_o, end_o, inc=True), step)
                                sw.add_figure("pred/t_%d" % i, plot_midi(pitch_p, start_t_p, end_p, inc=True), step)
                            # _ = input()

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

                checkpoint_path = logdir / "ckpt"
                checkpoint_path.mkdir(exist_ok=True)
                save_path = checkpoint_path / "cur"
                obj = {"model": model.state_dict(),
                       "optim": optimizer._optimizer.state_dict(),
                       "steps": step,
                       "epoch": e,
                       "max_pitch_prec": max_pitch_prec}
                torch.save(obj, str(save_path))

                if total_T / total_C > max_pitch_prec:
                    max_pitch_prec = total_T / total_C
                    save_path = checkpoint_path / "best"
                    torch.save(obj, str(save_path))

                model.train()
                
            step += 1 
