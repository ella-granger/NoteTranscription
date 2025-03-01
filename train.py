import os
from sacred import Experiment
from sacred.commands import print_config, save_config
from sacred.observers import FileStorageObserver
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import optim
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
from model import NoteTransformer
from transformer.Optim import ScheduledOptim
from tqdm import tqdm
from dataset.constants import *
from utils import *
import torch.multiprocessing as mp
mp.set_start_method("fork", force=True)

ex = Experiment("train_transcriber")

def patch_trg(trg):
    gold = trg[:, 1:].contiguous()
    trg = trg[:, :-1]
    return trg, gold


def masked_beta_nll(pred, gt, mask):
    batch_sum = 0
    epsilon = 1e-6
    for i in range(gt.size(0)):
        alpha = pred[i, :, 0][mask[i]]
        beta = pred[i, :, 1][mask[i]]
        tar = gt[i, :][mask[i]]
        tar[tar == 0] = epsilon
        tar[tar == 1] = 1 - epsilon
        
        dist = Beta(alpha, beta)
        lp = dist.log_prob(tar)
        batch_sum -= lp.sum()

    return batch_sum


def masked_normal_nll(pred, gt, mask, fix_var=None):
    batch_sum = 0
    for i in range(gt.size(0)):
        mu = pred[i, :, 0][mask[i]]
        if fix_var is None:
            var = pred[i, :, 1][mask[i]]
        else:
            var = torch.ones_like(mu).to(mu.device) * fix_var
        tar = gt[i, :][mask[i]]
        batch = F.gaussian_nll_loss(mu, tar, var, reduction="sum")
        batch_sum += batch

    return batch_sum


def masked_diou_loss(start_p, dur_p, start_g, dur_g, mask):
    batch_sum = 0
    for i in range(start_p.size(0)):
        l_p = start_p[i, :, 0][mask[i]]
        r_p = l_p + dur_p[i, :, 0][mask[i]]
        l_g = start_g[i, :][mask[i]]
        r_g = l_g + dur_g[i, :][mask[i]]

        c = torch.max(r_p, r_g) - torch.min(l_p, l_g)
        inter = torch.min(r_p, r_g) - torch.max(l_p, l_g)
        inter[inter < 0] = 0
        d = torch.abs((l_p + r_p) / 2 - (l_g + r_g) / 2)
        c[c < d] = d[c < d]
        c[c <= 0] = 1e-9

        b_diou = torch.sum(1 - inter / c + (d/c) ** 2)
        batch_sum += b_diou 
        
    return batch_sum


def masked_l2_loss(pred, gt, mask):
    diff = pred - gt.unsqueeze(-1)
    loss = torch.sum(diff[mask.unsqueeze(-1)] ** 2)
    return loss

def masked_l1_loss(pred, gt, mask):
    diff = pred - gt.unsqueeze(-1)
    loss = torch.sum(torch.abs(diff[mask.unsqueeze(-1)]))
    return loss


def get_time_loss(prob_model):
    if prob_model == "beta":
        return masked_beta_nll
    elif prob_model == "gaussian":
        return lambda p, o, m : masked_normal_nll(p, o, m)
    elif prob_model == "guassian-mu":
        return lambda p, o, m : masked_normal_nll(p, o, m, 0.05)
    elif prob_model == "l1":
        return masked_l1_loss
    elif prob_model == "l2":
        return masked_l2_loss
    elif prob_model == "diou":
        return lambda p, o, m : 0
    elif prob_model == "l1-diou":
        return masked_l1_loss


def get_mix_t(step, k, epsilon):
    # return 1
    if step < 100000:
        t = 1
    else:
        t = max(epsilon, 1 - k * (step - 100000))
    return t


def getOptimizerGroup(model, weight_decay):
    param_optimizer = list(model.named_parameters())
    # exclude GroupNorm and PositionEmbedding from weight decay

    noDecay = []
    for name, module in model.named_modules():
        if isinstance(module, nn.GroupNorm) \
                or isinstance(module, nn.LayerNorm) \
                or isinstance(module, nn.Embedding):
            noDecay.extend(list(module.parameters()))
        else:
            noDecay.extend([p for n, p in module.named_parameters() if "bias" in n])
    
    otherParams = set(model.parameters()) - set(noDecay)
    otherParams = [param for param in model.parameters() if param in otherParams]
    noDecay = set(noDecay)
    noDecay = [param for param in model.parameters() if param in noDecay]


    optimizerConfig = [{"params": otherParams, "weight_decay":weight_decay},
                        {"params": noDecay, "weight_decay":0e-7}]

    return optimizerConfig


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


@ex.automain
def train(logdir, device, n_layers, checkpoint_interval, batch_size,
          learning_rate, warmup_steps, mix_k, epsilon,
          clip_gradient_norm, epochs, data_path,
          output_interval, summary_interval, val_interval,
          loss_norm, time_loss_alpha, enable_encoder,
          scheduled_sampling, prob_model, seg_len, time_lambda):
    ex.observers.append(FileStorageObserver.create(logdir))
    sw = SummaryWriter(logdir)

    logdir = Path(logdir)
    print_config(ex.current_run)
    save_config(ex.current_run.config, logdir / "config.json")

    data_path = Path(data_path)
    train_data = MelDataset(data_path / "mel",
                            data_path / "note",
                            data_path / "train.json",
                            seg_len=seg_len,
                            device=device)

    valid_data = MelDataset(data_path / "mel",
                            data_path / "note",
                            data_path / "valid.json",
                            seg_len=seg_len,
                            device=device)

    train_loader = DataLoader(train_data, batch_size, shuffle=True, drop_last=False,
                              collate_fn=train_data.collate_fn, num_workers=8,
                              persistent_workers=True, prefetch_factor=4, pin_memory=True)
    eval_loader = DataLoader(valid_data, 1, shuffle=False, drop_last=False,
                             collate_fn=valid_data.collate_fn, num_workers=8, pin_memory=True)

    model = NoteTransformer(kernel_size=9,
                            d_model=256,
                            d_inner=512,
                            n_layers=n_layers,
                            seg_len=seg_len,
                            enable_encoder=enable_encoder,
                            prob_model=prob_model)
    total_params = sum(p.numel() for p in model.parameters())
    print(total_params)
    # exit()
    model = model.to(device)

    optimizerGroup = getOptimizerGroup(model, 1e-2)
    optimizer = optim.AdaBelief(
        optimizerGroup,
        1e-5,
        weight_decouple=True,
        eps=1e-8,
        weight_decay=1e-2,
        rectify=True
    )

    lrScheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 4e-4, 500000, pct_start=0.01, cycle_momentum=False, final_div_factor=2, div_factor = 20)

    step = 0
    max_pitch_prec = 0
    min_pitch_loss = np.inf
    
    ckpt_path = logdir / "ckpt" / "cur"
    if ckpt_path.exists():
        ckpt_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt_dict["model"])
        step = ckpt_dict["steps"]
        max_pitch_prec = ckpt_dict["max_pitch_prec"]
        optimizer.load_state_dict(ckpt_dict["optim"])
        lrScheduler.load_state_dict(ckpt_dict["scheduler"])

    time_loss = get_time_loss(prob_model)
    
    model.train()
    # torch.autograd.set_detect_anomaly(True)
    # epochs = 3
    for e in range(epochs):
        # itr = tqdm(train_loader)
        for x in tqdm(train_loader):
            mel = x["mel"].to(device)
            pitch = x["pitch"].to(device)
            start = x["start"].to(device)
            dur = x["dur"].to(device)

            pitch_i, pitch_o = patch_trg(pitch)
            start_i, start_o = patch_trg(start)
            dur_i, dur_o = patch_trg(dur)

            optimizer.zero_grad()
            result = model(mel, pitch_i, start_i, dur_i)
            pitch_p, start_p, dur_p = result
          
            pitch_loss = F.cross_entropy(torch.permute(pitch_p, (0, 2, 1)), pitch_o, ignore_index=PAD_IDX, reduction='sum')
            seq_mask = (pitch_o != PAD_IDX) * (pitch_o != 0)
            start_loss = time_loss(start_p, start_o, seq_mask)
            dur_loss = time_loss(dur_p, dur_o, seq_mask)

            diou_loss = 0  
            if "diou" in prob_model:
                diou_loss = masked_diou_loss(start_p, dur_p, start_o, dur_o, seq_mask) # diou loss

            loss = pitch_loss + time_lambda * (start_loss + dur_loss + diou_loss)

            loss.backward()
            optimizer.step()
            lrScheduler.step()

            # if step % output_interval == 0:
            #     itr.set_description("pitch: %.2f" % (pitch_loss.item()))

            if step % summary_interval == 0:
                sw.add_scalar("training/loss", loss.item(), step)
                sw.add_scalar("training/pitch_loss", pitch_loss.item(), step)
                if "diou" in prob_model:
                    sw.add_scalar("training/diou_loss", diou_loss.item(), step)
                if "f1" in prob_model:
                    sw.add_scalar("training/start_loss", start_loss.item(), step)
                    sw.add_scalar("training/dur_loss", dur_loss.item(), step)
                sw.add_scalar("training/lr", optimizer.param_groups[0]["lr"], step)

            if step % val_interval == 0:
                model.eval()
                with torch.no_grad():
                    total_loss = 0
                    total_pitch_loss = 0
                    total_start_loss = 0
                    total_dur_loss = 0
                    total_diou_loss = 0
                    total_T = 0
                    total_start_T = 0
                    total_dur_T = 0
                    total_C = 0
                    total_count = 0
                    for i, batch in tqdm(enumerate(eval_loader)):
                        mel = batch["mel"].to(device)
                        pitch = batch["pitch"].to(device)
                        start = batch["start"].to(device)
                        dur = batch["dur"].to(device)
                        
                        fid = batch["fid"][0]
                        begin_time = batch["begin_time"][0]
                        end_time = batch["end_time"][0]

                        pitch_i, pitch_o = patch_trg(pitch)
                        start_i, start_o = patch_trg(start)
                        dur_i, dur_o = patch_trg(dur)

                        # result, (enc_attn, dec_self_attn, dec_enc_attn) = model(mel, pitch_i, start_i, dur_i, start_t_i, end_i, return_attns=True)
                        result = model(mel, pitch_i, start_i, dur_i, return_cnn=True)
                        mel_result = model.mel_result
                        enc_result = model.enc_result
                        pitch_p, start_p, dur_p = result

                        start_loss = 0
                        dur_loss = 0
                        diou_loss = 0
                        pitch_loss = F.cross_entropy(torch.permute(pitch_p, (0, 2, 1)), pitch_o, ignore_index=PAD_IDX, reduction='sum')
                        seq_mask = (pitch_o != PAD_IDX) * (pitch_o != 0)
                        start_loss = time_loss(start_p, start_o, seq_mask)
                        dur_loss = time_loss(dur_p, dur_o, seq_mask)

                        if "diou" in prob_model:
                            diou_loss = masked_diou_loss(start_p, dur_p, start_o, dur_o, seq_mask) # diou loss

                        loss = pitch_loss + time_lambda * (start_loss + dur_loss + diou_loss)

                        if i < 1:
                            b = begin_time
                            e = end_time
                            if data_path.stem == "YouChorale":
                                # WebChorale
                                audio_path = Path("./dataset/YouChorale/audio_clean")
                            else:
                                # BachChorale
                                audio_path = Path("/storageNVME/huiran/BachChorale/BachChorale")
                            audio_f = audio_path / ("%s.mp4" % fid)
                            wav, sr = torchaudio.load(audio_f)
                            b = int(b * sr)
                            e = int(e * sr)
                            wav = wav.mean(dim=0)
                            wav = wav[b:e]
                            if len(wav) > 0:
                                sw.add_audio("%d" % i, wav, step, sr)
                                sw.add_text("info_%d" % i, "%s:%.3f-%.3f" % (fid, begin_time, end_time), step)
                            sw.add_figure("spec_%d" % i, plot_spec(mel[0].detach().cpu()), step)
                            sw.add_figure("cnn_%d" % i, plot_spec(mel_result[0].detach().cpu()), step)
                            sw.add_figure("enc_%d" % i, plot_spec(enc_result[0].detach().cpu()), step)


                            # for a_i, attn in enumerate(enc_attn):
                            #     sw.add_figure("Attn/enc_%d" % a_i, plot_attn(attn[0].detach().cpu()), step)
                            # for a_i, attn in enumerate(dec_self_attn):
                            #     sw.add_figure("Attn/dec_self_%d" % a_i, plot_attn(attn[0].detach().cpu()), step)
                            # for a_i, attn in enumerate(dec_enc_attn):
                            #     sw.add_figure("Attn/dec_enc_%d" % a_i, plot_attn(attn[0].detach().cpu()), step)
                            pred_list = get_list_t(pitch_p, start_p, dur_p, mode=prob_model)
                            gt_list = get_list_t(pitch_o, start_o, dur_o, mode=prob_model)

                            pred_list = [(n, s, e) for n, s, e in zip(*pred_list)]
                            gt_list = [(n, s, e) for n, s, e in zip(*gt_list)]

                            sw.add_text("t/%d/gt" % i, str(gt_list), step)
                            sw.add_text("t/%d/pred" % i, str(pred_list), step)

                            sw.add_figure("gt/t_%d" % i, plot_midi(pitch_o, start_o, dur_o, inc=True), step)
                            sw.add_figure("pred/t_%d" % i, plot_midi([x[0] for x in pred_list], [x[1] for x in pred_list], [x[2] for x in pred_list], inc=True), step)
                            # _ = input()

                        pitch_pred = torch.argmax(pitch_p, dim=-1)
                        total_T += torch.sum(pitch_pred == pitch_o).item()
                        total_C += pitch_pred.size(1)
                        total_loss += loss.item()
                        total_pitch_loss += pitch_loss.item()
                        if "l1" in prob_model:
                            total_start_loss += start_loss.item()
                            total_dur_loss += dur_loss.item()
                        if "diou" in prob_model:
                            total_diou_loss += diou_loss.item()
                        total_count += 1

                eval_loss = total_loss / total_count
                eval_pitch_loss = total_pitch_loss / total_count
                sw.add_scalar("eval/loss", eval_loss, step)
                sw.add_scalar("eval/pitch_loss", eval_pitch_loss, step)
                if "diou" in prob_model:
                    eval_diou_loss = total_diou_loss / total_count
                    sw.add_scalar("eval/diou_loss", eval_diou_loss, step)
                if "l1" in prob_model:
                    eval_start_loss = total_start_loss / total_count
                    eval_dur_loss = total_dur_loss / total_count
                    sw.add_scalar("eval/start_loss", eval_start_loss, step)
                    sw.add_scalar("eval/dur_loss", eval_dur_loss, step)
                sw.add_scalar("eval/pitch_prec", total_T / total_C, step)
                print(eval_loss, total_T / total_C)

                checkpoint_path = logdir / "ckpt"
                checkpoint_path.mkdir(exist_ok=True)
                save_path = checkpoint_path / "cur"
                obj = {"model": model.state_dict(),
                       "optim": optimizer.state_dict(),
                       "scheduler": lrScheduler.state_dict(),
                       "steps": step,
                       "epoch": e,
                       "max_pitch_prec": max_pitch_prec}
                torch.save(obj, str(save_path))

                if total_T / total_C > max_pitch_prec:
                # if eval_pitch_loss < min_pitch_loss:
                    # max_pitch_prec = total_T / total_C
                    min_pitch_loss = eval_pitch_loss
                    save_path = checkpoint_path / "best"
                    torch.save(obj, str(save_path))

                model.train()
                
            step += 1 
