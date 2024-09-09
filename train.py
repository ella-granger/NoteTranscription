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
from model import NoteTransformer, DETRLoss
from transformer.Optim import ScheduledOptim
from tqdm import tqdm
from dataset.constants import *
from utils import *

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
        # print("alpha")
        # print(alpha)
        # print("beta")
        # print(beta)
        tar = gt[i, :][mask[i]]
        # print("tar")
        # print(tar)
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


def masked_diou_loss(start_t_p, dur_p, start_t_g, dur_g, mask):
    batch_sum = 0
    for i in range(start_t_p.size(0)):
        l_p = start_t_p[i, :, 0][mask[i]]
        r_p = l_p + dur_p[i, :, 0][mask[i]]
        l_g = start_t_g[i, :][mask[i]]
        r_g = l_g + dur_g[i, :][mask[i]]

        c = torch.max(r_p, r_g) - torch.min(l_p, l_g)
        inter = torch.min(r_p, r_g) - torch.max(l_p, l_g)
        inter[inter < 0] = 0
        d = torch.abs((l_p + r_p) / 2 - (l_g + r_g) / 2)
        c[c < d] = d[c < d]
        c[c <= 0] = 1e-9

        b_diou = torch.sum(1 - inter / c + (d/c) ** 2)
        if torch.isnan(b_diou).any():
            print(i)
            print(mask[i])
            print(l_p)
            print(r_p)
            print(l_g)
            print(r_g)
            print(c)
            print(d)
            print(inter)
            print("--------------------ref----------------")

            i_ref = (i + 1) % (start_t_p.size(0))
            print(start_t_p[i_ref, :, 0][mask[i_ref]])
            print(start_t_g[i_ref, :][mask[i_ref]])
            raise RuntimeError("Found NaN!")

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
    train_mode = "TS"
    mix_k = 0.000001
    epsilon = 0.1
    loss_norm = 1
    time_lambda = 3
    enable_encoder = True
    scheduled_sampling = False
    num_samples = 800
    empty_weight = 0.01
    n_head = 4


@ex.automain
def train(logdir, device, n_layers, checkpoint_interval, batch_size,
          learning_rate, warmup_steps, mix_k, epsilon, empty_weight,
          clip_gradient_norm, epochs, data_path, num_samples, weight_dict,
          output_interval, summary_interval, val_interval, n_head,
          loss_norm, time_loss_alpha, train_mode, enable_encoder,
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
                            train_mode,
                            seg_len=seg_len,
                            device=device)

    valid_data = MelDataset(data_path / "mel",
                            data_path / "note",
                            data_path / "train.json", # "valid.json",
                            train_mode,
                            seg_len=seg_len,
                            device=device)

    train_loader = DataLoader(train_data, batch_size, shuffle=True, drop_last=False,
                              collate_fn=train_data.collate_fn, num_workers=8)
    eval_loader = DataLoader(valid_data, 1, shuffle=False, drop_last=False,
                             collate_fn=valid_data.collate_fn, num_workers=8)

    model = NoteTransformer(kernel_size=9,
                            d_model=256,
                            d_inner=512,
                            n_layers=n_layers,
                            n_head=n_head,
                            seg_len=seg_len,
                            train_mode=train_mode,
                            enable_encoder=enable_encoder,
                            prob_model=prob_model,
                            num_queries=num_samples)
    total_params = sum(p.numel() for p in model.parameters())
    print(total_params)
    # exit()
    model = model.to(device)

    num_classes = NUM_CLS
    loss_cal = DETRLoss(num_classes, weight_dict, empty_weight)

    optimizerGroup = getOptimizerGroup(model, 1e-2)
    optimizer = optim.AdaBelief(
        optimizerGroup,
        1e-5,
        weight_decouple=True,
        eps=1e-8,
        weight_decay=1e-2,
        rectify=True
    )

    lrScheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, learning_rate, 500000, pct_start=0.01, cycle_momentum=False, final_div_factor=2, div_factor = 20)
    
    # optimizer = ScheduledOptim(
    #     optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09),
    #     2.0, 256, warmup_steps) # weight decay: 0.01, without embedding

    step = 0
    # min_pitch_loss = np.inf
    max_note_f1 = 0
    
    ckpt_path = logdir / "ckpt" / "cur"
    if ckpt_path.exists():
        ckpt_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt_dict["model"])
        step = ckpt_dict["steps"]
        # max_pitch_prec = ckpt_dict["max_pitch_prec"]
        max_note_f1 = ckpt_dict["max_note_f1"]
        # optimizer._optimizer.load_state_dict(ckpt_dict["optim"])
        # optimizer.n_steps = step
        optimizer.load_state_dict(ckpt_dict["optim"])
        lrScheduler.load_state_dict(ckpt_dict["scheduler"])

        print(str(ckpt_path), "loaded.")
    
    model.train()
    torch.autograd.set_detect_anomaly(True)
    fin_flag = False
    for e in range(epochs):
        if fin_flag:
            break
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

            # pitch_i, pitch_o = patch_trg(pitch)
            # if "S" in train_mode:
            #     start_i, start_o = patch_trg(start)
            #     dur_i, dur_o = patch_trg(dur)
            # if "T" in train_mode:
            #     start_t_i, start_t_o = patch_trg(start_t)
            #     end_i, end_o = patch_trg(end)

            # print(start_t_o[0])
            # print(end_o[0])
            # print(pitch_o[0])
            # _ = input()

            if not scheduled_sampling:
                optimizer.zero_grad()
            result = model(mel)

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
            diou_loss = 0
            # pitch_loss = F.cross_entropy(torch.permute(pitch_p, (0, 2, 1)), pitch_o, ignore_index=PAD_IDX, reduction='sum')
            if "S" in train_mode:
                start_loss = F.cross_entropy(torch.permute(start_p, (0, 2, 1)), start_o, ignore_index=MAX_START+1, reduction='sum')
                dur_loss = F.cross_entropy(torch.permute(dur_p, (0, 2, 1)), dur_o, ignore_index=0, reduction='sum')
            if "T" in train_mode:
                # print(pitch_p.size())
                # print(start_t_p.size())
                # print(end_p.size())
                # print(pitch.size())
                # print(start_t.size())
                # print(end.size())
                # print(pitch)
                loss, losses, _ = loss_cal(pitch_p, start_t_p, end_p, pitch, start_t, end, x["length"])
                # print([x.item() for x in losses])
                # print(loss.item())
                
            loss.backward()
            optimizer.step()
            lrScheduler.step()

            # _ = input()

            if step % output_interval == 0:
                itr.set_description("loss: %.2f" % (loss.item()))

            if step % summary_interval == 0:
                sw.add_scalar("training/loss", loss.item(), step)
                """
                sw.add_scalar("training/pitch_loss", pitch_loss.item(), step)
                if "S" in train_mode:
                    sw.add_scalar("training/start_loss", start_loss.item(), step)
                    sw.add_scalar("training/dur_loss", dur_loss.item(), step)
                if "T" in train_mode:
                    if "diou" in prob_model:
                        sw.add_scalar("training/diou_loss", diou_loss.item(), step)
                    if "f1" in prob_model:
                        sw.add_scalar("training/start_t_loss", start_t_loss.item(), step)
                        sw.add_scalar("training/end_loss", end_loss.item(), step)
                        # if prob_model == "l1":
                        #     sw.add_scalar("training/start_diff_loss", start_diff_loss.item(), step)
                """
                sw.add_scalar("training/lr", optimizer.param_groups[0]["lr"], step)
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
                    total_diou_loss = 0
                    total_T = 0
                    total_start_T = 0
                    total_dur_T = 0
                    total_C = 0
                    total_count = 0
                    total_metrics = {}
                    for i, batch in tqdm(enumerate(eval_loader)):
                        if i == 500:
                            break
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

                        # pitch_i, pitch_o = patch_trg(pitch)
                        # if "S" in train_mode:
                        #     start_i, start_o = patch_trg(start)
                        #     dur_i, dur_o = patch_trg(dur)
                        # if "T" in train_mode:
                        #     start_t_i, start_t_o = patch_trg(start_t)
                        #     end_i, end_o = patch_trg(end)

                        # result, (enc_attn, dec_self_attn, dec_enc_attn) = model(mel, pitch_i, start_i, dur_i, start_t_i, end_i, return_attns=True)
                        # result = model(mel, pitch_i, start_i, dur_i, start_t_i, end_i, return_cnn=True)
                        result, (enc_attn, dec_self_attn, dec_enc_attn) = model(mel, return_cnn=True, return_attns=True)
                        mel_result = model.mel_result
                        enc_result = model.enc_result
                        if train_mode == "S":
                            pitch_p, start_p, dur_p = result
                        elif train_mode == "T":
                            pitch_p, start_t_p, end_p = result
                        else:
                            pitch_p, start_t_p, end_p, start_p, dur_p = result
                        
                        # pitch_loss = F.cross_entropy(torch.permute(pitch_p, (0, 2, 1)), pitch_o, ignore_index=PAD_IDX, reduction='sum')
                        if "S" in train_mode:
                            start_loss = F.cross_entropy(torch.permute(start_p, (0, 2, 1)), start_o, ignore_index=MAX_START+1, reduction='sum')
                            dur_loss = F.cross_entropy(torch.permute(dur_p, (0, 2, 1)), dur_o, ignore_index=0, reduction='sum')
                        if "T" in train_mode:
                            # print(pitch_p.size())
                            # _ = input()
                            loss, losses, out_mat = loss_cal(pitch_p, start_t_p, end_p, pitch, start_t, end, batch["length"])

                        mat, row, col = out_mat
            
                        total_loss += loss.item()
                        
                        note_p, note_s, note_e = decode_notes(pitch_p, start_t_p, end_p)
                        # print(note_p.size(), note_s.size(), note_e.size(), pitch.size(), start_t.size(), end.size())
                        try:
                            metrics = cal_mir_metrics(pitch[0], start_t[0], end[0], note_p, note_s, note_e, seg_len)
                            for k, v in metrics.items():
                                if k not in total_metrics:
                                    total_metrics[k] = 0
                                total_metrics[k] += v
                        except Exception as err:
                            print(err)
                        total_count += 1

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

                            # Bipartite
                            sw.add_figure("Bipartite/total", plot_mat(mat, row, col), step)
                            sw.add_figure("LossCal/pitch", plot_mat(pitch_p.detach().cpu().numpy()[0], row, pitch[0].detach().cpu()[col]), step)
                            for k, v in loss_cal.loss_dict_run.items():
                                sw.add_figure("Bipartite/%s" % k, plot_mat(v.detach().cpu().numpy(), row, col), step)

                            for a_i, attn in enumerate(enc_attn):
                                sw.add_figure("Attn/enc_%d" % a_i, plot_attn(attn[0].detach().cpu()), step)
                            for a_i, attn in enumerate(dec_self_attn):
                                sw.add_figure("Attn/dec_self_%d" % a_i, plot_attn(attn[0].detach().cpu()), step)
                            for a_i, attn in enumerate(dec_enc_attn):
                                sw.add_figure("Attn/dec_enc_%d" % a_i, plot_attn(attn[0].detach().cpu()), step)
                            if "S" in train_mode:
                                pred_list = get_list_s(pitch_p, start_p, dur_p)
                                gt_list = get_list_s(pitch, start, dur)
                                
                                pred_list = [(n, s, e) for n, s, e in zip(*pred_list)]
                                gt_list = [(n, s, e) for n, s, e in zip(*gt_list)]

                                sw.add_text("s/%d/gt" % i, str(gt_list), step)
                                sw.add_text("s/%d_pred" % i, str(pred_list), step)

                                sw.add_figure("gt/s_%d" % i, plot_score(pitch_o, start_o, dur_o), step)
                                sw.add_figure("pred/s_%d" % i, plot_score(pitch_p, start_p, dur_p), step)
                            if "T" in train_mode:
                                pred_list = get_list_t(note_p, note_s, note_e, mode=prob_model)
                                gt_list = get_list_t(pitch, start_t, end, mode=prob_model)

                                # print(pred_list)
                                # print(gt_list)

                                pred_list = [(n, s, e) for n, s, e in zip(*pred_list)]
                                gt_list = [(n, s, e) for n, s, e in zip(*gt_list)]

                                sw.add_text("t/%d/gt" % i, str(gt_list), step)
                                sw.add_text("t/%d/pred" % i, str(pred_list), step)
                                
                                sw.add_figure("gt/t_%d" % i, plot_midi(pitch, start_t, end, inc=False), step)
                                sw.add_figure("pred/t_%d" % i, plot_midi([x[0] for x in pred_list], [x[1] for x in pred_list], [x[2] for x in pred_list], inc=False), step)
                            # _ = input()


                for k in total_metrics:
                    total_metrics[k] /= total_count
                    sw.add_scalar(k, total_metrics[k], step)
                eval_note_f1 = total_metrics["metric/note/f1"]
                eval_loss = total_loss / total_count
                
                sw.add_scalar("eval/loss", eval_loss, step)
                
                print(eval_loss, eval_note_f1)

                checkpoint_path = logdir / "ckpt"
                checkpoint_path.mkdir(exist_ok=True)
                save_path = checkpoint_path / "cur" # ("cur_%05d" % step)
                obj = {"model": model.state_dict(),
                       "optim": optimizer.state_dict(),
                       "scheduler": lrScheduler.state_dict(),
                       "steps": step,
                       "epoch": e,
                       "max_note_f1": max_note_f1}
                
                if step % 100 == 0:
                    torch.save(obj, str(save_path))

                # if total_T / total_C > max_pitch_prec:
                if eval_note_f1 > max_note_f1:
                    # max_pitch_prec = total_T / total_C
                    max_note_f1 = eval_note_f1
                    save_path = checkpoint_path / "best"
                    torch.save(obj, str(save_path))

                model.train()
                
            step += 1
            # if step == 10001:
            #     fin_flag = True
            #     break
