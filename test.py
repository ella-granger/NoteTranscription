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

from mir_eval.multipitch import evaluate as evaluate_frames
from mir_eval.transcription import precision_recall_f1_overlap as evaluate_notes
from mir_eval.util import midi_to_hz
from scipy.stats import hmean
from collections import defaultdict

from copy import deepcopy

ex = Experiment("text_transcription")


def merge_notes(note_list):
    result = []
    note_list.sort(key=lambda x:x[0])

    prev_end = -1
    for n in note_list:
        if n[0] > prev_end:
            if prev_end != -1:
                result[-1][-1] = prev_end
            result.append(list(deepcopy(n)))
        prev_end = max(prev_end, n[1])
    result[-1][-1] = prev_end
    return result


def cal_mir_metrics(pitch, start_t, end, pitch_p, start_t_p, end_p):
    pitch = pitch.detach().cpu().numpy()[0]
    start_t = start_t.detach().cpu().numpy()[0]
    end = end.detach().cpu().numpy()[0]
    pitch_p = pitch_p.detach().cpu().numpy()[0]
    start_t_p = start_t_p.detach().cpu().numpy()[0]
    end_p = end_p.detach().cpu().numpy()[0]
    
    scaling = HOP_LENGTH / SAMPLE_RATE * SEG_LEN
    p_est = np.array([midi_to_hz(m + MIN_MIDI - 1) for m in pitch_p])
    p_ref = np.array([midi_to_hz(m + MIN_MIDI - 1) for m in pitch])
    i_est = np.array([(s * scaling, (s+d) * scaling) for (s, d) in zip(start_t_p, end_p)]).reshape(-1, 2)
    i_ref = np.array([(s * scaling, (s+d) * scaling) for (s, d) in zip(start_t, end)]).reshape(-1, 2)

    metrics = dict()
    p, r, f, o = evaluate_notes(i_ref, p_ref, i_est, p_est, offset_ratio=None)
    metrics['metric/note/precision'] = p
    metrics['metric/note/recall'] = r
    metrics['metric/note/f1'] = f
    metrics['metric/note/overlap'] = o

    p, r, f, o = evaluate_notes(i_ref, p_ref, i_est, p_est)
    metrics['metric/note-with-offsets/precision'] = p
    metrics['metric/note-with-offsets/recall'] = r
    metrics['metric/note-with-offsets/f1'] = f
    metrics['metric/note-with-offsets/overlap'] = o

    # frame_metrics = evaluate_frames(t_ref, f_ref, t_est, f_est)
    # metrics['metric/frame/f1'].append(hmean([frame_metrics['Precision'] + eps, frame_metrics['Recall'] + eps]) - eps)
    # for key, loss in frame_metrics.items():
    #     metrics['metric/frame/' + key.lower().replace(' ', '_')].append(loss)

    return metrics


def cal_metrics(pitch, start_t, end, pitch_p, start_t_p, end_p):
    pitch = pitch.detach().cpu().numpy()[0]
    start_t = start_t.detach().cpu().numpy()[0]
    end = end.detach().cpu().numpy()[0]
    pitch_p = pitch_p.detach().cpu().numpy()[0]
    start_t_p = start_t_p.detach().cpu().numpy()[0]
    end_p = end_p.detach().cpu().numpy()[0]

    trg_dict = {}
    prd_dict = {}

    for p, s, e in zip(pitch, start_t, end):
        if p == 0:
            continue
        if p not in trg_dict:
            trg_dict[p] = []
        trg_dict[p].append((s, s+e))

    for p, s, e in zip(pitch_p, start_t_p, end_p):
        if p == 0:
            continue
        if p not in prd_dict:
            prd_dict[p] = []
        prd_dict[p].append((s, s+e))

    print(trg_dict)
    print(prd_dict)

    frame_total = 0
    frame_true = 0
    frame_ori = 0
    onset_total = 0
    onset_true = 0
    onset_ori = 0
    offset_total = 0
    offset_true = 0
    offset_ori = 0
    threshold = 0.05 / 5.12

    for pitch in trg_dict:
        frame_ori += sum([(x[1] - x[0]) for x in trg_dict[pitch]])
        onset_ori += len(trg_dict[pitch])
        offset_ori += len(trg_dict[pitch])

        pred_notes = prd_dict.get(pitch, [])
        if len(pred_notes) == 0:
            continue
        onset_total += len(pred_notes)
        offset_total += len(pred_notes)
        trg_notes = trg_dict[pitch]

        start_p = np.array([x[0] for x in pred_notes])
        start_t = np.array([x[0] for x in trg_notes])
        end_p = np.array([x[1] for x in pred_notes])
        end_t = np.array([x[1] for x in trg_notes])

        start_t = np.repeat(np.expand_dims(start_t, 1), len(start_p), 1)
        end_t = np.repeat(np.expand_dims(end_t, 1), len(end_p), 1)

        start_overlap = np.abs(start_t - start_p) < threshold
        end_overlap = np.abs(end_t - end_p) < threshold
        onset_true += np.sum(np.any(start_overlap, axis=0))
        offset_true += np.sum(np.any(end_overlap, axis=0))

        pred_notes = merge_notes(pred_notes)
        trg_notes = merge_notes(trg_notes)
        # print("-----------------")
        # print(pred_notes)
        # print(trg_notes)
        frame_total += sum([(x[1] - x[0]) for x in pred_notes])

        i = 0
        for n in pred_notes:
            while trg_notes[i][1] < n[0]:
                i += 1
                if i == len(trg_notes):
                    break
            if i == len(trg_notes):
                break
            while trg_notes[i][0] <= n[1]:
                if trg_notes[i][0] < n[0]:
                    if trg_notes[i][1] <= n[1]:
                        # print(n[0], trg_notes[i][1])
                        frame_true += (trg_notes[i][1] - n[0])
                        i += 1
                    else:
                        # print(n[0], n[1])
                        frame_true += (n[1] - n[0])
                        break
                else:
                    if trg_notes[i][1] <= n[1]:
                        # print(trg_notes[i][0], trg_note[i][1])
                        frame_true += (trg_notes[i][1] - trg_notes[i][0])
                        i += 1
                    else:
                        # print(trg_notes[i][0], n[1])
                        frame_true += (n[1] - trg_notes[i][0])
                        break
                if i == len(trg_notes):
                    break
            if i == len(trg_notes):
                break
        

    for pitch in prd_dict:
        if pitch in trg_dict:
            continue
        pred_notes = prd_dict[pitch]
        onset_total += len(pred_notes)
        offset_total += len(pred_notes)

        pred_notes = merge_notes(pred_notes)
        frame_total += sum([(x[1] - x[0]) for x in pred_notes])

    print(frame_total, frame_true, frame_ori)
    print(onset_total, onset_true, onset_ori)
    print(offset_total, offset_true, offset_ori)
    # _ = input()

    return np.array((frame_total, frame_true, frame_ori)), np.array((onset_total, onset_true, onset_ori)), np.array((offset_total, offset_true, offset_ori))
    

@ex.config
def cfg():
    # ckpt_id = "00120000"
    ckpt_id = "best"
    # ckpt_id = "cur"
    mix_k = 0
    epsilon = 0


@ex.automain
def test(logdir, device, data_path, n_layers, ckpt_id, mix_k, epsilon,
        checkpoint_interval, batch_size, learning_rate, warmup_steps,
        clip_gradient_norm, epochs, output_interval, summary_interval,
         val_interval, loss_norm, time_loss_alpha, train_mode, enable_encoder,
         scheduled_sampling, prob_model):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logdir = Path(logdir)
    print_config(ex.current_run)

    data_path = "./dataset/test/BachChorale"

    data_path = Path(data_path)
    test_data = MelDataset(data_path / "mel",
                           data_path / "note",
                           data_path / "full.json",
                           train_mode,
                           device=device)
    print(len(test_data))

    model = NoteTransformer(kernel_size=9,
                            d_model=256,
                            d_inner=512,
                            n_layers=n_layers,
                            train_mode=train_mode,
                            enable_encoder=enable_encoder,
                            prob_model=prob_model).to(device)
    ckpt_path = logdir / "ckpt" / ckpt_id
    ckpt_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt_dict["model"])
    model = model.double()

    loader = DataLoader(test_data, 1, shuffle=False, drop_last=False)

    mets = defaultdict(list)
    model.eval()
    with torch.no_grad():
        f_c = 0
        on_c = 0
        off_c = 0
        for i, x in tqdm(enumerate(loader)):
            mel = x["mel"].to(device).double()
            pitch = x["pitch"]
            pitch = pitch[:, 1:-1]
            
            start = None
            dur = None
            start_t = None
            end = None
            if "S" in train_mode:
                start = x["start"].to(device)
                start = start[:, 1:-1]
                dur = x["dur"].to(device)
                dur = dur[:, 1:-1]
            if "T" in train_mode:
                start_t = x["start_t"].to(device)
                start_t = start_t[:, 1:-1]
                end = x["end"].to(device)
                end = end[:, 1:-1]

            fid = x["fid"][0]
            begin_time = x["begin_time"][0].item()
            end_time = x["end_time"][0].item()

            # print(pitch - 1 + MIN_MIDI)
            print(pitch)
            print(start_t)
            print(end)
            print(fid, begin_time, end_time)
            # _ = input()

            result = model.predict(mel, beam_size=4)

            if train_mode == "S":
                pitch_p, start_p, dur_p = result
            elif train_mode == "T":
                pitch_p, start_t_p, end_p = result
            else:
                pitch_p, start_t_p, end_p, start_p, dur_p = result

            # print("ori:")
            # print(pitch)
            # print(start_t)
            # print(end)
            # print("pred:")
            # print(pitch)
            # print(start_t)
            # print(pitch_p)
            # print(start_p)
            # print(dur_p)
            # print(start_t_p)
            # print(end_p)
            # print("------")
            # _ = input()

            frame, _, _ = cal_metrics(pitch, start_t, end, pitch_p, start_t_p, end_p)
            metrics = cal_mir_metrics(pitch, start_t, end, pitch_p, start_t_p, end_p)
            print(metrics)
            for k, v in metrics.items():
                mets[k].append(v)
            
            f_c += frame
            # on_c += onset
            # off_c += offset

            if i < 5:
                if "S" in train_mode:
                    pred_list = get_list_s(pitch_p, start_p, dur_p)
                    gt_list = get_list_s(pitch, start, dur)
                    fig_pred = plot_score(*pred_list)
                    fig_pred.savefig(logdir / ("pred_score_%d_%s_%.2f_%.2f.png" % (i, fid, begin_time, end_time)))
                    fig_gt = plot_score(*gt_list)
                    fig_gt.savefig(logdir / ("gt_score_%d_%s_%.2f_%.2f.png" % (i, fid, begin_time, end_time)))

                if "T" in train_mode:
                    pred_list = get_list_t(pitch_p, start_t_p, end_p)
                    gt_list = get_list_t(pitch, start_t, end)
                    
                    fig_pred = plot_midi(*pred_list, inc=True)
                    fig_pred.savefig(logdir / ("pred_trans_%d_%s_%.2f_%.2f.png" % (i, fid, begin_time, end_time)))
                    fig_gt = plot_midi(*gt_list, inc=True)
                    fig_gt.savefig(logdir / ("gt_trans_%d_%s_%.2f_%.2f.png" % (i, fid, begin_time, end_time)))
            # else:
            #     break

    print("Frame(prec/recall):", f_c[1] / f_c[0], f_c[1] / f_c[2])
    # print("Onset(prec/recall):", on_c[1] / on_c[0], on_c[1] / on_c[2])
    # print("Offset(prec/recall):", off_c[1] / off_c[0], off_c[1] / off_c[2])

    for k, v in mets.items():
        print(k, np.mean(v), np.std(v))
    
