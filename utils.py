import matplotlib.pyplot as plt
import numpy as np
import torch
from dataset.constants import *

from mir_eval.multipitch import evaluate as evaluate_frames
from mir_eval.transcription import precision_recall_f1_overlap as evaluate_notes
from mir_eval.util import midi_to_hz
from scipy.stats import hmean
from collections import defaultdict


def decode_notes(pitch, start, end, threshold=False):
    if len(pitch.size()) == 3:
        pitch = pitch[0]
        start = start[0]
        end = end[0]

    start = start[:, 0]
    end = end[:, 0]
    # print(pitch.size()) # (N, C)
    # print(start.size())
    # print(end.size())

    pitch_idx = torch.argmax(pitch, dim=-1)
    if threshold:
        pred_p = pitch[np.arange(pitch.size(0)), pitch_idx]
        valid = (pred_p > 0.85) * (pitch_idx != NUM_CLS)
    else:
        valid = (pitch_idx != NUM_CLS)

    pitch_v = pitch_idx[valid]
    start_v = start[valid]
    end_v = end[valid]

    # print(pitch_v.size())
    # print(start_v.size())
    # print(end_v.size())
    # _ = input()
    return pitch_v, start_v, end_v


def plot_mat(mat, row, col):
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(mat, aspect="auto", origin="upper", interpolation="none")
    plt.colorbar(im, ax=ax)

    plt.scatter(col, row, c="r")

    fig.canvas.draw()
    plt.close()
    return fig


def plot_attn(attn):
    K = attn.shape[0]
    n_cols = 2
    n_rows = K // 2
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(8,8))

    for i in range(K):
        axs[i // n_cols, i % n_cols].imshow(attn[i], origin="lower")

    plt.close()
    return fig


def plot_spec(spec):
    fig, ax = plt.subplots(figsize=(10, 3))
    im = ax.imshow(spec, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)

    fig.canvas.draw()
    plt.close()
    return fig


def plot_midi(pitch, start, end, inc=False):
    # print(pitch)
    # print(start)
    # print(end)
    # print(start, end)
    # _ = input()
    if type(pitch) != list:
        pitch, start, end = get_list_t(pitch, start, end)

    if inc:
        for i in range(len(start)):
            # if i > 0:
            #     start[i] += start[i-1]
            end[i] += start[i]

    # print(start, end)
    # _ = input()
    fig, ax = plt.subplots(figsize=(10, 4))
    for n, s, e in zip(pitch, start, end):
        if s <= e and n > 0:
            ax.hlines(n, s, e, linewidths=3)
        if n == 0:
            ax.vlines(s, min(pitch), max(pitch), linestyles="--", linewidths=2)

    ax.scatter(start, pitch)
    ax.scatter(end, pitch, marker="x")
    fig.canvas.draw()
    plt.close()
    return fig


def plot_score(pitch, start, dur):
    if type(pitch) != list:
        pitch, start, dur = get_list_s(pitch, start, dur)

    fig, ax = plt.subplots(figsize=(10, 4))
    cur_bar = 0
    for n, s, d in zip(pitch, start, dur):
        if n != 0:
            ax.hlines(n, s + cur_bar, s+d+cur_bar, linewidths=3)
        else:
            cur_bar += 16
            ax.vlines(cur_bar, min(pitch), max(pitch), linestyles="--", linewidths=2)

    fig.canvas.draw()
    plt.close()
    return fig


def get_list_s(pitch, start, end):
    pitch = pitch.detach().cpu().numpy()[0]
    start = start.detach().cpu().numpy()[0]
    end = end.detach().cpu().numpy()[0]

    if len(pitch.shape) > 1:
        pitch = np.argmax(pitch, axis=1, keepdims=False)
        start = np.argmax(start, axis=1, keepdims=False)
        end = np.argmax(end, axis=1, keepdims=False)
        # start = start.reshape(-1)
        # end = end.reshape(-1)

    return pitch.tolist(), start.tolist(), end.tolist()


def beta_mode(alpha, beta):
    result = (alpha - 1) / (alpha + beta - 2)
    alpha_mask = alpha < 1
    beta_mask = beta < 1

    zero_mask = alpha_mask * (1 - beta_mask)
    one_mask = beta_mask * (1 - alpha_mask)

    result[zero_mask.astype(bool)] = 0
    result[alpha_mask.astype(bool)] = 1

    return result


def get_list_t(pitch, start, end, mode="gaussian"):
    pitch = pitch.detach().cpu().numpy()
    start = start.detach().cpu().numpy()
    end = end.detach().cpu().numpy()
    if len(pitch.shape) == 2:
        pitch = pitch[0]
        start = start[0]
        end = end[0]

    if len(pitch.shape) > 1:
        pitch = np.argmax(pitch, axis=1, keepdims=False)
        if mode in ["gaussian", "l2", "l1", "diou", "gaussian-mu", "l1-diou"]:
            start = start[:, 0].reshape(-1)
            end = end[:, 0].reshape(-1)
        elif mode == "beta":
            start = beta_mode(start[:, 0], start[:, 1]).reshape(-1)
            end = beta_mode(end[:, 0], end[:, 1]).reshape(-1)

    return pitch.tolist(), start.tolist(), end.tolist()


def cal_mir_metrics(pitch, start_t, end, pitch_p, start_t_p, end_p, seg_len, tolerance=0.05):
    pitch = pitch.detach().cpu().numpy()
    start_t = start_t.detach().cpu().numpy()
    end = end.detach().cpu().numpy()
    pitch_p = pitch_p.detach().cpu().numpy()
    start_t_p = start_t_p.detach().cpu().numpy()
    end_p = end_p.detach().cpu().numpy()

    if len(pitch.shape) == 2:
        pitch = pitch[0]
        start_t = start_t[0]
        end = end[0]
        pitch_p = pitch_p[0]
        start_t_p = start_t_p[0]
        end_p = end_p[0]
    
    scaling = HOP_LENGTH / SAMPLE_RATE * seg_len
    p_est = np.array([midi_to_hz(m + MIN_MIDI - 1) for m in pitch_p])
    p_ref = np.array([midi_to_hz(m + MIN_MIDI - 1) for m in pitch])
    i_est = np.array([(s * scaling, (s+d) * scaling) for (s, d) in zip(start_t_p, end_p)]).reshape(-1, 2)
    i_ref = np.array([(s * scaling, (s+d) * scaling) for (s, d) in zip(start_t, end)]).reshape(-1, 2)

    metrics = dict()
    p, r, f, o = evaluate_notes(i_ref, p_ref, i_est, p_est, offset_ratio=None, onset_tolerance=tolerance)
    metrics['metric/note/precision'] = p
    metrics['metric/note/recall'] = r
    metrics['metric/note/f1'] = f
    metrics['metric/note/overlap'] = o

    p, r, f, o = evaluate_notes(i_ref, p_ref, i_est, p_est, onset_tolerance=tolerance)
    metrics['metric/note-with-offsets/precision'] = p
    metrics['metric/note-with-offsets/recall'] = r
    metrics['metric/note-with-offsets/f1'] = f
    metrics['metric/note-with-offsets/overlap'] = o

    # frame_metrics = evaluate_frames(t_ref, f_ref, t_est, f_est)
    # metrics['metric/frame/f1'].append(hmean([frame_metrics['Precision'] + eps, frame_metrics['Recall'] + eps]) - eps)
    # for key, loss in frame_metrics.items():
    #     metrics['metric/frame/' + key.lower().replace(' ', '_')].append(loss)

    return metrics
