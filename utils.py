import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform
from torch.distributions.transforms import SigmoidTransform, AffineTransform
from torch.distributions.transformed_distribution import TransformedDistribution

from dataset.constants import *

from mir_eval.transcription import precision_recall_f1_overlap as evaluate_notes
from mir_eval.util import midi_to_hz

def mask_out_seq(p, s, e, v=None):
    mask = (p < INI_IDX)
    scaling = HOP_LENGTH / SAMPLE_RATE * SEG_LEN

    p = p[mask]
    s = s[mask]
    e = e[mask]
    
    if v is not None:
        p_list = []
        itv_list = []
        for i, v_mask in enumerate(v):
            # print(v_mask)
            v_mask = v_mask[mask].astype(bool)
            # print(v_mask)
            if v_mask.sum() == 0:
                continue
            # print(p)
            print(p[v_mask])
            p_v = midi_to_hz(p[v_mask] + i * 0.25 + INI_IDX)
            s_v = s[v_mask] * scaling
            e_v = e[v_mask] * scaling
            # print(p_v.shape, s_v.shape, e_v.shape)
            itv = np.stack([s_v, e_v]).T
            valid_mask = (e_v > s_v)
            p_v = p_v[valid_mask]
            itv = itv[valid_mask]

            p_list.append(p_v)
            itv_list.append(itv)
            # print(p_v.shape)
            # print(itv.shape)
            # _ = input()
        p = np.concatenate(p_list)
        itv = np.concatenate(itv_list)

        print(p)
        print(itv)
        # _ = input()
        return p, itv

    p = midi_to_hz(p + INI_IDX)
    s = s * scaling
    e = e * scaling
    valid_mask = (e > s)
    itv = np.stack([s, e]).T

    p = p[valid_mask]
    itv = itv[valid_mask]
    
    print("MASKED SEQ")
    print(p)
    print(itv)

    return p, itv


def cal_reward(pred, gt):
    pitch_p, voice_p, start_p, end_p = pred
    pitch, voice, start, end = gt
    B = pitch_p.size(0)
    device = pitch_p.device

    r_list = []
    for i in range(B):
        p = pitch[i].detach().cpu().numpy().T
        v = voice[i].detach().cpu().numpy().T
        s = start[i].detach().cpu().numpy().T
        e = end[i].detach().cpu().numpy().T
        p_p = pitch_p[i].detach().cpu().numpy().T
        v_p = voice_p[i].detach().cpu().numpy().T
        s_p = start_p[i].detach().cpu().numpy().T[0]
        e_p = end_p[i].detach().cpu().numpy().T[0]

        p_m, i_m = mask_out_seq(p, s, e)
        p_p_m, i_p_m = mask_out_seq(p_p, s_p, e_p)

        _, _, o_f, _ = evaluate_notes(i_m, p_m, i_p_m, p_p_m, offset_ratio=None, pitch_tolerance=10.0, onset_tolerance=0.05)
        _, _, n_f, _ = evaluate_notes(i_m, p_m, i_p_m, p_p_m, offset_ratio=1.0, pitch_tolerance=10.0, onset_tolerance=0.05)

        p_m, i_m = mask_out_seq(p, s, e, v)
        p_p_m, i_p_m = mask_out_seq(p_p, s_p, e_p, v_p)

        _, _, vo_f, _ = evaluate_notes(i_m, p_m, i_p_m, p_p_m, offset_ratio=None, pitch_tolerance=10.0, onset_tolerance=0.05)
        _, _, vn_f, _ = evaluate_notes(i_m, p_m, i_p_m, p_p_m, offset_ratio=1.0, pitch_tolerance=10.0, onset_tolerance=0.05)

        r = o_f + n_f + vo_f + vn_f
        r_list.append(r)

    r = torch.Tensor(r_list).to(device)
    return r


def build_sigmoid_logistics(mu, sigma):
    device = mu.device
    uni = Uniform(torch.tensor(0.0).to(device), torch.tensor(1.0).to(device))
    trans = [SigmoidTransform().inv, AffineTransform(mu, sigma), SigmoidTransform()]
    return TransformedDistribution(uni, trans, validate_args=False)


def build_sigmoid_norm(mu, sigma):
    norm = Normal(mu, sigma)
    trans = [SigmoidTransform()]
    return TransformedDistribution(norm, trans, validate_args=False)


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


def plot_midi(note_list):
    # print(start, end)
    # _ = input()
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["red", "yellow", "green", "blue", "black"]
    for n, s, e, v in note_list:
        if s <= e:
            ax.hlines(n, s, e, colors[v], linewidths=3, alpha=0.4)

    pitch = [x[0] for x in note_list]
    start = [x[1] for x in note_list]
    end = [x[2] for x in note_list]
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


def get_list_t(pitch, start, dur, voice, mode="gaussian"):
    pitch = pitch.detach().cpu().numpy()[0]
    start = start.detach().cpu().numpy()[0]
    dur = dur.detach().cpu().numpy()[0]
    voice = voice.detach().cpu().numpy()[0]

    if len(pitch.shape) > 1:
        pitch = np.argmax(pitch, axis=1, keepdims=False)
        if mode in ["gaussian", "l2", "l1", "diou", "gaussian-mu", "l1-diou", "sig-log", "sig-norm"]:
            start = start[:, 0].reshape(-1)
            dur = dur[:, 0].reshape(-1)
        elif mode in ["beta"]:
            start = beta_mode(start[:, 0], start[:, 1]).reshape(-1)
            dur = beta_mode(dur[:, 0], dur[:, 1]).reshape(-1)

    voice = (voice > 0.5)
    note_list = []
    for i in range(len(pitch)):
        for j in range(voice.shape[-1]):
            if voice[i, j]:
                note_list.append((pitch[i], start[i], dur[i], j))
            if sum(voice[i]) == 0:
                note_list.append((pitch[i], start[i], dur[i], 4))

    return note_list
