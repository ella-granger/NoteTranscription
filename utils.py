import matplotlib.pyplot as plt
import numpy as np
from dataset.constants import *
import pretty_midi


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


def save_midi(pitch, start, end, fname, ratio, inc=False):
    if type(pitch) != list:
        pitch, start, end = get_list_t(pitch, start, end)

    if inc:
        for i in range(len(start)):
            end[i] += start[i]

    output = pretty_midi.PrettyMIDI()
    track = pretty_midi.Instrument(program=1)
    for n, s, e in zip(pitch, start, end):
        p = n + MIN_MIDI - 1
        s = s * ratio
        e = e * ratio
        print(p, s, e)
        note = pretty_midi.Note(velocity=100, pitch=p, start=s, end=e)
        track.notes.append(note)

    output.instruments.append(track)
    output.write(str(fname))


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
            ax.hlines(n + MIN_MIDI - 1, s, e, linewidths=3)
        if n == 0:
            ax.vlines(s, min(pitch), max(pitch), linestyles="--", linewidths=2)

    pitch = np.array(pitch)
    start = np.array(start)
    end = np.array(end)
    ax.scatter(start, pitch + MIN_MIDI - 1)
    ax.scatter(end, pitch + MIN_MIDI - 1, marker="x")
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
    pitch = pitch.detach().cpu().numpy()[0]
    start = start.detach().cpu().numpy()[0]
    end = end.detach().cpu().numpy()[0]

    if len(pitch.shape) > 1:
        pitch = np.argmax(pitch, axis=1, keepdims=False)
        if mode in ["gaussian", "l2", "l1", "diou", "gaussian-mu", "l1-diou"]:
            start = start[:, 0].reshape(-1)
            end = end[:, 0].reshape(-1)
        elif mode == "beta":
            start = beta_mode(start[:, 0], start[:, 1]).reshape(-1)
            end = beta_mode(end[:, 0], end[:, 1]).reshape(-1)

    return pitch.tolist(), start.tolist(), end.tolist()
