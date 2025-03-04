import matplotlib.pyplot as plt
import numpy as np


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
        if mode in ["gaussian", "l2", "l1", "diou", "gaussian-mu", "l1-diou"]:
            start = start[:, 0].reshape(-1)
            dur = dur[:, 0].reshape(-1)
        elif mode == "beta":
            start = beta_mode(start[:, 0], start[:, 1]).reshape(-1)
            dur = beta_mode(dur[:, 0], dur[:, 1]).reshape(-1)

    voice = (voice > 0.5)
    note_list = []
    for i in range(len(pitch)):
        for j in range(voice.shape[-1]):
            if voice[i, j]:
                note_list.append((pitch[i], start[i], start[i] + dur[i], j))
            if sum(voice[i]) == 0:
                note_list.append((pitch[i], start[i], start[i] + dur[i], 4))

    return note_list
