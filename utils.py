import matplotlib.pyplot as plt
import numpy as np


def plot_spec(spec):
    fig, ax = plt.subplots(figsize=(10, 3))
    im = ax.imshow(spec, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)

    fig.canvas.draw()
    plt.close()
    return fig


def plot_midi(pitch, start, end):
    if type(pitch) != list:
        pitch, start, end = get_list_t(pitch, start, end)

    fig, ax = plt.subplots(figsize=(10, 4))
    for n, s, e in zip(pitch, start, end):
        if s <= e and n > 0:
            ax.hlines(n, s, e, linewidths=3)
        if n == 0:
            ax.vlines(s, min(pitch), max(pitch), linestyles="--", linewidths=2)
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


def get_list_t(pitch, start, end):
    pitch = pitch.detach().cpu().numpy()[0]
    start = start.detach().cpu().numpy()[0]
    end = end.detach().cpu().numpy()[0]

    if len(pitch.shape) > 1:
        pitch = np.argmax(pitch, axis=1, keepdims=False)
        start = start.reshape(-1)
        end = end.reshape(-1)

    return pitch.tolist(), start.tolist(), end.tolist()
