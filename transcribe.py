import os
from sacred import Experiment
from sacred.commands import print_config
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchaudio
from tqdm import tqdm
import matplotlib.pyplot as plt
import pretty_midi

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

ex = Experiment("full_transcription")

@ex.config
def cfg():
    # ckpt_id = "00120000"
    ckpt_id = "best"
    # ckpt_id = "cur"
    mix_k = 0
    epsilon = 0
    seg_len = SEG_LEN
    # input_audio = "/media/ella/Yu/UR/datasets/BachChorale/audio/BC001.WAV"
    input_audio = "/media/ella/Yu/UR/datasets/BachChorale/audio/BC059.WAV"

    
@ex.automain
def test(logdir, device, data_path, n_layers, ckpt_id, mix_k, epsilon,
         checkpoint_interval, batch_size, learning_rate, warmup_steps,
         clip_gradient_norm, epochs, output_interval, summary_interval,
         val_interval, loss_norm, time_loss_alpha, train_mode, enable_encoder,
         scheduled_sampling, prob_model, seg_len, input_audio):
    # load model
    logdir = Path(logdir)
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
    print(ckpt_dict["epoch"], ckpt_dict["steps"])
    print([k for k in ckpt_dict])
    model = model.double()
    
    # convert data
    wave, sr = torchaudio.load(input_audio)
    wave_mono = wave.mean(dim=0)
    if sr != SAMPLE_RATE:
        trans = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
        wave_mono = trans(wave_mono)

    trans_mel = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE,
                                                         n_fft=N_FFT,
                                                         win_length=WIN_LENGTH,
                                                         hop_length=HOP_LENGTH,
                                                         pad_mode=PAD_MODE,
                                                         n_mels=N_MELS,
                                                         norm="slaney")
    mel_spec = trans_mel(wave_mono)
    mel_spec = torch.log(torch.clamp(mel_spec, min=MEL_EPSILON)) # (N_MELS, L)
    mel_spec = mel_spec[:, 1324:(1324+seg_len)]
    fig = plot_spec(mel_spec.numpy())
    fig.savefig("full_song.png")
    mel_spec = mel_spec.unsqueeze(0).double().to(device)

    with torch.no_grad():
        # transcribe seg by seg, use previous tail
        mel_hop = seg_len // 2
        begin_idx = 0
        end_idx = 0
        pref_s = 0
        pref_e = 0
        while end_idx < mel_spec.size(-1):
            end_idx = min(begin_idx + seg_len, mel_spec.size(-1))
            begin_idx = end_idx - seg_len
            # begin_time = begin_idx * HOP_LENGTH / SAMPLE_RATE
            begin_time = begin_idx / seg_len
            print(begin_time)
            mel = mel_spec[:, :, begin_idx:end_idx]
            if begin_idx == 0:
                new_pitch, new_start, new_dur = model.predict(mel)
                pitch = new_pitch[0]
                start = new_start[0]
                dur = new_dur[0]
                # print(pitch)
                # print(start)
                # print(dur)
                # _ = input()
            else:
                # print(pitch)
                # print(start)
                # print(dur)
                # print(begin_time)
                while start[pref_s] + dur[pref_s] < begin_time:
                    pref_s += 1
                pref_e = pref_s + 1
                if pref_e < len(start):
                    while start[pref_e] + dur[pref_e] < begin_time + 0.45:
                        pref_e += 1
                        if pref_e == len(start):
                            break
                # print(pref_s, pref_e)
                prev_pitch = pitch[pref_s:pref_e]
                prev_start = torch.clone(start[pref_s:pref_e])
                prev_start -= begin_time
                prev_dur = torch.clone(dur[pref_s:pref_e])
                prev_dur[prev_start < 0] += prev_start[prev_start < 0]
                prev_start[prev_start < 0] = 0
                # print(prev_pitch)
                # print(prev_start)
                # print(prev_dur)
                new_pitch, new_start, new_dur = model.predict(mel, prev_pitch, prev_start, prev_dur)
                # print(new_pitch)
                # print(new_start)
                # print(new_dur)
                new_pitch = new_pitch[0, len(prev_pitch):]
                new_start = new_start[0, len(prev_pitch):] + begin_time
                new_dur = new_dur[0, len(prev_pitch):]
                # print(new_pitch)
                # print(new_start)
                # print(new_dur)
                pitch = torch.cat([pitch[:pref_e], new_pitch]).detach()
                start = torch.cat([start[:pref_e], new_start]).detach()
                dur = torch.cat([dur[:pref_e], new_dur]).detach()
                # print(pitch)
                # print(start)
                # print(dur)
                # _ = input()


            begin_idx += mel_hop
        # aggregate and write result
        print(pitch)
        print(start)
        print(dur)

        pitch += (MIN_MIDI - 1)

        scale = seg_len * HOP_LENGTH / SAMPLE_RATE
        start = start * scale
        dur = dur * scale
        print(scale)

        pitch = pitch.detach().cpu().numpy()
        start = start.detach().cpu().numpy()
        dur = dur.detach().cpu().numpy()

        mf = pretty_midi.PrettyMIDI()
        prg = pretty_midi.Instrument(program=1)
        for n, s, d in zip(pitch, start, dur):
            note = pretty_midi.Note(velocity=100, pitch=n, start=s, end=s+d)
            prg.notes.append(note)
        mf.instruments.append(prg)
        mf.write("transcribed.mid")
        
