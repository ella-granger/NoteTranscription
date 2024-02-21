import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformer.Models import Encoder, Decoder, get_pad_mask, get_subsequent_mask
from dataset.constants import *


def get_mix_mask(pitch, t):
    # reach lowest point at 30k
    U = torch.rand((pitch.size(0), pitch.size(1)))
    U = (U > t)
    U[:, 0] = False
    # print(U)
    # _ = input()
    return U


class TimeEncoding(nn.Module):

    def __init__(self, d_hid):
        super(TimeEncoding, self).__init__()

        self.d_hid = d_hid


    def forward(self, x):
        # x: (B, L, 1) ~ (0, 1)
        enc = 200 * x
        enc = enc.repeat(1, 1, self.d_hid)

        for j in range(self.d_hid):
            div = np.power(10000, 2 * (j // 2) / self.d_hid)
            enc[:, :, j] = enc[:, :, j] / div

        enc[:, :, 0::2] = torch.sin(enc[:, :, 0::2])
        enc[:, :, 1::2] = torch.cos(enc[:, :, 1::2])

        return enc


class NoteTransformer(nn.Module):

    def __init__(self, kernel_size, d_model, d_inner, n_layers, train_mode, alpha=10):
        super(NoteTransformer, self).__init__()

        self.alpha = alpha

        # ConvNet
        padding_len = kernel_size // 2
        self.cnn = nn.Sequential(
            nn.Conv1d(N_MELS, d_model, kernel_size, padding=padding_len),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size, padding=padding_len),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.d_model = d_model
        # Encoder
        self.encoder = Encoder(d_word_vec=d_model,
                               n_layers=n_layers,
                               n_head=N_HEAD,
                               d_model=d_model,
                               d_inner=d_inner,
                               n_position=SEG_LEN,
                               scale_emb=True)

        # Decoder
        self.trg_pitch_emb = nn.Embedding(PAD_IDX+1, d_model // 2, padding_idx=PAD_IDX)
        if "S" in train_mode:
            self.trg_start_emb = nn.Embedding(MAX_START+2, d_model // 4 // len(train_mode), padding_idx=MAX_START+1)
            self.trg_dur_emb = nn.Embedding(MAX_DUR+1, d_model // 4 // len(train_mode), padding_idx=0)
        if "T" in train_mode:
            # self.time_enc = TimeEncoding(d_model // 8)
            self.start_prj = nn.Linear(1, d_model // 4 // len(train_mode))
            self.end_prj = nn.Linear(1, d_model // 4 // len(train_mode))
        
        self.decoder = Decoder(d_word_vec=d_model,
                               n_layers=n_layers,
                               n_head=N_HEAD,
                               d_model=d_model,
                               d_inner=d_inner)

        # Result
        # self.trg_pitch_prj = nn.Linear(d_model * 3 // 4, PAD_IDX+1)
        # self.trg_start_prj = nn.Linear(d_model // 8, 1)
        # self.trg_end_prj = nn.Linear(d_model // 8, 1)
        self.trg_pitch_prj = nn.Linear(d_model, PAD_IDX+1)
        if "S" in train_mode:
            self.trg_start_s_prj = nn.Linear(d_model, MAX_START+2)
            self.trg_dur_s_prj = nn.Linear(d_model, MAX_DUR+1)
        if "T" in train_mode:
            self.trg_start_t_prj = nn.Linear(d_model, 1)
            self.trg_end_prj = nn.Linear(d_model, 1)

        self.train_mode = train_mode

    def forward(self, mel, pitch, start_s, dur, start_t, end):
        # mel feature extraction
        mel = self.cnn(mel)
        mel = torch.permute(mel, (0, 2, 1))

        # encoding
        enc, *_ = self.encoder(mel)

        # decoding
        trg_mask = get_pad_mask(pitch, PAD_IDX) & get_subsequent_mask(pitch)

        if "T" in self.train_mode:
            start_t = torch.unsqueeze(start_t, -1)
            end = torch.unsqueeze(end, -1)
            start_t = self.start_prj(start_t)
            end = self.end_prj(end)
            # start = self.time_enc(start)
            # end = self.time_enc(end)
        pitch = self.trg_pitch_emb(pitch)
        if "S" in self.train_mode:
            start_s = self.trg_start_emb(start_s)
            dur = self.trg_dur_emb(dur)

        if self.train_mode == "T":
            trg_seq = torch.cat([pitch, start_t, end], dim=-1)
        elif self.train_mode == "S":
            trg_seq = torch.cat([pitch, start_s, dur], dim=-1)
        else:
            # print(pitch.size())
            # print(start_t.size())
            # print(end.size())
            # print(start_s.size())
            # print(dur.size())
            trg_seq = torch.cat([pitch, start_t, end, start_s, dur], dim=-1)
        
        dec, *_ = self.decoder(trg_seq, trg_mask, enc)

        pitch_out = self.trg_pitch_prj(dec)

        if "S" in self.train_mode:
            start_s_out = self.trg_start_s_prj(dec)
            dur_out = self.trg_dur_s_prj(dec)
        if "T" in self.train_mode:
            start_t_out = self.trg_start_t_prj(dec)
            start_t_out = F.sigmoid(start_t_out)
            end_out = self.trg_end_prj(dec)
            end_out = F.sigmoid(end_out)
        
        # pitch_out = self.trg_pitch_prj(dec[:, :, :(self.d_model * 3 // 4)])
        # start_out = self.trg_start_prj(dec[:, :, (-self.d_model // 4):(-self.d_model // 8)])
        # start_out = F.sigmoid(start_out)
        # end_out = self.trg_end_prj(dec[:, :, (-self.d_model // 8):])
        # end_out = F.sigmoid(end_out)

        if self.train_mode == "T":
            return pitch_out, start_t_out, end_out
        elif self.train_mode == "S":
            return pitch_out, start_s_out, dur_out
        else:
            return pitch_out, start_t_out, end_out, start_s_out, dur_out
        
        # return pitch_out, start_out, end_out


    def get_mix_emb(self, p, i, emb):
        # p
        e = emb.weight #(idx_num, emb_dim)
        # print(p.mean())
        # print(p.std())
        G_y = torch.rand(e.size(0))
        # print(G_y)
        G_y = -torch.log(-torch.log(G_y))
        # print(G_y)

        device = p.device
        G_y = G_y.to(device)

        # print(p.size())
        # print(G_y.size())
        s = self.alpha * p + G_y
        # print(s.mean())
        # print(s.std())
        # print(s[0, 3, :])
        w = F.softmax(s, dim=-1)
        # print(w[0, 3, :])

        # print(w.size())
        # print(e.size())
        e = torch.matmul(w, e)
        # print(e.size())
        # _ = input()
        return e


    def forward_mix(self, mel, t,
                    pitch_p, start_p, dur_p, start_t_p, end_p,
                    pitch_i, start_i, dur_i, start_t_i, end_i):
        mel = self.cnn(mel)
        mel = torch.permute(mel, (0, 2, 1))

        enc, *_ = self.encoder(mel)

        trg_mask = get_pad_mask(pitch_i, PAD_IDX) & get_subsequent_mask(pitch_i)

        mix_mask = get_mix_mask(pitch_i, t)

        pitch = self.get_mix_emb(pitch_p, pitch_i, self.trg_pitch_emb)
        pitch_i = self.trg_pitch_emb(pitch_i)
        # print(pitch_i.size())
        # print(pitch.size())
        # print(mix_mask.size())
        pitch_i[mix_mask] = pitch[mix_mask]
        if "S" in self.train_mode:
            start = self.get_mix_emb(start_p, start_i, self.trg_start_emb)
            dur = self.get_mix_emb(dur_p, dur_i, self.trg_dur_emb)
            start_i = self.trg_start_emb(start_i)
            dur_i = self.trg_dur_emb(dur_i)
            start_i[mix_mask] = start[mix_mask]
            dur_i[mix_mask] = dur[mix_mask]
        if "T" in self.train_mode:
            # print(start_t_i.size())
            # print(start_t_p.size())
            # print(mix_mask.size())
            start_t_i = torch.unsqueeze(start_t_i, -1)
            end_i = torch.unsqueeze(end_i, -1)
            start_t_i[mix_mask] = start_t_p[mix_mask]
            end_i[mix_mask] = end_p[mix_mask]
            start_t_i = self.start_prj(start_t_i)
            end_i = self.end_prj(end_i)

        if self.train_mode == "T":
            trg_seq = torch.cat([pitch_i, start_t_i, end_i], dim=-1)
        elif self.train_mode == "S":
            trg_seq = torch.cat([pitch_i, start_i, dur_i], dim=-1)
        else:
            trg_seq = torch.cat([pitch_i, start_t_i, end_i, start_i, dur_i], dim=-1)
        
        dec, *_ = self.decoder(trg_seq, trg_mask, enc)

        pitch_out = self.trg_pitch_prj(dec)

        if "S" in self.train_mode:
            start_s_out = self.trg_start_s_prj(dec)
            dur_out = self.trg_dur_s_prj(dec)
        if "T" in self.train_mode:
            start_t_out = self.trg_start_t_prj(dec)
            start_t_out = F.sigmoid(start_t_out)
            end_out = self.trg_end_prj(dec)
            end_out = F.sigmoid(end_out)

        if self.train_mode == "T":
            return pitch_out, start_t_out, end_out
        elif self.train_mode == "S":
            return pitch_out, start_s_out, dur_out
        else:
            return pitch_out, start_t_out, end_out, start_s_out, dur_out           


    def predict(self, mel):
        device = mel.device
        mel = self.cnn(mel)
        mel = torch.permute(mel, (0, 2, 1))

        enc, *_ = self.encoder(mel)

        MAX_LEN = 800
        pitch = torch.zeros((1, MAX_LEN), dtype=int).to(device)
        start_t = torch.zeros((1, MAX_LEN, 1), dtype=torch.float64).to(device)
        end = torch.zeros((1, MAX_LEN, 1), dtype=torch.float64).to(device)
        start = torch.zeros((1, MAX_LEN), dtype=int).to(device)
        dur = torch.zeros((1, MAX_LEN), dtype=int).to(device)
        mask = torch.zeros((1, 1, MAX_LEN), dtype=bool).to(device)
        pitch[:, 0] = INI_IDX

        for i in range(MAX_LEN-1):
            mask[:, :, i] = True

            pitch_emb = self.trg_pitch_emb(pitch)
            if "T" in self.train_mode:
                start_t_emb = self.start_prj(start_t)
                end_emb = self.end_prj(end)
            if "S" in self.train_mode:
                start_emb = self.trg_start_emb(start)
                dur_emb = self.trg_dur_emb(dur)

            if self.train_mode == "T":
                trg_seq = torch.cat([pitch_emb, start_t_emb, end_emb], dim=-1)
            elif self.train_mode == "S":
                trg_seq = torch.cat([pitch_emb, start_emb, dur_emb], dim=-1)
            else:
                trg_seq = torch.cat([pitch_emb, start_t_emb, end_emb, start_emb, dur_emb], dim=-1)

            dec, *_ = self.decoder(trg_seq, mask, enc)

            pitch_out = self.trg_pitch_prj(dec)
            if "S" in self.train_mode:
                start_out = self.trg_start_s_prj(dec)
                dur_out = self.trg_dur_s_prj(dec)
            if "T" in self.train_mode:
                start_t_out = self.trg_start_t_prj(dec)
                end_out = self.trg_end_prj(dec)
                start_t_out = F.sigmoid(start_out)
                end_out = F.sigmoid(end_out)

            pitch_pred = torch.argmax(pitch_out[:, i, :])
            if pitch_pred.item() == EOS_IDX:
                break
            if "S" in self.train_mode:
                start_pred = torch.argmax(start_out[:, i, :])
                dur_pred = torch.argmax(dur_out[:, i, :])

            pitch[:, i+1] = pitch_pred
            if "S" in self.train_mode:
                start[:, i+1] = start_pred
                dur[:, i+1] = dur_pred
            if "T" in self.train_mode:
                start_t[:, i+1, 0] = start_t_out[:, i, 0]
                end[:, i+1, 0] = end_out[:, i, 0]
            # print(pitch[:, :i+1])
            # _ = input()

        pitch = pitch[:, 1:i]
        if "S" in self.train_mode:
            start = start[:, 1:i]
            dur = dur[:, 1:i]
        if "T" in self.train_mode:
            start_t = start_t[:, 1:i, 0]
            end = end[:, 1:i, 0]

        if self.train_mode == "T":
            return pitch, start_t, end
        elif self.train_mode == "S":
            return pitch, start, dur
        else:
            print("------")
            print(pitch)
            print(start)
            print(dur)
            _ = input()
            return pitch, start_t, end, start, dur
        
