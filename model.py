import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
import numpy as np
from torchvision.ops import MLP
from transformer.Models import Encoder, Decoder, get_pad_mask, get_subsequent_mask
from dataset.constants import *
from tqdm import tqdm


def get_mix_mask(pitch, t):
    # reach lowest point at 30k
    U = torch.rand((pitch.size(0), pitch.size(1)))
    U = (U > t)
    U[:, 0] = False
    # print(U)
    # _ = input()
    return U


class TimeEncoding(nn.Module):

    def __init__(self, d_hid, d_model, seg_len):
        super(TimeEncoding, self).__init__()

        self.d_hid = d_hid
        self.d_model = d_model
        self.seg_len = seg_len


    def forward(self, x):
        # x: (B, L, 1) ~ (0, 1)
        # enc = 200 * x
        enc = x.repeat(1, 1, self.d_hid)

        for j in range(self.d_hid):
            enc[:, :, j] = enc[:, :, j] * self.seg_len / np.power(10000, 2 * (j // 2) / self.d_model)
            # div = np.power(10000, 2 * (j // 2) / self.d_hid)
            # enc[:, :, j] = enc[:, :, j] / div

        enc[:, :, 0::2] = torch.sin(enc[:, :, 0::2])
        enc[:, :, 1::2] = torch.cos(enc[:, :, 1::2])

        return enc


class ConvStack(nn.Module):

    def __init__(self, input_features, output_features):
        super().__init__()

        # input is batch_size * 1 channel * frames * input_features
        self.cnn = nn.Sequential(
            # layer 0
            nn.Conv2d(1, output_features // 16, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 16),
            nn.ReLU(),
            # layer 1
            nn.Conv2d(output_features // 16, output_features // 16, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 16),
            nn.ReLU(),
            # layer 2
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
            nn.Conv2d(output_features // 16, output_features // 8, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 8),
            nn.ReLU(),
            # layer 3
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
        )
        self.fc = nn.Sequential(
            nn.Linear((output_features // 8) * (input_features // 4), output_features),
            nn.Dropout(0.5)
        )

    def forward(self, mel):
        x = mel.view(mel.size(0), 1, mel.size(1), mel.size(2))
        x = x.transpose(2,3)
        x = self.cnn(x)
        x = x.transpose(1, 2).flatten(-2)
        x = self.fc(x)
        x = x.transpose(1, 2)
        return x


class NoteTransformer(nn.Module):

    def __init__(self, kernel_size, d_model, d_inner, n_layers, seg_len=320, enable_encoder=True, alpha=10, prob_model="gaussian"):
        super(NoteTransformer, self).__init__()

        self.alpha = alpha
        self.enable_encoder = enable_encoder
        self.prob_model = prob_model
        self.seg_len = seg_len

        # ConvNet
        # """
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
        # """
        # self.cnn = ConvStack(N_MELS, d_model)

        self.d_model = d_model
        # Encoder
        if enable_encoder:
            self.encoder = Encoder(d_word_vec=d_model,
                                   n_layers=n_layers,
                                   n_head=N_HEAD,
                                   d_model=d_model,
                                   d_inner=d_inner,
                                   n_position=self.seg_len,
                                   scale_emb=True)

        # Decoder
        self.trg_pitch_emb = nn.Embedding(PAD_IDX+1, d_model // 2, padding_idx=PAD_IDX)
        self.start_prj = TimeEncoding(d_model // 4, d_model, seg_len)
        self.dur_prj = TimeEncoding(d_model // 4, d_model, seg_len)
        
        self.decoder = Decoder(d_word_vec=d_model,
                               n_layers=n_layers,
                               n_head=N_HEAD,
                               d_model=d_model,
                               d_inner=d_inner)

        # Result
        # self.trg_pitch_prj = nn.Linear(d_model, PAD_IDX+1)
        self.trg_pitch_prj = MLP(d_model, [d_model, d_model, PAD_IDX+1])
        # self.trg_start_prj = nn.Linear(d_model // 8, 1)
        # self.trg_dur_prj = nn.Linear(d_model // 8, 1)
        # self.trg_pitch_prj = nn.Linear(d_model * 3 // 4, PAD_IDX+1)
        if prob_model in ["gaussian", "beta"]:
            # self.trg_start_prj = nn.Linear(d_model // 8 // len(train_mode), 2) # mu, std
            # self.trg_dur_prj = nn.Linear(d_model // 8 // len(train_mode), 2) # mu, std
            self.trg_start_prj = nn.Linear(d_model, 2)
            self.trg_dur_prj = nn.Linear(d_model, 2)
        elif prob_model in ["l1", "l2", "diou", "gaussian-mu", "l1-diou"]:
            # self.trg_start_prj = nn.Linear(d_model // 8 // len(train_mode), 1)
            # self.trg_dur_prj = nn.Linear(d_model // 8 // len(train_mode), 1)
            self.trg_start_prj = MLP(d_model, [d_model, d_model, 1])
            self.trg_dur_prj = MLP(d_model, [d_model, d_model, 1])
            # self.trg_start_prj = nn.Linear(d_model, 1)
            # self.trg_dur_prj = nn.Linear(d_model, 1)


    def forward(self, mel, pitch, start, dur, return_attns=False, return_cnn=False):
        return_attns = return_attns & self.enable_encoder
        # mel feature extraction
        mel = self.cnn(mel)
        if return_cnn:
            self.mel_result = mel
        mel = torch.permute(mel, (0, 2, 1))

        # encoding
        if self.enable_encoder:
            if return_attns:
                mel, enc_attn = self.encoder(mel, return_attns=True)
            else:
                mel, *_ = self.encoder(mel)

        if return_cnn:
            self.enc_result = torch.permute(mel, (0, 2, 1))

        # decoding
        trg_mask = get_pad_mask(pitch, PAD_IDX) & get_subsequent_mask(pitch)

        start = torch.unsqueeze(start, -1)
        dur = torch.unsqueeze(dur, -1)
        start = self.start_prj(start)
        dur = self.dur_prj(dur)
        pitch = self.trg_pitch_emb(pitch)

        # print(pitch.size())
        # print(start.size())
        # print(dur.size())
        trg_seq = torch.cat([pitch, start, dur], dim=-1)

        if return_attns:
            dec, dec_self_attn, dec_enc_attn = self.decoder(trg_seq, trg_mask, mel, return_attns=True)
        else:
            dec, *_ = self.decoder(trg_seq, trg_mask, mel)

        # pitch_out = self.trg_pitch_prj(dec[:, :, :(self.d_model * 3 // 4)])
        pitch_out = self.trg_pitch_prj(dec)

        # start_out = self.trg_start_prj(dec[:, :, (-self.d_model // 4):(-self.d_model // 8)])
        # dur_out = self.trg_dur_prj(dec[:, :, (-self.d_model // 8):])
        start_out = self.trg_start_prj(dec)
        dur_out = self.trg_dur_prj(dec)
        
        start_out = F.sigmoid(start_out)
        dur_out = F.sigmoid(dur_out)
        # start_out = F.elu(start_out) + 2
        # dur_out = F.elu(dur_out) + 2
        
        # pitch_out = self.trg_pitch_prj(dec[:, :, :(self.d_model * 3 // 4)])
        # start_out = self.trg_start_prj(dec[:, :, (-self.d_model // 4):(-self.d_model // 8)])
        # start_out = F.sigmoid(start_out)
        # dur_out = self.trg_dur_prj(dec[:, :, (-self.d_model // 8):])
        # dur_out = F.sigmoid(dur_out)

        result = (pitch_out, start_out, dur_out)
        
        # return pitch_out, start_out, dur_out
        if return_attns:
            return result, (enc_attn, dec_self_attn, dec_enc_attn)
        else:
            return result


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


    def predict(self, mel, prev_pitch=None, prev_start=None, prev_dur=None, beam_size=2):
        device = mel.device
        mel = self.cnn(mel)
        mel = torch.permute(mel, (0, 2, 1))

        enc, *_ = self.encoder(mel)
        enc = enc.repeat((beam_size, 1, 1))
        # print(enc.size())

        MAX_LEN = 200
        p_start = 0
        pitch = torch.zeros((beam_size, MAX_LEN), dtype=int).to(device)
        start = torch.zeros((beam_size, MAX_LEN, 1), dtype=torch.float64).to(device)
        dur = torch.zeros((beam_size, MAX_LEN, 1), dtype=torch.float64).to(device)
        mask = torch.zeros((beam_size, 1, MAX_LEN), dtype=bool).to(device)
        pitch[:, 0] = INI_IDX
        if prev_pitch is not None:
            for b in range(beam_size):
                pitch[b, 1:len(prev_pitch)+1] = prev_pitch
                start[b, 1:len(prev_pitch)+1, 0] = prev_start
                dur[b, 1:len(prev_pitch)+1, 0] = prev_dur
                mask[b, :, :len(prev_pitch)+1] = True
            p_start = len(prev_pitch)

        scores = torch.zeros((beam_size, 1), dtype=torch.float64).to(device)
        len_map = torch.arange(1, MAX_LEN + 1, dtype=torch.long).unsqueeze(0).to(device)
        for i in tqdm(range(p_start, MAX_LEN-1)):
            mask[:, :, i] = True
            # print(mask)
            # print(pitch)
            # _ = input()

            pitch_emb = self.trg_pitch_emb(pitch)
            start_emb = self.start_prj(start)
            dur_emb = self.dur_prj(dur)

            trg_seq = torch.cat([pitch_emb, start_emb, dur_emb], dim=-1)

            # print(trg_seq.size())
            dec, *_ = self.decoder(trg_seq[:, :(i+1), :], mask[:, :, :(i+1)], enc)

            pitch_out = self.trg_pitch_prj(dec)
            start_out = self.trg_start_prj(dec)
            dur_out = self.trg_dur_prj(dec)
            start_out = F.sigmoid(start_out)
            dur_out = F.sigmoid(dur_out)

            # Beam Search
            pitch_p = F.softmax(pitch_out, dim=-1)
            if i == 0:
                best_k2_probs, best_k2_idx = pitch_p[0:1, i, :].topk(beam_size)
            else:
                best_k2_probs, best_k2_idx = pitch_p[:, i, :].topk(beam_size)
            scores = torch.log(best_k2_probs).view(beam_size, -1) + scores.view(beam_size, 1)

            scores, best_k_idx_in_k2 = scores.view(-1).topk(beam_size)
            # print(scores)

            best_k_r_idxs, best_k_c_idxs = best_k_idx_in_k2 // beam_size, best_k_idx_in_k2 % beam_size
            best_k_idx = best_k2_idx[best_k_r_idxs, best_k_c_idxs]
            # print("best r:", best_k_r_idxs)

            pitch[:, :i+1] = pitch[best_k_r_idxs, :i+1]
            pitch[:, i+1] = best_k_idx

            # print(pitch[:, :i+2])
            # _ = input()

            alpha = 0.7
            eos_locs = pitch == EOS_IDX
            seq_lens, _ = len_map.masked_fill(~eos_locs, MAX_LEN).min(1)
            if (eos_locs.sum(1) > 0).sum(0).item() == beam_size or i == MAX_LEN - 2:
                _, ans_idx = scores.div(seq_lens.float() ** alpha).max(0)
                ans_idx = ans_idx.item()
                break
            # _ = input()

            # pitch_pred = torch.argmax(pitch_out[:, i, :]) # Greedy
            # if pitch_pred.item() == EOS_IDX:
            #     break
            # pitch[:, i+1] = pitch_pred
            if "T" in self.train_mode:
                start[:, :i+1] = start[best_k_r_idxs, :i+1]
                start[:, i+1, 0] = start_out[best_k_r_idxs, i, 0]
                dur[:, :i+1] = dur[best_k_r_idxs, :i+1]
                dur[:, i+1, 0] = dur_out[best_k_r_idxs, i, 0]
            # print(pitch[:, :i+2])
            # print(start[:, :i+2, 0])
            # print(dur[:, :i+2, 0])
            # _ = input()

        # print(pitch)
        # print(ans_idx)
        ans_len = seq_lens[ans_idx]
        # print(ans_len)
        # _ = input()
        pitch = pitch[ans_idx:ans_idx+1, 1:ans_len-1]
        start = start[ans_idx:ans_idx+1, 1:ans_len-1, 0]
        dur = dur[ans_idx:ans_idx+1, 1:ans_len-1, 0]

        return pitch, start, dur
        


if __name__ == "__main__":
    model = NoteTransformer(kernel_size=9,
                            d_model=256,
                            d_inner=512,
                            n_layers=2,
                            seg_len=320,
                            train_mode="T",
                            enable_encoder=True,
                            prob_model="l1")

    summary(model)
