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

    def __init__(self, kernel_size, d_model, d_inner, n_layers, train_mode, seg_len=320, enable_encoder=True, alpha=10, prob_model="gaussian"):
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
        if "S" in train_mode:
            self.trg_start_emb = nn.Embedding(MAX_START+2, d_model // 4 // len(train_mode), padding_idx=MAX_START+1)
            self.trg_dur_emb = nn.Embedding(MAX_DUR+1, d_model // 4 // len(train_mode), padding_idx=0)
        if "T" in train_mode:
            self.start_prj = TimeEncoding(d_model // 4 // len(train_mode), d_model, seg_len)
            self.end_prj = TimeEncoding(d_model // 4 // len(train_mode), d_model, seg_len)
            # self.start_prj = nn.Linear(1, d_model // 4 // len(train_mode))
            # self.end_prj = nn.Linear(1, d_model // 4 // len(train_mode))
        
        self.decoder = Decoder(d_word_vec=d_model,
                               n_layers=n_layers,
                               n_head=N_HEAD,
                               d_model=d_model,
                               d_inner=d_inner)

        # Result
        # self.trg_pitch_prj = nn.Linear(d_model, PAD_IDX+1)
        self.trg_pitch_prj = MLP(d_model, [d_model, d_model, PAD_IDX+1])
        # self.trg_start_prj = nn.Linear(d_model // 8, 1)
        # self.trg_end_prj = nn.Linear(d_model // 8, 1)
        # self.trg_pitch_prj = nn.Linear(d_model * 3 // 4, PAD_IDX+1)
        if "S" in train_mode:
            # self.trg_start_s_prj = nn.Linear(d_model // 8 // len(train_mode), MAX_START+2)
            # self.trg_dur_s_prj = nn.Linear(d_model // 8 // len(train_mode), MAX_DUR+1)
            self.trg_start_s_prj = nn.Linear(d_model, MAX_START+2)
            self.trg_dur_s_prj = nn.Linear(d_model, MAX_DUR+1)
        if "T" in train_mode:
            if prob_model in ["gaussian", "beta"]:
                # self.trg_start_t_prj = nn.Linear(d_model // 8 // len(train_mode), 2) # mu, std
                # self.trg_end_prj = nn.Linear(d_model // 8 // len(train_mode), 2) # mu, std
                self.trg_start_t_prj = nn.Linear(d_model, 2)
                self.trg_end_prj = nn.Linear(d_model, 2)
            elif prob_model in ["l1", "l2", "diou", "gaussian-mu", "l1-diou"]:
                # self.trg_start_t_prj = nn.Linear(d_model // 8 // len(train_mode), 1)
                # self.trg_end_prj = nn.Linear(d_model // 8 // len(train_mode), 1)
                self.trg_start_t_prj = MLP(d_model, [d_model, d_model, 1])
                self.trg_end_prj = MLP(d_model, [d_model, d_model, 1])
                # self.trg_start_t_prj = nn.Linear(d_model, 1)
                # self.trg_end_prj = nn.Linear(d_model, 1)

        self.train_mode = train_mode

    def forward(self, mel, pitch, start_s, dur, start_t, end, return_attns=False, return_cnn=False):
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
            # print(pitch.size())
            # print(start_t.size())
            # print(end.size())
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

        if return_attns:
            dec, dec_self_attn, dec_enc_attn = self.decoder(trg_seq, trg_mask, mel, return_attns=True)
        else:
            dec, *_ = self.decoder(trg_seq, trg_mask, mel)

        # pitch_out = self.trg_pitch_prj(dec[:, :, :(self.d_model * 3 // 4)])
        pitch_out = self.trg_pitch_prj(dec)

        if "S" in self.train_mode:
            #################################BUG!!!!!!!!!!!!!!!!!!!!!!!!
            start_s_out = self.trg_start_s_prj(dec)
            dur_out = self.trg_dur_s_prj(dec)
        if "T" in self.train_mode:
            # start_t_out = self.trg_start_t_prj(dec[:, :, (-self.d_model // 4):(-self.d_model // 8)])
            # end_out = self.trg_end_prj(dec[:, :, (-self.d_model // 8):])
            start_t_out = self.trg_start_t_prj(dec)
            end_out = self.trg_end_prj(dec)
            
            start_t_out = F.sigmoid(start_t_out)
            end_out = F.sigmoid(end_out)
            # start_t_out = F.elu(start_t_out) + 2
            # end_out = F.elu(end_out) + 2
        
        # pitch_out = self.trg_pitch_prj(dec[:, :, :(self.d_model * 3 // 4)])
        # start_out = self.trg_start_prj(dec[:, :, (-self.d_model // 4):(-self.d_model // 8)])
        # start_out = F.sigmoid(start_out)
        # end_out = self.trg_end_prj(dec[:, :, (-self.d_model // 8):])
        # end_out = F.sigmoid(end_out)

        if self.train_mode == "T":
            result = (pitch_out, start_t_out, end_out)
        elif self.train_mode == "S":
            result = (pitch_out, start_s_out, dur_out)
        else:
            result = (pitch_out, start_t_out, end_out, start_s_out, dur_out)
        
        # return pitch_out, start_out, end_out
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


    def forward_mix(self, mel, t,
                    pitch_p, start_p, dur_p, start_t_p, end_p,
                    pitch_i, start_i, dur_i, start_t_i, end_i):
        mel = self.cnn(mel)
        mel = torch.permute(mel, (0, 2, 1))

        if self.enable_encoder:
            mel, *_ = self.encoder(mel)

        trg_mask = get_pad_mask(pitch_i, PAD_IDX) & get_subsequent_mask(pitch_i)

        mix_mask = get_mix_mask(pitch_i, t)

        # pitch = self.get_mix_emb(pitch_p, pitch_i, self.trg_pitch_emb)
        pitch = torch.argmax(pitch_p, dim=-1)
        pitch_i[mix_mask] = pitch[mix_mask]
        pitch_i = self.trg_pitch_emb(pitch_i)
        # print(pitch_i.size())
        # print(pitch.size())
        # print(mix_mask.size())
        # pitch_i[mix_mask] = pitch[mix_mask]
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
        
        dec, *_ = self.decoder(trg_seq, trg_mask, mel)

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


    def predict(self, mel, beam_size=2):
        device = mel.device
        mel = self.cnn(mel)
        mel = torch.permute(mel, (0, 2, 1))

        enc, *_ = self.encoder(mel)
        enc = enc.repeat((beam_size, 1, 1))
        print(enc.size())

        MAX_LEN = 800
        pitch = torch.zeros((beam_size, MAX_LEN), dtype=int).to(device)
        start_t = torch.zeros((beam_size, MAX_LEN, 1), dtype=torch.float64).to(device)
        end = torch.zeros((beam_size, MAX_LEN, 1), dtype=torch.float64).to(device)
        start = torch.zeros((beam_size, MAX_LEN), dtype=int).to(device)
        dur = torch.zeros((beam_size, MAX_LEN), dtype=int).to(device)
        mask = torch.zeros((beam_size, 1, MAX_LEN), dtype=bool).to(device)
        pitch[:, 0] = INI_IDX

        scores = torch.zeros((beam_size, 1), dtype=torch.float64).to(device)
        len_map = torch.arange(1, MAX_LEN + 1, dtype=torch.long).unsqueeze(0).to(device)
        for i in tqdm(range(MAX_LEN-1)):
            mask[:, :, i] = True
            # print(mask)
            # print(pitch)
            # _ = input()

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

            # print(trg_seq.size())
            dec, *_ = self.decoder(trg_seq[:, :(i+1), :], mask[:, :, :(i+1)], enc)

            pitch_out = self.trg_pitch_prj(dec)
            # pitch_out = self.trg_pitch_prj(dec[:, :, :(self.d_model * 3 // 4)])
            if "S" in self.train_mode:
                start_out = self.trg_start_s_prj(dec)
                dur_out = self.trg_dur_s_prj(dec)
            if "T" in self.train_mode:
                start_t_out = self.trg_start_t_prj(dec)
                end_out = self.trg_end_prj(dec)
                # start_t_out = self.trg_start_t_prj(dec[:, :, (-self.d_model // 4):(-self.d_model // 8)])
                # end_out = self.trg_end_prj(dec[:, :, (-self.d_model // 8):])
                start_t_out = F.sigmoid(start_t_out)
                end_out = F.sigmoid(end_out)

            # print(pitch_out[:, i, :])
            # print(pitch_out.shape)

            # Beam Search
            pitch_p = F.softmax(pitch_out, dim=-1)
            # print(pitch_p[:, i, :])
            # print(pitch_p.shape)
            # _ = input()
            if i == 0:
                best_k2_probs, best_k2_idx = pitch_p[0:1, i, :].topk(beam_size)
                print(best_k2_probs)
                print(best_k2_idx)
            else:
                best_k2_probs, best_k2_idx = pitch_p[:, i, :].topk(beam_size)
            # print(best_k2_probs)
            # print(best_k2_idx)
            scores = torch.log(best_k2_probs).view(beam_size, -1) + scores.view(beam_size, 1)
            if i == 0:
                print(scores)

            scores, best_k_idx_in_k2 = scores.view(-1).topk(beam_size)

            best_k_r_idxs, best_k_c_idxs = best_k_idx_in_k2 // beam_size, best_k_idx_in_k2 % beam_size
            best_k_idx = best_k2_idx[best_k_r_idxs, best_k_c_idxs]

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
            
            if "S" in self.train_mode:
                # print(start_out.size())
                start_pred = torch.argmax(start_out[:, i, :], dim=-1)
                # print(start_pred.size())
                dur_pred = torch.argmax(dur_out[:, i, :], dim=-1)

            if "S" in self.train_mode:
                # TODO: rename all the parameters
                start[:, :i+1] = start[best_k_r_idxs, :i+1]
                # print(start_pred.size())
                # _ = input()
                start[:, i+1] = start_pred[:]
                dur[:, :i+1] = dur[best_k_r_idxs, :i+1]
                dur[:, i+1] = dur_pred[:]
            if "T" in self.train_mode:
                start_t[:, :i+1] = start_t[best_k_r_idxs, :i+1]
                start_t[:, i+1, 0] = start_t_out[best_k_r_idxs, i, 0]
                end[:, :i+1] = end[best_k_r_idxs, :i+1]
                end[:, i+1, 0] = end_out[best_k_r_idxs, i, 0]
            # print(pitch[:, :i+1])
            # _ = input()

        print(pitch)
        print(ans_idx)
        ans_len = seq_lens[ans_idx]
        print(ans_len)
        # _ = input()
        pitch = pitch[ans_idx:ans_idx+1, 1:ans_len-1]
        if "S" in self.train_mode:
            start = start[ans_idx:ans_idx+1, 1:ans_len-1]
            dur = dur[ans_idx:ans_idx+1, 1:ans_len-1]
        if "T" in self.train_mode:
            start_t = start_t[ans_idx:ans_idx+1, 1:ans_len-1, 0]
            end = end[ans_idx:ans_idx+1, 1:ans_len-1, 0]

        if self.train_mode == "T":
            return pitch, start_t, end
        elif self.train_mode == "S":
            return pitch, start, dur
        else:
            return pitch, start_t, end, start, dur
        


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
