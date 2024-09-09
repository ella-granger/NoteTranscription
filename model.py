import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
import numpy as np
from torchvision.ops import MLP
from transformer.Models import Encoder, Decoder, get_pad_mask, get_subsequent_mask
from dataset.constants import *
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment


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

    def __init__(self, kernel_size, d_model, d_inner, n_layers, train_mode, seg_len=320, enable_encoder=True, alpha=10, prob_model="gaussian", num_queries=800, n_head=8):
        super(NoteTransformer, self).__init__()

        self.alpha = alpha
        self.enable_encoder = enable_encoder
        self.prob_model = prob_model
        self.seg_len = seg_len
        self.num_queries = num_queries

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
                                   n_head=n_head,
                                   d_model=d_model,
                                   d_inner=d_inner,
                                   n_position=self.seg_len,
                                   scale_emb=True)

        # Decoder
        """
        self.trg_pitch_emb = nn.Embedding(PAD_IDX+1, d_model // 2, padding_idx=PAD_IDX)
        if "S" in train_mode:
            self.trg_start_emb = nn.Embedding(MAX_START+2, d_model // 4 // len(train_mode), padding_idx=MAX_START+1)
            self.trg_dur_emb = nn.Embedding(MAX_DUR+1, d_model // 4 // len(train_mode), padding_idx=0)
        if "T" in train_mode:
            self.start_prj = TimeEncoding(d_model // 4 // len(train_mode), d_model, seg_len)
            self.end_prj = TimeEncoding(d_model // 4 // len(train_mode), d_model, seg_len)
            # self.start_prj = nn.Linear(1, d_model // 4 // len(train_mode))
            # self.end_prj = nn.Linear(1, d_model // 4 // len(train_mode))
        """

        self.query_embed = nn.Embedding(num_queries, d_model)
        
        self.decoder = Decoder(d_word_vec=d_model,
                               n_layers=n_layers,
                               n_head=n_head,
                               d_model=d_model,
                               d_inner=d_inner)

        # Result
        # self.trg_pitch_prj = nn.Linear(d_model, PAD_IDX+1)
        self.trg_pitch_prj = MLP(d_model, [d_model, d_model, NUM_CLS+1])
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

    def forward(self, mel, return_attns=False, return_cnn=False):
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
        queries = self.query_embed.weight
        B = mel.size(0)
        # print(queries.size()) # (N_queries, d_model)
        queries = queries.unsqueeze(0).repeat(B, 1, 1)
        # print(queries.size()) # (N_queries, d_model)
        # print(mel.size()) # (B, L, d_model)
        # _ = input()
        if return_attns:
            dec, dec_self_attn, dec_enc_attn = self.decoder(queries, None, mel, return_attns=True)
        else:
            dec, *_ = self.decoder(queries, None, mel)

        # pitch_out = self.trg_pitch_prj(dec[:, :, :(self.d_model * 3 // 4)])
        pitch_out = self.trg_pitch_prj(dec)

        if "S" in self.train_mode:
            #################################BUG!!!!!!!!!!!!!!!!!!!!!!!!
            start_s_out = self.trg_start_s_prj(dec)
            dur_out = self.trg_dur_s_prj(dec)
        if "T" in self.train_mode:
            start_t_out = self.trg_start_t_prj(dec)
            end_out = self.trg_end_prj(dec)
            
            start_t_out = F.sigmoid(start_t_out)
            end_out = F.sigmoid(end_out)

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
        

    def predict(self, mel, prev_pitch=None, prev_start=None, prev_dur=None, beam_size=2):
        device = mel.device
        mel = self.cnn(mel)
        mel = torch.permute(mel, (0, 2, 1))

        enc, *_ = self.encoder(mel)
        enc = enc.repeat((beam_size, 1, 1))
        # print(enc.size())

        MAX_LEN = 800
        p_start = 0
        pitch = torch.zeros((beam_size, MAX_LEN), dtype=int).to(device)
        start_t = torch.zeros((beam_size, MAX_LEN, 1), dtype=torch.float64).to(device)
        end = torch.zeros((beam_size, MAX_LEN, 1), dtype=torch.float64).to(device)
        start = torch.zeros((beam_size, MAX_LEN), dtype=int).to(device)
        dur = torch.zeros((beam_size, MAX_LEN), dtype=int).to(device)
        mask = torch.zeros((beam_size, 1, MAX_LEN), dtype=bool).to(device)
        pitch[:, 0] = INI_IDX
        if prev_pitch is not None:
            for b in range(beam_size):
                pitch[b, 1:len(prev_pitch)+1] = prev_pitch
                start_t[b, 1:len(prev_pitch)+1, 0] = prev_start
                end[b, 1:len(prev_pitch)+1, 0] = prev_dur
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
                # print(best_k2_probs)
                # print(best_k2_idx)
            else:
                best_k2_probs, best_k2_idx = pitch_p[:, i, :].topk(beam_size)
            # print(torch.log(best_k2_probs))
            # print(best_k2_idx)
            # print(start_t_out[:, :i+1, 0])
            # print(end_out[:, :i+1, 0])
            scores = torch.log(best_k2_probs).view(beam_size, -1) + scores.view(beam_size, 1)
            # if i == 0:
            # print(scores)

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
            # print(pitch[:, :i+2])
            # print(start_t[:, :i+2, 0])
            # print(end[:, :i+2, 0])
            # _ = input()

        # print(pitch)
        # print(ans_idx)
        ans_len = seq_lens[ans_idx]
        # print(ans_len)
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


class DETRLoss(nn.Module):

    def __init__(self, num_classes, weight_dict, empty_weight):
        super(DETRLoss, self).__init__()
        self.num_classes = num_classes
        self.weight_dict = weight_dict
        self.empty_weight = empty_weight
        self.loss_dict = {"box": self.loss_box,
                          "l1": self.loss_l1,
                          "pitch": self.loss_p}
        self.loss_dict = {k:v for k,v in self.loss_dict.items() if k in self.weight_dict}
        self.lambda_l1 = 1
        self.labmda_diou = 0.4


    def loss_p(self, pred, trg):
        p_p = pred[0]
        p_t = trg[0][:trg[-1]]

        # print(p_p.size())
        # _ = input()
        # p_p = F.softmax(p_p, dim=-1)
        
        loss = -p_p[:, p_t]
        return loss


    def loss_box(self, pred, trg):
        length = trg[-1]
        # print(length)
        # print(trg[1].size())

        start_p = pred[1]
        end_p = pred[2]
        start_t = trg[1][:length]
        end_t = trg[2][:length]

        N = start_p.size(0)

        start_p = start_p.repeat(1, length)
        end_p = end_p.repeat(1, length)
        start_t = start_t.unsqueeze(0).repeat(N, 1)
        end_t = end_t.unsqueeze(0).repeat(N, 1)

        c = torch.max(end_p, end_t) - torch.min(start_p, start_t)
        inter = torch.min(end_p, end_t) - torch.max(start_p, start_t)
        inter[inter < 0] = 0
        d = torch.abs((start_p + end_p) / 2 - (start_t + end_t) / 2)

        loss = 1 - inter / c + (d/c) ** 2
        return loss


    def loss_l1(self, pred, trg):
        length = trg[-1]

        start_p = pred[1]
        end_p = pred[2]
        start_t = trg[1][:length]
        end_t = trg[2][:length]

        N = start_p.size(0)

        start_p = start_p.repeat(1, length)
        end_p = end_p.repeat(1, length)
        start_t = start_t.unsqueeze(0).repeat(N, 1)
        end_t = end_t.unsqueeze(0).repeat(N, 1)

        start_loss = torch.abs(start_p - start_t)
        end_loss = torch.abs(end_p - end_t)

        loss = start_loss + end_loss
        return loss


    def forward(self, pitch_p, start_p, end_p, pitch, start, end, length):
        # print(pitch_p.size())
        # print(start_p.size())
        # print(pitch.size())
        pitch_p = F.softmax(pitch_p, dim=-1)

        B = pitch_p.size(0)
        N = pitch_p.size(1)
        losses = []

        out_mat = None
        for i in range(B):
            pred = (pitch_p[i], start_p[i], end_p[i], length[i])
            trg = (pitch[i], start[i], end[i], length[i])

            # Calculate each single loss
            loss_dict = {}
            for k, func in self.loss_dict.items():
                loss_dict[k] = func(pred, trg)

            loss_i = 0
            for k, v in loss_dict.items():
                loss_i += v * self.weight_dict[k]

            # print(loss_i.size())

            # Find the matching path
            mat = loss_i.detach().cpu().numpy()
            row, col = linear_sum_assignment(mat)

            if out_mat is None:
                out_mat = (mat, row, col)
                self.loss_dict_run = loss_dict
            # print(row)
            # print(col)

            # Calculate loss along the matching path
            loss_i = torch.sum(loss_i[row, col])
            if len(row) > 0:
                loss_i /= len(row)

            # for k, v in loss_dict.items():
            #     print(k, torch.sum(v[row, col]).item())

            # Add the empty loss
            empty_p = pitch_p[i, :, -1]
            empty_idx = list(set([x for x in range(N)]) - set(row))
            empty_p = empty_p[empty_idx]
            # print(empty_p.size())
            empty_loss =  - torch.sum(empty_p) * self.empty_weight
            # print(empty_loss.item())
            empty_loss /= len(empty_idx)
            loss_i += empty_loss
            # print(empty_loss.item())
            # _ = input()

            losses.append(loss_i)
        
        return sum(losses), losses, out_mat


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
