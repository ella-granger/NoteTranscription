import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from torch.distributions.categorical import Categorical
from torch.distributions.bernoulli import Bernoulli
from utils import *
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


def get_trg_mask(pitch):
    return get_pad_mask(pitch, PAD_IDX) & get_subsequent_mask(pitch)


class TimeEncoding(nn.Module):

    def __init__(self, d_hid, d_model, seg_len):
        super(TimeEncoding, self).__init__()

        self.d_hid = d_hid
        self.d_model = d_model
        self.seg_len = seg_len


    def forward(self, x):
        # x: (B, L, 1) ~ (0, 1)
        # enc = 200 * x
        # print(x.size())
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
        self.trg_pitch_emb = nn.Embedding(PAD_IDX+1, d_model, padding_idx=PAD_IDX)
        self.trg_voice_emb = nn.Linear(4, d_model)
        self.start_prj = TimeEncoding(d_model, d_model, seg_len)
        self.dur_prj = TimeEncoding(d_model, d_model, seg_len)
        
        self.decoder = Decoder(d_word_vec=d_model,
                               n_layers=n_layers,
                               n_head=N_HEAD,
                               d_model=d_model,
                               d_inner=d_inner)

        # Result
        # self.trg_pitch_prj = nn.Linear(d_model, PAD_IDX+1)
        self.trg_pitch_prj = MLP(d_model, [d_model, d_model, PAD_IDX+1])
        self.trg_voice_prj = MLP(d_model, [d_model // 2, d_model // 4, 4])
        # self.trg_start_prj = nn.Linear(d_model // 8, 1)
        # self.trg_dur_prj = nn.Linear(d_model // 8, 1)
        # self.trg_pitch_prj = nn.Linear(d_model * 3 // 4, PAD_IDX+1)
        if prob_model in ["gaussian", "beta", "sig-log", "sig-norm"]:
            # self.trg_start_prj = nn.Linear(d_model // 8 // len(train_mode), 2) # mu, std
            # self.trg_dur_prj = nn.Linear(d_model // 8 // len(train_mode), 2) # mu, std

            self.trg_start_prj = MLP(d_model, [d_model // 2, d_model // 4, 2])
            self.trg_dur_prj = MLP(d_model, [d_model // 2, d_model // 4, 2])
            
        elif prob_model in ["l1", "l2", "diou", "gaussian-mu", "l1-diou"]:
            # self.trg_start_prj = nn.Linear(d_model // 8 // len(train_mode), 1)
            # self.trg_dur_prj = nn.Linear(d_model // 8 // len(train_mode), 1)
            self.trg_start_prj = MLP(d_model, [d_model // 2, d_model // 4, 1])
            self.trg_dur_prj = MLP(d_model, [d_model // 2, d_model // 4, 1])
            # self.trg_start_prj = nn.Linear(d_model, 1)
            # self.trg_dur_prj = nn.Linear(d_model, 1)


    def encode(self, mel, return_attns=False, return_cnn=False):
        return_attns = return_attns & self.enable_encoder
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

        if return_attns:
            return mel, enc_attn
        else:
            return mel


    def decode(self, mel, trg_seq, trg_mask, return_attns=False):
        # decoding
        if return_attns:
            dec, dec_self_attn, dec_enc_attn = self.decoder(trg_seq, trg_mask, mel, return_attns=True)
        else:
            dec, *_ = self.decoder(trg_seq, trg_mask, mel)

        pitch_out = self.trg_pitch_prj(dec)
        voice_out = self.trg_voice_prj(dec)
        voice_out = F.sigmoid(voice_out)

        start_out = self.trg_start_prj(dec)
        dur_out = self.trg_dur_prj(dec)

        if self.prob_model in ["l1", "l2", "diou", "gaussian-mu", "l1-diou"]:
            start_out = F.sigmoid(start_out)
            dur_out = F.sigmoid(dur_out)
        if self.prob_model in ["sig-log", "sig-norm"]:
            start_out[:, :, 1] = F.sigmoid(start_out[:, :, 1])
            dur_out[:, :, 1] = F.sigmoid(dur_out[:, :, 1])

        result = (pitch_out, start_out, dur_out, voice_out)

        # return pitch_out, start_out, dur_out
        if return_attns:
            return result, (dec_self_attn, dec_enc_attn)
        else:
            return result


    def get_trg_emb(self, pitch, start, dur, voice):
        if len(start.size()) < 3:
            start = torch.unsqueeze(start, -1)
            dur = torch.unsqueeze(dur, -1)
        # print(start.size())
        # print(dur.size())
        start = self.start_prj(start)
        dur = self.dur_prj(dur)
        
        pitch = self.trg_pitch_emb(pitch)
        voice = self.trg_voice_emb(voice)

        trg_seq = pitch + start + dur + voice

        return trg_seq


    def forward(self, mel, pitch, start, dur, voice, return_attns=False, return_cnn=False):
        if return_attns:
            mel, enc_attn = self.encode(mel, return_attns, return_cnn)
        else:
            mel = self.encode(mel, return_attns, return_cnn)

        trg_mask = get_trg_mask(pitch)
        trg_seq = self.get_trg_emb(pitch, start, dur, voice)
        
        # return pitch_out, start_out, dur_out
        if return_attns:
            result, (dec_self_attn, dec_enc_attn) = self.decode(mel, trg_seq, trg_mask, return_attns)
            return result, (enc_attn, dec_self_attn, dec_enc_attn)
        else:
            result = self.decode(mel, trg_seq, trg_mask)
            return result


    def get_mix_emb(self, p, i, emb, mix):
        # p
        e = emb.weight #(idx_num, emb_dim)
        G_y = torch.rand(e.size(0))
        G_y = -torch.log(-torch.log(G_y))

        device = p.device
        G_y = G_y.to(device)

        s = self.alpha * p + G_y
        w = F.softmax(s, dim=-1)
        e = torch.matmul(w, e)

        emb_i = emb(i)
        emb_i[mix] = e[:, :-1][mix[:, 1:]]
        return emb_i


    def get_mix_trg(self, pred, ref, t):
        pitch_p, start_p, dur_p, voice_p = pred
        pitch_r, start_r, dur_r, voice_r = ref
        mix_mask = get_mix_mask(pitch_r, t)

        pitch = self.get_mix_emb(pitch_p, pitch_r, self.trg_pitch_emb, mix_mask)
        # voice = self.get_mix_emb(voice_p, voice_r, self.trg_voice_emb, mix_mask)

        start = torch.unsqueeze(start_r, -1)
        dur = torch.unsqueeze(dur_r, -1)
        start[mix_mask] = start_p[:, :-1][mix_mask[:, 1:]]
        dur[mix_mask] = dur_p[:, :-1][mix_mask[:, 1:]]
        voice = voice_r
        voice[mix_mask] = voice_p[:, :-1][mix_mask[:, 1:]]
        start = self.start_prj(start)
        dur = self.dur_prj(dur)
        voice = self.trg_voice_emb(voice)

        trg_seq = pitch + start + dur + voice

        return trg_seq


    def sample(self, mel, mode):
        B = mel.size(0)
        device = mel.device
        mel = self.encode(mel, False, False)

        MAX_LEN = 200
        pitch = torch.ones((B, MAX_LEN), dtype=int).to(device) * PAD_IDX
        voice = torch.zeros((B, MAX_LEN, 4), dtype=torch.float32).to(device)
        start = torch.zeros((B, MAX_LEN, 1), dtype=torch.float32).to(device)
        dur = torch.zeros((B, MAX_LEN, 1), dtype=torch.float32).to(device)
        ll_total = torch.zeros(B, dtype=torch.float32).to(device)
        
        mask = torch.zeros((B, 1, MAX_LEN), dtype=bool).to(device)
        pitch[:, 0] = INI_IDX

        gen_mask = torch.ones(B, dtype=bool).to(device)
        for i in range(MAX_LEN-1):
            mask[:, :, i] = True

            trg_seq = self.get_trg_emb(pitch[gen_mask], start[gen_mask], dur[gen_mask], voice[gen_mask])
            pitch_p, start_p, dur_p, voice_p = self.decode(mel[gen_mask], trg_seq, mask[gen_mask])

            # print(pitch_p.size())
            # print(start_p.size())
            # print(dur_p.size())
            # print(voice_p.size())
            if mode == "greedy":
                # print("GREEDY")
                pitch[gen_mask, i+1] = torch.argmax(pitch_p[:, i], dim=-1)
                # print(pitch[:, :i+2])
                # print(voice_p[:, i])
                voice[gen_mask, i+1] = (voice_p[:, i] > 0.5).float()
                # print(voice[:, :i+2])
                start[gen_mask, i+1, 0] = F.sigmoid(start_p[:, i, 0])
                dur[gen_mask, i+1, 0] = F.sigmoid(dur_p[:, i, 0])
                # print(start[:, :i+2])
                # print(dur[:, :i+2])
                # _ = input()
            else:
                # print("SAMPLE")
                pitch_dist = Categorical(logits=pitch_p[:, i])
                pitch_s = pitch_dist.sample()
                pitch_ll = pitch_dist.log_prob(pitch_s)
                pitch[gen_mask, i+1] = pitch_s
                # print(pitch[:, :i+2])

                voice_dist = Bernoulli(voice_p[:, i])
                voice_s = voice_dist.sample()
                voice_ll = voice_dist.log_prob(voice_s)
                voice[gen_mask, i+1] = voice_s

                if self.prob_model == "sig-log":
                    start_dist = build_sigmoid_logistics(start_p[:, i, 0], start_p[:, i, 1])
                    dur_dist = build_sigmoid_logistics(dur_p[:, i, 0], dur_p[:, i, 1])
                elif self.prob_model == "sig-norm":
                    start_dist = build_sigmoid_norm(start_p[:, i, 0], start_p[:, i, 1])
                    dur_dist = build_sigmoid_norm(dur_p[:, i, 0], dur_p[:, i, 1])
                start_s = start_dist.sample()
                start_ll = start_dist.log_prob(start_s)
                start[gen_mask, i+1, 0] = start_s

                dur_s = dur_dist.sample()
                dur_ll = dur_dist.log_prob(dur_s)
                dur[gen_mask, i+1, 0] = dur_s

                ll = pitch_ll + voice_ll.sum(dim=-1) + start_ll + dur_ll
                # print(ll)
                ll_total[gen_mask] += ll
                # _ = input()

            gen_mask = gen_mask & (pitch[:, i+1] != EOS_IDX)
            # print(gen_mask)
            # _ = input()
            if not gen_mask.any():
                break
        # print(pitch)
        # print(voice)
        # print(start)
        # print(dur)
        result = (pitch, voice, start, dur)
        # _ = input()
        if mode == "greedy":
            return result
        elif mode == "sample":
            return result, ll_total
                


    def predict(self, mel, prev_pitch=None, prev_start=None, prev_dur=None, prev_voice=None, beam_size=2):
        device = mel.device
        mel = self.encode(mel, False, False)
        mel = mel.repeat((beam_size, 1, 1))
        print(mel.size())

        MAX_LEN = 200
        p_start = 0
        pitch = torch.zeros((beam_size, MAX_LEN), dtype=int).to(device)
        voice = torch.zeros((beam_size, MAX_LEN, 4), dtype=torch.float64).to(device)
        start = torch.zeros((beam_size, MAX_LEN, 1), dtype=torch.float64).to(device)
        dur = torch.zeros((beam_size, MAX_LEN, 1), dtype=torch.float64).to(device)
        mask = torch.zeros((beam_size, 1, MAX_LEN), dtype=bool).to(device)
        pitch[:, 0] = INI_IDX
        voice[:, 0] = 4
        if prev_pitch is not None:
            for b in range(beam_size):
                pitch[b, 1:len(prev_pitch)+1] = prev_pitch
                voice[b, 1:len(prev_pitch)+1] = prev_voice
                start[b, 1:len(prev_pitch)+1, 0] = prev_start
                dur[b, 1:len(prev_pitch)+1, 0] = prev_dur
                mask[b, :, :len(prev_pitch)+1] = True
            p_start = len(prev_pitch)

        scores = torch.zeros((beam_size, 1), dtype=torch.float64).to(device)
        len_map = torch.arange(1, MAX_LEN + 1, dtype=torch.long).unsqueeze(0).to(device)
        for i in tqdm(range(p_start, MAX_LEN-1)):
            mask[:, :, i] = True

            trg_seq = self.get_trg_emb(pitch, start, dur, voice)
            pitch_p, start_p, dur_p, voice_p = self.decode(mel, trg_seq, mask)

            # Beam Search
            pitch_p = F.softmax(pitch_p, dim=-1)
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

            # pitch_pred = torch.argmax(pitch_p[:, i, :]) # Greedy
            # if pitch_pred.item() == EOS_IDX:
            #     break
            # pitch[:, i+1] = pitch_pred
            # print(voice_p.size())
            voice_p = torch.argmax(voice_p[:, i, :], dim=-1)
            # print(voice_p)
            voice[:, :i+1] = voice[best_k_r_idxs, :i+1]
            for j in range(voice.size(0)):
                voice[j, i+1, voice_p[j]] = 1 # voice_p[best_k_r_idxs, i]
            # if "T" in self.train_mode:
            start[:, :i+1] = start[best_k_r_idxs, :i+1]
            start[:, i+1, 0] = start_p[best_k_r_idxs, i, 0]
            dur[:, :i+1] = dur[best_k_r_idxs, :i+1]
            dur[:, i+1, 0] = dur_p[best_k_r_idxs, i, 0]
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
        voice = voice[ans_idx:ans_idx+1, 1:ans_len-1]
        start = start[ans_idx:ans_idx+1, 1:ans_len-1, 0]
        dur = dur[ans_idx:ans_idx+1, 1:ans_len-1, 0]

        return pitch, start, dur, voice
        


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
