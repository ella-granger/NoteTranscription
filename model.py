import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer.Models import Encoder, Decoder, get_pad_mask, get_subsequent_mask
from dataset.constants import *


class NoteTransformer(nn.Module):

    def __init__(self, kernel_size, d_model, d_inner, n_layers):
        super(NoteTransformer, self).__init__()

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
        self.trg_pitch_emb = nn.Embedding(PAD_IDX+1, d_model * 3 // 4, padding_idx=PAD_IDX)
        self.start_prj = nn.Linear(1, d_model // 8)
        self.end_prj = nn.Linear(1, d_model // 8)
        self.decoder = Decoder(d_word_vec=d_model,
                               n_layers=n_layers,
                               n_head=N_HEAD,
                               d_model=d_model,
                               d_inner=d_inner)

        # Result
        self.trg_pitch_prj = nn.Linear(d_model * 3 // 4, PAD_IDX+1)
        self.trg_start_prj = nn.Linear(d_model // 8, 1)
        self.trg_end_prj = nn.Linear(d_model // 8, 1)


    def forward(self, mel, pitch, start, end):
        # mel feature extraction
        mel = self.cnn(mel)
        mel = torch.permute(mel, (0, 2, 1))

        # encoding
        enc, *_ = self.encoder(mel)

        # decoding
        trg_mask = get_pad_mask(pitch, PAD_IDX) & get_subsequent_mask(pitch)

        start = torch.unsqueeze(start, -1)
        end = torch.unsqueeze(end, -1)

        pitch = self.trg_pitch_emb(pitch)
        start = self.start_prj(start)
        end = self.end_prj(end)
        trg_seq = torch.cat([pitch, start, end], dim=-1)
        
        dec, *_ = self.decoder(trg_seq, trg_mask, enc)

        pitch_out = self.trg_pitch_prj(dec[:, :, :(self.d_model * 3 // 4)])
        start_out = self.trg_start_prj(dec[:, :, (-self.d_model // 4):(-self.d_model // 8)])
        start_out = F.sigmoid(start_out)
        end_out = self.trg_end_prj(dec[:, :, (-self.d_model // 8):])
        end_out = F.sigmoid(end_out)
        
        return pitch_out, start_out, end_out
