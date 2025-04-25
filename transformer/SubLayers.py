''' Define the sublayers in encoder/decoder layer '''
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformer.Modules import ScaledDotProductAttention

__author__ = "Yu-Hsiang Huang"

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, enable_bn=False):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        if not enable_bn:
            self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.enable_bn = enable_bn
        if enable_bn:
            self.batch_norm = nn.BatchNorm1d(d_model, affine=False, momentum=None, eps=0e-7, track_running_stats=False)
            # self.batch_norm = nn.BatchNorm1d(d_model)


    def forward(self, q, k, v, mask=None):

        save_png = lambda t, f: __import__('torchvision').utils.save_image(((t - t.min()) / (t.max() - t.min())).unsqueeze(0), f)
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q
        self.residual = residual.clone()
        # save_png(residual[0], "residual.png")

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        self.q_prj = q.clone()
        self.q_prj = self.q_prj.view(sz_b, len_q, -1)
        self.v_prj = v.clone()
        self.v_prj = self.v_prj.view(sz_b, len_v, -1)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        if self.enable_bn:
            # print("BATCH NORM", q.size())
            # print([x for x in self.batch_norm.parameters()])
            # print(q[0].mean().item(), q[0].std().item())
            q = q.transpose(1, 2)
            # print(q.size())
            q = self.batch_norm(q)
            q = q.transpose(1, 2)
            # print(q[0].mean().item(), q[0].std().item())
        self.q_enc = q.clone()
        # print(q.size())
        # _ = input()
        q = self.dropout(self.fc(q))
        self.q = q.clone()
        # save_png(attn[0], "attn.png")
        # save_png(q[0], "after_attn.png")
        # _ = input()
        q += residual

        if not self.enable_bn:
            q = self.layer_norm(q)

        return q, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1, enable_bn=False):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.enable_bn = enable_bn
        if enable_bn:
            self.batch_norm = nn.BatchNorm1d(d_in, affine=False, momentum=None, eps=0e-7, track_running_stats=False)
            # self.batch_norm = nn.BatchNorm1d(d_in)
        else:
            self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        if self.enable_bn:
            x = x.transpose(1, 2)
            x = self.batch_norm(x)
            x = x.transpose(1, 2)
        else:
            x = self.layer_norm(x)

        return x
