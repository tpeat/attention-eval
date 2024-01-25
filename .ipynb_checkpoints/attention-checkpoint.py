import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def multiply_by_ychunks(x, y, chunks=1):
    if chunks <= 1:
        return x @ y
    else:
        return torch.cat([x @ _y for _y in y.chunk(chunks, dim=-1)], dim=-1)
    
def multiply_by_xchunks(x, y, chunks=1):
    if chunks <= 1:
        return x @ y
    else:
        return torch.cat([_x @ y for _x in x.chunk(chunks, dim=-2)], dim=-2)

class CustomAttention(nn.Module):
    def __init__(self, d_model, num_head=8, dropout=0., use_linear=True, d_att=None, use_dis=False, qk_chunks=1, max_mem_len_ratio=-1, top_k=-1):
        super().__init__()
        print("Change")
        self.d_model = d_model
        self.num_head = num_head
        self.use_dis = use_dis
        self.qk_chunks = qk_chunks
        self.max_mem_len_ratio = float(max_mem_len_ratio)
        self.top_k = top_k

        self.hidden_dim = d_model // num_head
        self.d_att = self.hidden_dim if d_att is None else d_att
        self.T = self.d_att**0.5

        # Linear projections for Q and K
        if use_linear:
            self.linear_Q = nn.Linear(d_model, d_model)
            self.linear_K = nn.Linear(d_model, d_model)


        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(2 * d_model, d_model)  # Adjusted for concatenated output
        self._init_weight()

    def forward(self, Q, K, V):
        print("Simplified Attention")
        bs = Q.size(1)
        num_head = self.num_head
        hidden_dim = self.hidden_dim

        # Linear projections for Q and K
        if hasattr(self, 'linear_Q'):
            Q = self.linear_Q(Q)
            K = self.linear_K(K)

        # No linear transformation for V, use MLP instead
        # V_mlp = self.mlp_V(V)

        # Scale
        Q = Q / self.T

        if not self.training and self.max_mem_len_ratio > 0:
            mem_len_ratio = float(K.size(0)) / Q.size(0)
            if mem_len_ratio > self.max_mem_len_ratio:
                scaling_ratio = math.log(mem_len_ratio) / math.log(self.max_mem_len_ratio)
                Q = Q * scaling_ratio

        # Multi-head for Q and K
        Q = Q.view(-1, bs, num_head, self.d_att).permute(1, 2, 0, 3)
        K = K.view(-1, bs, num_head, self.d_att).permute(1, 2, 3, 0)
        V_tmp = V.view(-1, bs, num_head, hidden_dim).permute(1, 2, 0, 3)
      
        QK = multiply_by_ychunks(Q, K, self.qk_chunks)
        if self.use_dis:
            QK = 2 * QK - K.pow(2).sum(dim=-2, keepdim=True)
        self.top_k = False
        attn = torch.softmax(QK, dim=-1) if not self.top_k else self.top_k_softmax(QK)
        attn_dropout = self.dropout(attn)
        attn_output = multiply_by_xchunks(attn, V_tmp, self.qk_chunks).permute(2, 0, 1, 3).reshape(-1, bs, self.d_model)

#         # Concatenate attention output and V
        print(attn.shape, V.shape)
        # attn_output = attn_output.permute(2, 0, 1, 3).reshape(-1, bs, self.d_model)
        print(attn_output.shape)
        combined_output = torch.cat([attn_output, V], dim=-1)

        # Final projection
        return self.projection(combined_output), attn_dropout

    def top_k_softmax(self, QK):
        top_QK, indices = torch.topk(QK, k=self.top_k, dim=-1)
        top_attn = torch.softmax(top_QK, dim=-1)
        return torch.zeros_like(QK).scatter_(-1, indices, top_attn)

    def _init_weight(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)