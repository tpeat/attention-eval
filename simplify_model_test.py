import torch
import torch.nn as nn
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


def simplify_attention(block_class):
    class SimplifiedAttention(block_class):
        # no need to chagne any of the variables, already initialized in parent
        # block_class.projection = self.projection = nn.Linear(self.d_model * 2, self.d_model)
        # def __init__(self):
        #     self.projection = nn.Linear(block_class.d_model*2, block_class)

        def forward(self, Q, K, V):
            print("Simplified Attention")
            print(Q.shape, K.shape, V.shape)
            bs = Q.size(1)
            num_head = self.num_head
            hidden_dim = self.hidden_dim

            # Linear projections for Q and K
            if hasattr(self, 'linear_Q'):
                Q = self.linear_Q(Q)
                K = self.linear_K(K)

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

            # Attention computation
            QK = multiply_by_ychunks(Q, K, self.qk_chunks)
            if self.use_dis:
                QK = 2 * QK - K.pow(2).sum(dim=-2, keepdim=True)
            self.top_k = False
            attn = torch.softmax(QK, dim=-1) if not self.top_k else self.top_k_softmax(QK)
            attn = self.dropout(attn)
            attn_output = multiply_by_xchunks(attn, V_tmp, self.qk_chunks).permute(2, 0, 1, 3).reshape(-1, bs, self.d_model)

            combined_output = attn_output + V
            print(combined_output.shape)
            
            # Final projection
            return self.projection(combined_output), attn

        def top_k_softmax(self, QK):
            top_QK, indices = torch.topk(QK, k=self.top_k, dim=-1)
            top_attn = torch.softmax(top_QK, dim=-1)
            return torch.zeros_like(QK).scatter_(-1, indices, top_attn)

        def _init_weight(self):
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
                    
    return SimplifiedAttention

def make_tristans(model):
    for module in model.modules():
        if "MultiheadAttention" == module.__class__.__name__:
            print("Called")
            module.__class__ = simplify_attention(module.__class__)
            module.__class__.__name__ = "TristanAttention"
    return model