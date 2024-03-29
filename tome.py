import torch
import math
import torch
import math
from typing import Tuple, Callable

def bipartite_soft_matching(
    metric: torch.Tensor,
    r: int,
    class_token: bool = False,
    distill_token: bool = False,
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with a balanced matching set (50%, 50%).

    Input size is [batch, tokens, channels].
    r indicates the number of tokens to remove (max 50% of tokens).

    Extra args:
     - class_token: Whether or not there's a class token.
     - distill_token: Whether or not there's also a distillation token.

    When enabled, the class token and distillation tokens won't get merged.
    """
    protected = 0
    if class_token:
        protected += 1
    if distill_token:
        protected += 1

    # We can only reduce by a maximum of 50% tokens
    t = metric.shape[1]
    r = min(r, (t - protected) // 2)

    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)

        if class_token:
            scores[..., 0, :] = -math.inf
        if distill_token:
            scores[..., :, 0] = -math.inf

        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

        if class_token:
            # Sort to ensure the class token is at the start
            unm_idx = unm_idx.sort(dim=1)[0]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        # with torch.no_grad():
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        # altered to avoid no grad
        dst = torch.scatter_add(dst, -2, dst_idx.expand(n, r, c), src)

        if distill_token:
            return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
        else:
            return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        # i think we really do want gradient to pass back
        # with torch.no_grad():
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))

        out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)

        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)

        return out

    return merge, unmerge


def apply_merge(x, merge, size=None,  mode="add"):
    """
    x: tensor = [bs, seq_len, embed_dim]
    merge: merge function from bipartite matching
    r: int = portion of tokens to reduce by (must be more than 50%)
    """
    if size is None:
        size = torch.ones_like(x[...,0, None])
    
    x = merge(x * size, mode=mode)
    size = merge(size, mode=mode)

    x = x / size
    return x



def make_tome_block(block_class):
    class ToMeBlock(block_class):
        print("TomeBlock")
        
        def forward(self, Q, K, V):
            # check input sizes
            # if input size 3: permute to b s d -> merge tokens -> call forward -> unmerge -> undo permute to s, b, d
            # if input size 4: boolean that its size 4
            # gonna be much harder to merge the tokens for the [4, 245, 30, 30]
            # save h, w --> reshape to b c h w -> b c (hw)
            # permute to [hw, hw, c] 
            # merge tokens
            # unmerge, undo permute, undo reshapes
            print("In tome block", Q.shape)
            if len(Q.shape) == 3:
                Q, K, V = Q.permute(1, 0, 2), K.permute(1, 0, 2), V.permute(1, 0, 2)  # Permute to S, B, D
                merge_q, unmerge_q = bipartite_soft_matching(Q, Q.shape[1]//2)
                Q = apply_merge(Q, merge_q)
                
                merge_k, unmerge_k = bipartite_soft_matching(K, K.shape[1]//2)
                K = apply_merge(K, merge_k)
                
                merge_v, unmerge_v = bipartite_soft_matching(V, V.shape[1]//2)
                V = apply_merge(V, merge_v)

                Q, K, V = Q.permute(1, 0, 2), K.permute(1, 0, 2), V.permute(1, 0, 2)  # Permute to S, B, D

                output, dropout = super().forward(Q, K, V)  # Call the forward method of the superclass

                # unmerge all of them, or one after another, or just via Q
                output = output.permute(1, 0, 2)
                output = unmerge_q(output)
                output= output.permute(1, 0, 2)

                return output, dropout
            
    return ToMeBlock
            
def make_tome(model):
    for module in model.modules():
        if "MultiheadAttention" == module.__class__.__name__:
            print("Called")
            module.__class__ = make_tome_block(module.__class__)
            module.__class__.__name__ = "ToMeAttention"
    return model
    