import torch
import torch.nn as nn

import numpy as np
from functools import partial

from timm.models.vision_transformer import _cfg, Mlp, HybridEmbed, PatchEmbed
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, DropPath

from lion.curves import compute_curve_order,coords_to_index,index_to_coords_indexes

import warnings
warnings.filterwarnings("ignore")

__all__ = ['lion',
           ]

# Mask Functions 
def Causal_Mask_Decay(a_i , L):
    idx = torch.arange(L,device=a_i.device)
    I, J = torch.meshgrid(idx, idx, indexing='ij')
    E = (torch.abs((I-J)).float().view(1,1,L,L))
    M = torch.sigmoid(a_i).view(1,-1,1,1)**E
    return M

def Casual_Mask_Decay_Partial(a_i , L, start, end):
    idx = torch.arange(L,device=a_i.device)
    I, J = torch.meshgrid(idx, idx[start:end], indexing='ij')
    E = (torch.abs((I-J)).float().view(1,1,L,len(idx[start:end])))
    M = torch.sigmoid(a_i).view(1,-1,1,1)**E
    return M

def create_matrix_from_tensor(tensor):
    cumsum = torch.cumsum(tensor, dim=-1)
    prepend_zeros = torch.zeros(*tensor.shape[:-1], 1, dtype=tensor.dtype, device=tensor.device)
    cumsum = torch.cat((prepend_zeros, cumsum), dim=-1)
    A = cumsum[..., 1:].unsqueeze(-2) - cumsum[..., :-1].unsqueeze(-1)
    A = torch.tril(A.transpose(-1, -2))
    zero_row = torch.zeros(*A.shape[:-2], 1, A.shape[-1], dtype=A.dtype, device=A.device)
    A = torch.cat((zero_row, A[..., :-1, :]), dim=-2)
    return torch.tril(torch.exp(A))

def Causal_Mask_Selective(vec):
    vec_shape = vec.shape
    A_for = create_matrix_from_tensor(vec.unsqueeze(-1).transpose(-1,-2)).squeeze()
    A_back = create_matrix_from_tensor(torch.cat((vec,torch.ones((vec_shape[0],vec_shape[1],1),device=vec.device)),dim=-1)[:,:,1:].unsqueeze(-1).transpose(-1,-2)).transpose(-1,-2).squeeze()    
    return A_for + A_back - torch.eye(A_for.shape[-1]).to(A_for.device)

def create_matrix_from_tensor_forward(tensor, chunk_index,chunk_len):
    cumsum = torch.cumsum(tensor, dim=-1)
    prepend_zeros = torch.zeros(*tensor.shape[:-1], 1, dtype=tensor.dtype, device=tensor.device)
    cumsum = torch.cat((prepend_zeros, cumsum), dim=-1)
    A = cumsum[..., 1:].unsqueeze(-2) - cumsum[..., :-1].unsqueeze(-1)[...,chunk_index*chunk_len:(chunk_index+1)*chunk_len,:]
    A = A.transpose(-1, -2)
    zero_row = torch.zeros(*A.shape[:-2], 1, A.shape[-1], dtype=A.dtype, device=A.device)
    A = torch.cat((zero_row, A[..., :-1, :]), dim=-2)
    return torch.tril(torch.exp(A), diagonal = -chunk_index*chunk_len)

def create_matrix_from_tensor_backward(tensor,chunk_index,chunk_len):
    cumsum = torch.cumsum(tensor, dim=-1)
    prepend_zeros = torch.zeros(*tensor.shape[:-1], 1, dtype=tensor.dtype, device=tensor.device)
    cumsum = torch.cat((prepend_zeros, cumsum), dim=-1)
    A = cumsum[..., :-2].unsqueeze(-2)[...,chunk_index*chunk_len:(chunk_index+1)*chunk_len]  - cumsum[..., :-2].unsqueeze(-1)
    A = A.transpose(-1, -2)
    return torch.tril(torch.exp(A), diagonal = chunk_index*chunk_len)    
   
def Casual_Mask_Selective_Partial(vec, chunk_index, chunk_len):
    B,H,L = vec.shape
    vec_shape = vec.shape
    A_for = create_matrix_from_tensor_forward(vec.unsqueeze(-1).transpose(-1,-2), chunk_index, chunk_len).squeeze()
    A_back = create_matrix_from_tensor_backward(torch.cat((vec,torch.ones((vec_shape[0],vec_shape[1],2),device=vec.device)),dim=-1)[:,:,1:].unsqueeze(-1).transpose(-1,-2), chunk_index, chunk_len).transpose(-1,-2).squeeze()
    I  = torch.diag_embed(torch.ones((B,H,L-chunk_index*chunk_len)),offset = -chunk_index*chunk_len)[...,:A_for.shape[-1]]
    return A_for + A_back - I.to(A_for.device)

# Nonlinearities
def elu_shifted(x):
    s = nn.ELU()
    return s(x) + 1

def silu_shifted(x, silunorm: bool = True):
    s = nn.SiLU()
    res = s(x) + 0.5
    if silunorm:
        res = res / torch.norm(res, dim=-1, keepdim=True)
    return res

class Lion_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 mask_type='Selective', format='Attention', order='Normal', chunk_size=32, num_patches=197):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.mask_type = mask_type
        self.format = format
        self.order = order
        self.chunk_size = chunk_size

        if mask_type == 'Lit':
            self.non_lin = elu_shifted
        elif mask_type == 'Decay':
            self.non_lin = silu_shifted
            self.a_i = nn.Parameter(torch.randn(num_heads))
            if order == 'S':
                self.a2_i = nn.Parameter(torch.randn(num_heads))
        elif mask_type == 'Selective':
            self.non_lin = silu_shifted
            self.a_i = nn.Linear(dim, num_heads)
            if order == 'S':
                self.a2_i = nn.Linear(dim, num_heads)

        if order == 'S':
            N = num_patches
            order = torch.range(0,N-2)
            S = int(np.sqrt(N-1))
            grid = order.view(S,S).clone()
            curve_coords = compute_curve_order(grid, 's')
            self.s_curv_ind = torch.tensor( coords_to_index(grid, curve_coords)  , dtype=torch.long   )
            self.s_curv_ind_inv = torch.tensor(index_to_coords_indexes(curve_coords, S,S)  , dtype=torch.long )  
            
            curve_coords = compute_curve_order(grid, 'sr')
            self.rev_s_curv_ind = torch.tensor( coords_to_index(grid, curve_coords)  , dtype=torch.long   )
            self.rev_s_curv_ind_inv = torch.tensor(index_to_coords_indexes(curve_coords, S,S)  , dtype=torch.long )  

    def forward(self, x):
        B, N, C = x.shape
        H = self.num_heads
        D = C // H
        device = x.device

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = self.non_lin(qkv[0]), self.non_lin(qkv[1]), qkv[2]
        if self.mask_type == 'Selective':
            a_i = torch.tensor(self.a_i(x).transpose(-1, -2).sigmoid_().neg_().add_(1),dtype=torch.float32)[:, :, 1:]
            if self.order == 'S':
                a_i = a_i[..., self.s_curv_ind]
                a2_i = torch.tensor(self.a2_i(x).transpose(-1, -2).sigmoid_().neg_().add_(1),dtype=torch.float32)[:, :, 1:][..., self.rev_s_curv_ind]
                a2_i = torch.cat((torch.ones_like(a2_i[..., :1]), a2_i), dim=-1)
            a_i = torch.cat((torch.ones_like(a_i[..., :1]), a_i), dim=-1)

        if self.order == 'S':
            s_curv_ind = torch.cat((torch.tensor([-1]),self.s_curv_ind)) + 1
            s_curv_ind_inv = torch.cat((torch.tensor([-1]),self.s_curv_ind_inv)) + 1

            rev_s_curv_ind = torch.cat((torch.tensor([-1]),self.rev_s_curv_ind)) + 1
            rev_s_curv_ind_inv = torch.cat((torch.tensor([-1]),self.rev_s_curv_ind_inv)) + 1

        if self.format == 'Attention':
            # Calculate attention matrix
            attn = (q @ k.transpose(-2, -1)) * self.scale
            if self.mask_type == 'Decay':
                M = Causal_Mask_Decay(self.a_i , attn.shape[-1])
                if self.order == 'S':
                    M2 = Causal_Mask_Decay(self.a2_i , attn.shape[-1])
                    M  = (M[:,:,s_curv_ind_inv][:,:,:,s_curv_ind_inv] 
                        + M2[:,:,rev_s_curv_ind_inv][:,:,:,rev_s_curv_ind_inv]) * 0.5
                attn = attn * M

            elif self.mask_type == 'Selective':
                M = Causal_Mask_Selective(torch.log(a_i))
                if self.order == 'S':
                    M2 = Causal_Mask_Selective(torch.log(a2_i))
                    M  = (M[:,:,s_curv_ind_inv][:,:,:,s_curv_ind_inv] 
                        + M2[:,:,rev_s_curv_ind_inv][:,:,:,rev_s_curv_ind_inv]) * 0.5
                attn = attn * M

            # Scale the attention and calculate scores
            attn = attn / (attn.sum(dim=-1,keepdim=True) + 1e-6)
            # attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        elif self.format == 'RNN':
            if self.mask_type == 'Selective':
                a_i =  torch.cat((torch.ones(B,H,1).to(device),a_i,torch.ones(B,H,1).to(device)),dim=-1)

            Si_f = q.new_zeros((B, H, D, D)).to(device)
            Si_b = q.new_zeros((B, H, D, D)).to(device)

            Zi = q.new_zeros((B,H,D)).to(device)
            Zi_b = q.new_zeros((B,H,D)).to(device)

            x = q.new_zeros((B,H,N,D)).to(device)
            cf = q.new_zeros((B,H,N)).to(device)

            for l in range(N):
                if self.order == 'Normal':
                    ind = l
                    ind_b = N-l-1
                else:
                    ind = s_curv_ind[l]
                    ind_b = s_curv_ind[N-l-1]

                # Forward RNN
                Ki = k[:,:,ind,:]
                Qi = q[:,:,ind,:]
                Vi = v[:,:,ind,:]

                KVi = torch.einsum("nhd,nhm->nhdm", Ki, Vi) 

                if self.mask_type == 'Lit':
                    Si_f += KVi
                    Zi += Ki
                elif self.mask_type == 'Decay':
                    Si_f = torch.einsum("h,nhdm->nhdm",  torch.sigmoid(self.a_i), Si_f) + KVi
                    Zi = torch.einsum("h,nhd->nhd",  torch.sigmoid(self.a_i), Zi) + Ki
                elif self.mask_type == 'Selective':
                    Si_f = torch.einsum("nh,nhdm->nhdm", a_i[:,:,l], Si_f) + KVi
                    Zi = torch.einsum("nh,nhd->nhd", a_i[:,:,l], Zi) + Ki

                cf[:,:,ind] += torch.einsum("nhd,nhd->nh", Qi, Zi - Ki/2) 
                x[:,:,ind] += torch.einsum("nhd,nhdm->nhm", Qi, (Si_f - KVi /2))

                # Backward RNN
                Ki = k[:,:,ind_b,:]
                Qi = q[:,:,ind_b,:]
                Vi = v[:,:,ind_b,:]

                KVi = torch.einsum("nhd,nhm->nhdm", Ki, Vi) 
                if self.mask_type == 'Lit':
                    Si_b += KVi
                    Zi_b += Ki
                elif self.mask_type == 'Decay':
                    Si_b = torch.einsum("h,nhdm->nhdm", torch.sigmoid(self.a_i), Si_b) + KVi
                    Zi_b = torch.einsum("h,nhd->nhd", torch.sigmoid(self.a_i), Zi_b) + Ki
                elif self.mask_type == 'Selective':
                    Si_b = torch.einsum("nh,nhdm->nhdm", a_i[:,:,N-l+1], Si_b) + KVi
                    Zi_b = torch.einsum("nh,nhd->nhd", a_i[:,:,N-l+1], Zi_b) + Ki
                
                x[:,:,ind_b] += torch.einsum("nhd,nhdm->nhm", Qi, (Si_b - KVi /2))
                cf[:,:,ind_b] += torch.einsum("nhd,nhd->nh", Qi, Zi_b - Ki/2)
            
            if self.order == 'S':
                Si_f = q.new_zeros((B, H, D, D)).to(device)
                Si_b = q.new_zeros((B, H, D, D)).to(device)

                Zi = q.new_zeros((B,H,D)).to(device)
                Zi_b = q.new_zeros((B,H,D)).to(device)
                
                if self.mask_type == 'Selective':
                    a2_i =  torch.cat((torch.ones(B,H,1).to(device),a2_i,torch.ones(B,H,1).to(device)),dim=-1)

                for l in range(N):
                    ind = rev_s_curv_ind[l]
                    ind_b = rev_s_curv_ind[N-l-1]

                    # Forward RNN
                    Ki = k[:,:,ind,:]
                    Qi = q[:,:,ind,:]
                    Vi = v[:,:,ind,:]

                    KVi = torch.einsum("nhd,nhm->nhdm", Ki, Vi) 

                    if self.mask_type == 'Lit':
                        Si_f += KVi
                        Zi += Ki
                    elif self.mask_type == 'Decay':
                        Si_f = torch.einsum("h,nhdm->nhdm",  torch.sigmoid(self.a2_i), Si_f) + KVi
                        Zi = torch.einsum("h,nhd->nhd",  torch.sigmoid(self.a2_i), Zi) + Ki
                    elif self.mask_type == 'Selective':
                        Si_f = torch.einsum("nh,nhdm->nhdm", a2_i[:,:,l], Si_f) + KVi
                        Zi = torch.einsum("nh,nhd->nhd", a2_i[:,:,l], Zi) + Ki

                    cf[:,:,ind] += torch.einsum("nhd,nhd->nh", Qi, Zi - Ki/2) 
                    x[:,:,ind] += torch.einsum("nhd,nhdm->nhm", Qi, (Si_f - KVi /2))

                    # Backward RNN
                    Ki = k[:,:,ind_b,:]
                    Qi = q[:,:,ind_b,:]
                    Vi = v[:,:,ind_b,:]

                    KVi = torch.einsum("nhd,nhm->nhdm", Ki, Vi) 
                    if self.mask_type == 'Lit':
                        Si_b += KVi
                        Zi_b += Ki
                    elif self.mask_type == 'Decay':
                        Si_b = torch.einsum("h,nhdm->nhdm", torch.sigmoid(self.a2_i), Si_b) + KVi
                        Zi_b = torch.einsum("h,nhd->nhd", torch.sigmoid(self.a2_i), Zi_b) + Ki
                    elif self.mask_type == 'Selective':
                        Si_b = torch.einsum("nh,nhdm->nhdm", a2_i[:,:,N-l+1], Si_b) + KVi
                        Zi_b = torch.einsum("nh,nhd->nhd", a2_i[:,:,N-l+1], Zi_b) + Ki
                    
                    x[:,:,ind_b] += torch.einsum("nhd,nhdm->nhm", Qi, (Si_b - KVi /2))
                    cf[:,:,ind_b] += torch.einsum("nhd,nhd->nh", Qi, Zi_b - Ki/2)

            x = torch.einsum("nhld,nhl->nhld",x, 1 / cf).transpose(1, 2).reshape(B, N, C) 
            
        elif self.format == 'Chunk':
            cf = torch.zeros((B,H,N), device=device)
            chunk_size = self.chunk_size
            x = torch.zeros_like(q)
            if self.order == 'Normal':
                for j in range(0, N, chunk_size):
                    K_chunk = k[...,j:j + chunk_size,:]  
                    V_chunk = v[...,j:j + chunk_size,:]
                    qk = torch.matmul(q, K_chunk.transpose(-2, -1))
                    if self.mask_type == 'Decay':
                        M = Casual_Mask_Decay_Partial(self.a_i,N,j,j + chunk_size)
                        qk = qk * M
                    elif self.mask_type == 'Selective':
                        M = Casual_Mask_Selective_Partial(torch.log(a_i),j//chunk_size,chunk_size)
                        qk = qk * M
                    attn = torch.matmul(qk, V_chunk)
                    cf += qk.sum(3)            
                    x += attn
                x = (x / cf.unsqueeze(-1)).transpose(1, 2).reshape(B, N, C)

            elif self.order == 'S':
                for j in range(0, N, chunk_size):
                    K_chunk = k[...,s_curv_ind[j:j + chunk_size],:]  
                    V_chunk = v[...,s_curv_ind[j:j + chunk_size],:]
                    qk = torch.matmul(q[...,s_curv_ind,:], K_chunk.transpose(-2, -1))
                    if self.mask_type == 'Decay':
                        M = Casual_Mask_Decay_Partial(self.a_i,N,j,j + chunk_size)            
                        qk = qk * M
                    elif self.mask_type == 'Selective':
                        M = Casual_Mask_Selective_Partial(torch.log(a_i),j//chunk_size,chunk_size)
                        qk = qk * M
                    qk = qk[...,s_curv_ind_inv,:]
                    attn = torch.matmul(qk, V_chunk)
                    cf += qk.sum(3)            
                    x += attn

                    K_chunk = k[...,rev_s_curv_ind[j:j + chunk_size],:]  
                    V_chunk = v[...,rev_s_curv_ind[j:j + chunk_size],:]
                    qk = torch.matmul(q[...,rev_s_curv_ind,:], K_chunk.transpose(-2, -1))
                    if self.mask_type == 'Decay':
                        M = Casual_Mask_Decay_Partial(self.a2_i,N,j,j + chunk_size)            
                        qk = qk * M
                    elif self.mask_type == 'Selective':
                        M = Casual_Mask_Selective_Partial(torch.log(a2_i),j//chunk_size,chunk_size)
                        qk = qk * M
                    qk = qk[...,rev_s_curv_ind_inv,:]
                    attn = torch.matmul(qk, V_chunk)
                    cf += qk.sum(3)            
                    x += attn
                x = (x / cf.unsqueeze(-1)).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Lion_Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 mask_type='Selective', format='Attention', order='Normal', chunk_size=32, num_patches=197):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Lion_Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            mask_type=mask_type, format=format, order=order, chunk_size=chunk_size, num_patches=num_patches)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class Lion_VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, 
                 mask_type='Selective', format='Attention', order='Normal', chunk_size=32, pos_emb=True, cls_tok=False):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_emb = pos_emb
        self.cls_tok = cls_tok

        if pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
            self.pos_drop = nn.Dropout(p=drop_rate)
            trunc_normal_(self.pos_embed, std=.02)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Lion_Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                mask_type=mask_type, format=format, order=order, chunk_size=chunk_size, num_patches=num_patches+1)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # NOTE as per official impl, we could have a pre-logits representation dense layer + tanh here
        #self.repr = nn.Linear(embed_dim, representation_size)
        #self.repr_act = nn.Tanh()

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)

        if self.pos_emb:
            x = x + self.pos_embed
            x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        if self.cls_tok:
            return x[:, 0]
        return torch.mean(x, 1)
        

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


# Lion models
@register_model
def lion_base_patch16_224(pretrained=False, **kwargs):
    model = Lion_VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def lion_small_patch16_224(pretrained=False, **kwargs):
    model = Lion_VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def lion_tiny_patch16_224(pretrained=False, **kwargs):
    model = Lion_VisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model