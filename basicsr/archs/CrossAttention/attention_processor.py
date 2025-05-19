
# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# modified from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# from diffusers.utils import deprecate, logging, maybe_allow_in_graph
from diffusers.utils import deprecate, logging
import warnings
from typing import Callable, Optional, Union

try:
    from torch._dynamo import allow_in_graph as maybe_allow_in_graph
except (ImportError, ModuleNotFoundError):

    def maybe_allow_in_graph(cls):
        return cls


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

@maybe_allow_in_graph
class Attention(nn.Module):
    r"""
    A cross attention layer.
    Parameters:
        query_dim (`int`): The number of channels in the query.
        cross_attention_dim (`int`, *optional*):
            The number of channels in the encoder_hidden_states. If not given, defaults to `query_dim`.
        heads (`int`,  *optional*, defaults to 8): The number of heads to use for multi-head attention.
        dim_head (`int`,  *optional*, defaults to 64): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        bias (`bool`, *optional*, defaults to False):
            Set to `True` for the query, key, and value linear layers to contain a bias parameter.
    """

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias=False,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        cross_attention_norm: Optional[str] = None,
        cross_attention_norm_num_groups: int = 32,
        added_kv_proj_dim: Optional[int] = None,
        norm_num_groups: Optional[int] = None,
        spatial_norm_dim: Optional[int] = None,
        out_bias: bool = True,
        scale_qk: bool = True,
        only_cross_attention: bool = False,
        eps: float = 1e-5,
        rescale_output_factor: float = 1.0,
        residual_connection: bool = False,
        _from_deprecated_attn_block=False,
        processor: Optional["AttnProcessor"] = None,
    ):
        super().__init__()
        inner_dim = dim_head * heads # 64 * 8 = 512
        cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.rescale_output_factor = rescale_output_factor
        self.residual_connection = residual_connection

        # we make use of this private variable to know whether this class is loaded
        # with an deprecated state dict so that we can convert it on the fly
        self._from_deprecated_attn_block = _from_deprecated_attn_block

        self.scale_qk = scale_qk
        self.scale = dim_head**-0.5 if self.scale_qk else 1.0

        self.heads = heads
        # for slice_size > 0 the attention score computation
        # is split across the batch axis to save memory
        # You can set slice_size with `set_attention_slice`
        self.sliceable_head_dim = heads

        self.added_kv_proj_dim = added_kv_proj_dim
        self.only_cross_attention = only_cross_attention

        if self.added_kv_proj_dim is None and self.only_cross_attention:
            raise ValueError(
                "`only_cross_attention` can only be set to True if `added_kv_proj_dim` is not None. Make sure to set either `only_cross_attention=False` or define `added_kv_proj_dim`."
            )

        if norm_num_groups is not None:
            self.group_norm = nn.GroupNorm(num_channels=query_dim, num_groups=norm_num_groups, eps=eps, affine=True)
        else:
            self.group_norm = None

        if spatial_norm_dim is not None:
            self.spatial_norm = SpatialNorm(f_channels=query_dim, zq_channels=spatial_norm_dim)
        else:
            self.spatial_norm = None

        if cross_attention_norm is None:
            self.norm_cross = None
        elif cross_attention_norm == "layer_norm":
            self.norm_cross = nn.LayerNorm(cross_attention_dim)
        elif cross_attention_norm == "group_norm":
            if self.added_kv_proj_dim is not None:
                # The given `encoder_hidden_states` are initially of shape
                # (batch_size, seq_len, added_kv_proj_dim) before being projected
                # to (batch_size, seq_len, cross_attention_dim). The norm is applied
                # before the projection, so we need to use `added_kv_proj_dim` as
                # the number of channels for the group norm.
                norm_cross_num_channels = added_kv_proj_dim
            else:
                norm_cross_num_channels = cross_attention_dim

            self.norm_cross = nn.GroupNorm(
                num_channels=norm_cross_num_channels, num_groups=cross_attention_norm_num_groups, eps=1e-5, affine=True
            )
        else:
            raise ValueError(
                f"unknown cross_attention_norm: {cross_attention_norm}. Should be None, 'layer_norm' or 'group_norm'"
            )

        self.to_q = nn.Linear(query_dim, inner_dim, bias=bias)

        if not self.only_cross_attention:
            # only relevant for the `AddedKVProcessor` classes
            self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias=bias)
            self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias=bias)
        else:
            self.to_k = None
            self.to_v = None

        if self.added_kv_proj_dim is not None:
            self.add_k_proj = nn.Linear(added_kv_proj_dim, inner_dim)
            self.add_v_proj = nn.Linear(added_kv_proj_dim, inner_dim)

        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(inner_dim, query_dim, bias=out_bias))
        self.to_out.append(nn.Dropout(dropout))

        # set attention processor
        # We use the AttnProcessor2_0 by default when torch 2.x is used which uses
        # torch.nn.functional.scaled_dot_product_attention for native Flash/memory_efficient_attention
        # but only if it has the default `scale` argument. TODO remove scale_qk check when we move to torch 2.1
        if processor is None:
            # processor = (
            #     AttnProcessor2_0() if hasattr(F, "scaled_dot_product_attention") and self.scale_qk else AttnProcessor()
            # )
            # Note: efficient attention is not used. We can use efficient attention to speed up.
            processor = AttnProcessor()
        self.set_processor(processor)

    def set_processor(self, processor: "AttnProcessor"):
        # if current processor is in `self._modules` and if passed `processor` is not, we need to
        # pop `processor` from `self._modules`
        if (
            hasattr(self, "processor")
            and isinstance(self.processor, torch.nn.Module)
            and not isinstance(processor, torch.nn.Module)
        ):
            logger.info(f"You are removing possibly trained weights of {self.processor} with {processor}")
            self._modules.pop("processor")

        self.processor = processor

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, return_attntion_probs=False, **cross_attention_kwargs):
        # The `Attention` class can call different attention processors / attention functions
        # here we simply pass along all tensors to the selected processor class
        # For standard processors that are defined here, `**cross_attention_kwargs` is empty
        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            return_attntion_probs=return_attntion_probs,
            **cross_attention_kwargs,
        )

    def batch_to_head_dim(self, tensor):
        head_size = self.heads
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor

    def head_to_batch_dim(self, tensor, out_dim=3):
        head_size = self.heads
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3)

        if out_dim == 3:
            tensor = tensor.reshape(batch_size * head_size, seq_len, dim // head_size)

        return tensor

    def get_attention_scores(self, query, key, attention_mask=None):
        dtype = query.dtype
        if self.upcast_attention:
            query = query.float()
            key = key.float()

        if attention_mask is None:
            baddbmm_input = torch.empty(
                query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device
            )
            beta = 0
        else:
            baddbmm_input = attention_mask
            beta = 1

        attention_scores = torch.baddbmm(
            baddbmm_input,
            query,
            key.transpose(-1, -2),
            beta=beta,
            alpha=self.scale,
        )
        del baddbmm_input

        if self.upcast_softmax:
            attention_scores = attention_scores.float()

        attention_probs = attention_scores.softmax(dim=-1)
        del attention_scores

        attention_probs = attention_probs.to(dtype)

        return attention_probs

    def prepare_attention_mask(self, attention_mask, target_length, batch_size=None, out_dim=3):
        if batch_size is None:
            deprecate(
                "batch_size=None",
                "0.0.15",
                (
                    "Not passing the `batch_size` parameter to `prepare_attention_mask` can lead to incorrect"
                    " attention mask preparation and is deprecated behavior. Please make sure to pass `batch_size` to"
                    " `prepare_attention_mask` when preparing the attention_mask."
                ),
            )
            batch_size = 1

        head_size = self.heads
        if attention_mask is None:
            return attention_mask

        current_length: int = attention_mask.shape[-1]
        if current_length != target_length:
            if attention_mask.device.type == "mps":
                # HACK: MPS: Does not support padding by greater than dimension of input tensor.
                # Instead, we can manually construct the padding tensor.
                padding_shape = (attention_mask.shape[0], attention_mask.shape[1], target_length)
                padding = torch.zeros(padding_shape, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat([attention_mask, padding], dim=2)
            else:
                # TODO: for pipelines such as stable-diffusion, padding cross-attn mask:
                #       we want to instead pad by (0, remaining_length), where remaining_length is:
                #       remaining_length: int = target_length - current_length
                # TODO: re-enable tests/models/test_models_unet_2d_condition.py#test_model_xattn_padding
                attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)

        if out_dim == 3:
            if attention_mask.shape[0] < batch_size * head_size:
                attention_mask = attention_mask.repeat_interleave(head_size, dim=0)
        elif out_dim == 4:
            attention_mask = attention_mask.unsqueeze(1)
            attention_mask = attention_mask.repeat_interleave(head_size, dim=1)

        return attention_mask

    def norm_encoder_hidden_states(self, encoder_hidden_states):
        assert self.norm_cross is not None, "self.norm_cross must be defined to call self.norm_encoder_hidden_states"

        if isinstance(self.norm_cross, nn.LayerNorm):
            encoder_hidden_states = self.norm_cross(encoder_hidden_states)
        elif isinstance(self.norm_cross, nn.GroupNorm):
            # Group norm norms along the channels dimension and expects
            # input to be in the shape of (N, C, *). In this case, we want
            # to norm along the hidden dimension, so we need to move
            # (batch_size, sequence_length, hidden_size) ->
            # (batch_size, hidden_size, sequence_length)
            encoder_hidden_states = encoder_hidden_states.transpose(1, 2)
            encoder_hidden_states = self.norm_cross(encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states.transpose(1, 2)
        else:
            assert False

        return encoder_hidden_states



# https://huggingface.co/spaces/longlian/llm-grounded-diffusion/blob/main/models/attention_processor.py
class AttnProcessor(nn.Module):
    r"""
    Default processor for performing attention-related computations.Q
    attention，利用 text feature 引导 image feature
    下面举例说明：
        Q: [64, 512], 由 image feature 得到
        K，V: [77, 512] 由 text feature 得到
        attention map: M = Q*K^T = [64, 77]
        M * V = [64, 512]， 尺寸与图像一样大
    """

    def __init__(
        self,
        hidden_size=None,
        cross_attention_dim=None,
    ):
        super().__init__()

    def __call__(
        self,
        attn,           # 注意力模块
        hidden_states,  # 输入特征
        encoder_hidden_states=None, # 引导特征
        attention_mask=None,
        temb=None,
    ):

        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        # print(f'hidden states: {hidden_states.shape}')
        # print(f'encoder_hidden_states states: {encoder_hidden_states.shape}')
        # if encoder_hidden_states is None:
        #     batch_size, sequence_length, _ = hidden_states.shape
        # else:
        #     batch_size, sequence_length = encoder_hidden_states.shape

        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        # 计算 Q、K、V
        # Q: 由 输入特征 得到
        # K, V: 由 引导特征 得到
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # 获得 attention map, 并与 value 相乘，得到注意力处理的结果
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.0):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn

class CrossAttention_wx(nn.Module):

    def __init__(self, embed_dim, num_heads):
        super(CrossAttention_wx, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, mask=None):
        batch_size, num_queries, embed_dim = query.size()
        _, num_keys, _ = key.size()

        q = self.q_proj(query).view(batch_size, num_queries, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, num_keys, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, num_keys, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling

        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))

        attn_probs = F.softmax(attn_weights, dim=-1)

        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, num_queries, embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output



class CrossAttention_wx2(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_input, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_input = d_input
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_input, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_input, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_input, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_input, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_input, eps=1e-6)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # print(attn.shape, '2')
        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        # q = self.layer_norm(q)
        # return q, attn
        return q


class CrossAttention_wx3(nn.Module):
    def __init__(self, dim, num_heads, bias=False):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))


        self.q_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.k_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.v_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.k_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.v_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)


        # self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        # self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, q, k, v):
        b,c,h,w = x.shape

        q = self.q_dwconv(self.q_proj(q))
        k = self.k_dwconv(self.k_proj(k))
        v = self.v_dwconv(self.v_proj(v))

        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)

        return out



class IPAttnProcessor(nn.Module):
    r"""
    将 image、text 分别用两个 cross attention 处理
    image: 可学习的 cross attention
    text: 不可学习的 cross attention

    query: image、text 共用

    Attention processor for IP-Adapater.
    Args:
        hidden_size (`int`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`):
            The number of channels in the `encoder_hidden_states`.
        scale (`float`, defaults to 1.0):
            the weight scale of image prompt.
        num_tokens (`int`, defaults to 4 when do ip_adapter_plus it should be 16):
            The context length of the image features.
    """

    def __init__(self, hidden_size, cross_attention_dim=None, scale=1.0, num_tokens=4):
        super().__init__()

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale
        self.num_tokens = num_tokens

        self.to_k_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        # ------------------------------------------------------
        # 将 imgae 、text 分开
        # encoder_hidden_states 包含 image、text的 embedding 信息，两个通过concat的形式融合成 encoder_hidden_states
        # encoder_hidden_states: text_input
        # ip_hidden_states: image input
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            # get encoder_hidden_states, ip_hidden_states
            end_pos = encoder_hidden_states.shape[1] - self.num_tokens
            encoder_hidden_states, ip_hidden_states = (
                encoder_hidden_states[:, :end_pos, :],
                encoder_hidden_states[:, end_pos:, :],
            )
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
                
        # ------------------------------------------------------
        # attention for text
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # ------------------------------------------------------
        # attention for image
        ip_key = self.to_k_ip(ip_hidden_states)
        ip_value = self.to_v_ip(ip_hidden_states)

        ip_key = attn.head_to_batch_dim(ip_key)
        ip_value = attn.head_to_batch_dim(ip_value)

        ip_attention_probs = attn.get_attention_scores(query, ip_key, None)
        self.attn_map = ip_attention_probs
        ip_hidden_states = torch.bmm(ip_attention_probs, ip_value)
        ip_hidden_states = attn.batch_to_head_dim(ip_hidden_states)

        # ------------------------------------------------------
        # 将 image、text 的 cross attention输出进行融合
        hidden_states = hidden_states + self.scale * ip_hidden_states

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class AttnProcessor2_0(torch.nn.Module):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(
        self,
        hidden_size=None,
        cross_attention_dim=None,
    ):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class IPAttnProcessor2_0(torch.nn.Module):
    r"""
    Attention processor for IP-Adapater for PyTorch 2.0.
    Args:
        hidden_size (`int`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`):
            The number of channels in the `encoder_hidden_states`.
        scale (`float`, defaults to 1.0):
            the weight scale of image prompt.
        num_tokens (`int`, defaults to 4 when do ip_adapter_plus it should be 16):
            The context length of the image features.
    """

    def __init__(self, hidden_size, cross_attention_dim=None, scale=1.0, num_tokens=4):
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale
        self.num_tokens = num_tokens

        self.to_k_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            # get encoder_hidden_states, ip_hidden_states
            end_pos = encoder_hidden_states.shape[1] - self.num_tokens
            encoder_hidden_states, ip_hidden_states = (
                encoder_hidden_states[:, :end_pos, :],
                encoder_hidden_states[:, end_pos:, :],
            )
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # for ip-adapter
        ip_key = self.to_k_ip(ip_hidden_states)
        ip_value = self.to_v_ip(ip_hidden_states)

        ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        ip_hidden_states = F.scaled_dot_product_attention(
            query, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
        )
        with torch.no_grad():
            self.attn_map = query @ ip_key.transpose(-2, -1).softmax(dim=-1)
            #print(self.attn_map.shape)

        ip_hidden_states = ip_hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        ip_hidden_states = ip_hidden_states.to(query.dtype)

        hidden_states = hidden_states + self.scale * ip_hidden_states

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


## for controlnet
class CNAttnProcessor:
    r"""
    Default processor for performing attention-related computations.
    """

    def __init__(self, num_tokens=4):
        self.num_tokens = num_tokens

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            end_pos = encoder_hidden_states.shape[1] - self.num_tokens
            encoder_hidden_states = encoder_hidden_states[:, :end_pos]  # only use text
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class CNAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self, num_tokens=4):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        self.num_tokens = num_tokens

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            end_pos = encoder_hidden_states.shape[1] - self.num_tokens
            encoder_hidden_states = encoder_hidden_states[:, :end_pos]  # only use text
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
