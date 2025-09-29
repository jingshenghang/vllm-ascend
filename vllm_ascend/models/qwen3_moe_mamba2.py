from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Optional, Union, List, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from einops import rearrange, repeat

from vllm.attention.backends.utils import PAD_SLOT_ID
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig, LayerBlockType
from vllm.distributed import (
    divide, get_tensor_model_parallel_world_size, get_tensor_model_parallel_rank,
    tensor_model_parallel_all_gather, tensor_model_parallel_all_reduce)
from vllm.distributed import get_pp_group
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear, RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.mamba2_metadata import (
    Mamba2Metadata, prepare_mamba2_metadata)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.models.interfaces import (
    HasInnerState, IsHybrid, SupportsLoRA, SupportsPP)
from vllm.model_executor.models.mamba_cache import MambaCacheParams
from vllm.model_executor.models.qwen3_moe import (
    Qwen3MoeDecoderLayer, Qwen3MoeSparseMoeBlock)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader, extract_layer_index, is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory, make_layers, maybe_prefix)
from vllm.model_executor.model_loader.weight_utils import (
    LoaderFunction, composed_weight_loader,
    sharded_weight_loader, default_weight_loader)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.utils import set_weight_attrs
from vllm.sequence import IntermediateTensors
from vllm_ascend.models.qwen3_moe import AscendQwen3MoeDecoderLayer
from vllm_ascend.ops.fused_moe import AscendSparseMoeBlock

from vllm_ascend.models.qwen3_moe import save_tensor_sequentially
from vllm.distributed import (
            divide, get_tensor_model_parallel_world_size, get_tensor_model_parallel_rank,
            tensor_model_parallel_all_gather, tensor_model_parallel_all_reduce)
from vllm.model_executor.layers.mamba.ops.ssd_combined import (
    mamba_chunk_scan_combined)
from vllm_ascend.models.qwen3_moe import create_stop_flag
import torch_npu



logger = init_logger(__name__)

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def extra_groups_for_head_shards(ngroups: int, tp_size: int):
    """Compute the increase in group numbers to account for
    replication in order to accompany the head shards."""

    # in the case ngoups % tp_size == 0, this will be zero
    if ngroups % tp_size == 0:
        return 0

    # for n_groups == 1, this is exactly tp_size - n_groups
    return tp_size - ngroups


def mamba_v2_sharded_weight_loader(
    shard_spec: list[tuple[int, int, float]],
    tp_size: int,
    tp_rank: int,
) -> LoaderFunction:
    """Create a weight loader for mamba v2. This ensures that the projections 
    are correctly sharded so that they can be split into x, B, C. It also 
    ensures that all the groups corresponding to a head shard is placed 
    together with it.
    """

    def loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:

        # - track boundary of (sharded) param, and loaded_weight, respectively
        boundary, loaded_boundary = 0, 0

        # - iterate over the shard specs
        for full_dim, extra, duplicate_groups in shard_spec:
            # - full dim is the model dim (before TP).
            # - extra > 0, means there is expected overall increase
            #   of dimensions. This is so because of replication.
            # - ratio is used map the tp_rank to the actual shard
            #   rank. This is useful when there is replication of
            #   groups to accompany head shards.

            # - size of the loaded shard
            shard_size = full_dim // tp_size

            # - compute the rank into the loaded shard.
            # - if there is replication, different TP shards will
            #   take from the same rank.
            # NOTE: currently we only support duplication
            # in the case where num_groups == 1
            rank = 0 if duplicate_groups else tp_rank

            # - leftmost boundary index into loaded weight.
            loaded_skip = rank * shard_size
            loaded_start_idx = loaded_boundary + loaded_skip

            # - take these many dims from the loaded weight.
            take = min(shard_size, full_dim - extra - loaded_skip)

            # - always shard on dim 0
            # - the ignore is for a mundane mypy error as it does not
            #   seem to handle slices well.
            # https://github.com/python/mypy/issues/2410
            param.data[
                boundary:(boundary + take),
                ...  # type: ignore[misc]
            ] = loaded_weight[loaded_start_idx:(loaded_start_idx +
                                                take)  # type: ignore[misc]
                              ]  # type: ignore[misc]

            # move indexing boundaries
            boundary += shard_size
            loaded_boundary += full_dim - extra

    return loader


def unbatch_hidden_states(batched_hidden, seq_lens):
    """
    Args:
        batched_hidden: [batch, max(seq_lens), dim]
        seq_lens: [s1, s2, ..., s_batch]

    Returns:
        hidden_states: [seqlen, dim]
    """
    batch_size = batched_hidden.shape[0]
    sequences = []

    for i in range(batch_size):
        seq_len = seq_lens[i]
        sequence = batched_hidden[i, :seq_len]
        sequences.append(sequence)

    hidden_states = torch.cat(sequences, dim=0)

    return hidden_states


def batch_hidden_states(hidden_states, seq_lens):
    """
    Args:
        batched_hidden: [seqlen, dim]
        seq_lens: [s1, s2, ..., s_batch]

    Returns:
        hidden_states: [batch, max(seq_lens), dim]
    """
    sequences = []
    start_idx = 0
    for seq_len in seq_lens:
        sequences.append(hidden_states[start_idx:start_idx + seq_len])
        start_idx += seq_len

    # 对序列进行padding
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)

    return padded_sequences


def pad_for_causal_conv(x, lengths, width):
    batch, _, _ = x.shape
    final_states = []

    for i in range(batch):
        L_i = lengths[i]
        pad_length = max(width - 1 - L_i, 0)

        if L_i >= width - 1:
            # 截断最后 width - 1 个状态
            padded = x[i, :, - (width - 1):]  # [dim, width - 1]
        else:
            # 左边填充 pad_length 个零
            padded = F.pad(x[i], (pad_length, 0))  # [dim, width - 1]

        final_states.append(padded.unsqueeze(0))  # [1, dim, width - 1]

    final_states = torch.cat(final_states, dim=0)  # [batch, dim, width - 1]
    return final_states


def causal_conv1d_update(x, conv_state, weight, bias=None, activation=None, cache_seqlens=None, seq_len=None):
    """
    x: (batch, dim) or (batch, dim, seqlen)
    conv_state: (batch, dim, state_len), where state_len >= width - 1
    weight: (dim, width)
    bias: (dim,)
    cache_seqlens: (batch,), dtype int32.
        If not None, the conv_state is treated as a circular buffer.
        The conv_state will be updated by copying x to the conv_state starting at the index
        @cache_seqlens % state_len before performing the convolution.

    out: (batch, dim) or (batch, dim, seqlen)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    unsqueeze = x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)
    batch, dim, seqlen = x.shape
    width = weight.shape[1]
    state_len = conv_state.shape[-1]
    assert conv_state.shape == (batch, dim, state_len)
    assert weight.shape == (dim, width)
    if cache_seqlens is None:
        x_new = torch.cat([conv_state, x], dim=-1).to(weight.dtype)  # (batch, dim, state_len + seqlen)
        conv_state.copy_(x_new[:, :, -state_len:])
    else:
        width_idx = torch.arange(-(width - 1), 0, dtype=torch.long, device=x.device).unsqueeze(0) + cache_seqlens.unsqueeze(1)
        width_idx = torch.remainder(width_idx, state_len).unsqueeze(1).expand(-1, dim, -1)
        x_new = torch.cat([conv_state.gather(2, width_idx), x], dim=-1).to(weight.dtype)
        copy_idx = torch.arange(seqlen, dtype=torch.long, device=x.device).unsqueeze(0) + cache_seqlens.unsqueeze(1)
        copy_idx = torch.remainder(copy_idx, state_len).unsqueeze(1).expand(-1, dim, -1)
        conv_state.scatter_(2, copy_idx, x)
    out = F.conv1d(x_new, weight.unsqueeze(1), bias, padding=0, groups=dim)[:, :, -seqlen:]
    if unsqueeze:
        out = out.squeeze(-1)
    return (out if activation is None else F.silu(out)).to(dtype=dtype_in), conv_state


def causal_conv1d_fn(
    x,
    weight,
    bias=None,
    initial_states=None,
    return_final_states=False,
    final_states_out=None,
    activation=None,
    seq_len=None
):
    """
    x: (batch, dim, seqlen)
    weight: (dim, width)
    bias: (dim,)
    initial_states: (batch, dim, width - 1)
    final_states_out: (batch, dim, width - 1)

    out: (batch, dim, seqlen)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    x = x.to(weight.dtype)
    seqlen = x.shape[-1]
    dim, width = weight.shape
    if initial_states is None:
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=width - 1, groups=dim)
    else:
        x = torch.cat([initial_states, x], dim=-1)
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=0, groups=dim)
    out = out[..., :seqlen]

    final_states = pad_for_causal_conv(x, seq_len, width)
    if final_states_out is not None:
        final_states_out.copy_(final_states)
    else:
        final_states_out = final_states
    out = (out if activation is None else F.silu(out)).to(dtype=dtype_in)
    return out if not return_final_states else (out, final_states_out)


def chunk_state(B, x, dt, dA_cumsum):
    """
    Argument:
        B: (batch, seqlen, ngroups, headdim)
        x: (batch, seqlen, nheads, headdim)
        dt: (batch, nheads, nchunks, chunk_size)
        dA_cumsum: (batch, nheads, nchunks, chunk_size)
    Return:
        states: (batch, nchunks, nheads, headdim, dstate)
    """
    # Check constraints.
    batch, seqlen, nheads, headdim = x.shape
    dstate = B.shape[-1]
    _, _, nchunks, chunk_size = dt.shape
    assert seqlen <= nchunks * chunk_size
    assert x.shape == (batch, seqlen, nheads, headdim)
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    ngroups = B.shape[2]
    assert nheads % ngroups == 0
    assert B.shape == (batch, seqlen, ngroups, dstate)
    B = repeat(B, "b l g d -> b l (g h) d", h=nheads // ngroups)
    assert dA_cumsum.shape == (batch, nheads, nchunks, chunk_size)
    if seqlen < nchunks * chunk_size:
        x = F.pad(x, (0, 0, 0, 0, 0, nchunks * chunk_size - seqlen))
        B = F.pad(B, (0, 0, 0, 0, 0, nchunks * chunk_size - seqlen))
    x = rearrange(x, "b (c l) h p -> b c l h p", l=chunk_size)
    B = rearrange(B, "b (c l) ... -> b c l ...", l=chunk_size)
    decay_states = torch.exp((dA_cumsum[:, :, :, -1:] - dA_cumsum))
    return torch.einsum("bclhn,bhcl,bhcl,bclhp->bchpn", B.to(x.dtype), decay_states.to(x.dtype), dt.to(x.dtype), x)


def chunk_scan(B, C, x, dt, dA_cumsum, prev_states, D=None, z=None):
    """
    Argument:
        B: (batch, seqlen, ngroups, dstate)
        C: (batch, seqlen, ngroups, dstate)
        x: (batch, seqlen, nheads, headdim)
        dt: (batch, nheads, nchunks, chunk_size)
        dA_cumsum: (batch, nheads, nchunks, chunk_size)
        prev_states: (batch, nchunks, nheads, headdim, dstate)
        D: (nheads, headdim) or (nheads,)
        z: (batch, seqlen, nheads, headdim)
    Return:
        out: (batch, seqlen, nheads, headdim)
    """
    batch, seqlen, nheads, headdim = x.shape
    _, _, ngroups, dstate = B.shape
    assert B.shape == (batch, seqlen, ngroups, dstate)
    _, _, nchunks, chunk_size = dt.shape
    if seqlen != nchunks * chunk_size:
        a = 1
    assert seqlen == nchunks * chunk_size
    assert C.shape == B.shape
    B = repeat(B, "b l g d -> b l (g h) d", h=nheads // ngroups)
    C = repeat(C, "b l g d -> b l (g h) d", h=nheads // ngroups)
    CB = torch.einsum("bclhn,bcshn->bchls", rearrange(C, "b (c l) h n -> b c l h n", c=nchunks),
                      rearrange(B, "b (c s) h n -> b c s h n", c=nchunks))
    # (batch, nheads, nchunks, chunksize, chunksize)
    dt_segment_sum = dA_cumsum[:, :, :, :, None] - dA_cumsum[:, :, :, None, :]
    decay = torch.exp(dt_segment_sum)
    scores_decay = CB * rearrange(decay, "b h c l s -> b c h l s")
    causal_mask = torch.tril(torch.ones(chunk_size, chunk_size, device=x.device, dtype=bool), diagonal=0)
    scores_decay = scores_decay.masked_fill(~causal_mask, 0)
    out = torch.einsum('bchls,bhcs,bcshp->bclhp', scores_decay.to(x.dtype), dt.to(x.dtype),
                       rearrange(x, "b (c s) h p -> b c s h p", c=nchunks))
    state_decay_out = torch.exp(rearrange(dA_cumsum, "b h c l -> b c l h 1"))
    out_prev = torch.einsum('bclhn,bchpn->bclhp', rearrange(C, "b (c l) h n -> b c l h n", c=nchunks),
                            prev_states.to(C.dtype)) * state_decay_out
    out = out + out_prev
    out = rearrange(out, "b c l h p -> b (c l) h p")
    if D is not None:
        if D.dim() == 1:
            D = rearrange(D, "h -> h 1")
        out = out + x * D
    return out if z is None else out * F.silu(z)


def state_passing(states, dA_chunk_cumsum, initial_states=None):
    """
    Argument:
        states: (batch, nchunks, nheads, dim)
        dA_chunk_cumsum: (batch, nheads, nchunks)
        initial_states: (batch, nheads, dim)
    Return:
        out: (batch, nchunks, nheads, dim)
        final_states: (batch, nheads, dim)
    """
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, 0])
    states = torch.cat([rearrange(initial_states, "b h d -> b 1 h d"), states], dim=1)
    dA_chunk_cumsum = F.pad(dA_chunk_cumsum, (1, 0))
    dA_chunk_cumsum = torch.cumsum(dA_chunk_cumsum, dim=-1)
    nchunks = dA_chunk_cumsum.shape[-1]
    # (batch, nheads, nchunks, nchunks)
    dt_chunk_segment_sum = dA_chunk_cumsum[:, :, :, None] - dA_chunk_cumsum[:, :, None, :]
    # (batch, nheads, nchunks, nchunks)
    decay_chunk = torch.exp(dt_chunk_segment_sum)
    causal_mask = torch.tril(torch.ones(nchunks, nchunks, device=states.device, dtype=bool), diagonal=0)
    decay_chunk = decay_chunk.masked_fill(~causal_mask, 0)
    out = torch.einsum("bhzc,bchd->bzhd", decay_chunk.to(dtype=states.dtype), states)
    return out[:, :-1], out[:, -1]


def ssd_chunk_scan_combined(x, dt, A, B, C, chunk_size=256, D=None, z=None, dt_bias=None, dt_softplus=False):
    """
    Argument:
        x: (batch, seqlen, nheads, headdim)
        dt: (batch, seqlen, nheads)
        A: (nheads)
        B: (batch, seqlen, ngroups, dstate)
        C: (batch, seqlen, ngroups, dstate)
        D: (nheads, headdim) or (nheads,)
        z: (batch, seqlen, nheads, headdim)
        dt_bias: (nheads,)
    Return:
        out: (batch, seqlen, nheads, headdim)
    """
    batch, seqlen, nheads, headdim = x.shape
    dstate = B.shape[-1]
    if seqlen < chunk_size:
        chunk_size = seqlen
    origin_seqlen = seqlen
    if seqlen % chunk_size != 0:
        pad_len = chunk_size - seqlen % chunk_size
        dt = F.pad(dt, (0, 0, 0, pad_len))
        B = F.pad(B, (0, 0, 0, 0, 0, pad_len))
        C = F.pad(C, (0, 0, 0, 0, 0, pad_len))
        x = F.pad(x, (0, 0, 0, 0, 0, pad_len))
        seqlen += pad_len
    dt = rearrange(dt, "b (c l) h -> b h c l", l=chunk_size)
    dt = dt.float()  # We want high precision for this before cumsum
    if dt_bias is not None:
        dt = dt + rearrange(dt_bias, "h -> h 1 1")
    if dt_softplus:
        dt = F.softplus(dt)
    dA = dt * rearrange(A, "h -> h 1 1")
    dA_cumsum = torch.cumsum(dA, dim=-1)
    # 1. Compute the state for each chunk
    states = chunk_state(B, x, dt, dA_cumsum)
    states_dtype = states.dtype
    if states.dtype not in [torch.float32, torch.float64]:
        states = states.to(torch.float32)
    # 2. Pass the state to all the chunks by weighted cumsum.
    # state_passing_ref is much less numerically stable
    states, final_states = state_passing(rearrange(states, "... p n -> ... (p n)"), dA_cumsum[:, :, :, -1])
    states, final_states = [rearrange(t,"... (p n) -> ... p n", n=dstate) for t in [states, final_states]]
    states = states.to(states_dtype)
    final_states = final_states.to(states_dtype)
    # 3. Compute the output for each chunk
    out = chunk_scan(B, C, x, dt, dA_cumsum, states, D=D, z=z)
    if origin_seqlen != seqlen:
        out = out[:, :origin_seqlen]

    return out, final_states


@dataclass
class MambaConfig:
    d_model: int = 2048
    d_xb: int = 512
    d_state: int = 128
    d_conv: int = 4
    d_inner: int = None
    expand: int = 2
    ngroups: int = 32
    use_conv_bias: bool = True
    use_bias: bool = False
    rms_norm_eps: float = 1e-6
    attn_layers: List[int] = field(default_factory=list)
    num_experts: int = None
    num_experts_per_tok: int = None
    hidden_size: int = 2048
    moe_intermediate_size: int = 768
    norm_topk_prob: bool = True


class MambaCacheManager:
    def __init__(self, vllm_config: VllmConfig, dtype: torch.dtype,
                 num_mamba_layers: int, conv_state_shape: tuple[int, int],
                 temporal_state_shape: tuple[int, int]):

        max_batch_size = vllm_config.scheduler_config.max_num_seqs

        self.cache_indices_mapping: dict[str, dict[int, int]] = {}
        self.free_cache_indices = list(range(max_batch_size))

        conv_state = torch.empty(size=(num_mamba_layers, max_batch_size) +
                                 conv_state_shape,
                                 dtype=dtype,
                                 device="npu")
        temporal_state = torch.empty(size=(num_mamba_layers, max_batch_size) +
                                     temporal_state_shape,
                                     dtype=dtype,
                                     device="npu")

        self._mamba_cache = (conv_state, temporal_state)

    @property
    def cache(self):
        return self._mamba_cache

    def current_run_tensors(self, **kwargs) -> MambaCacheParams:
        """
        Return the tensors for the current run's conv and ssm state.
        """
        if "seqlen_agnostic_capture_inputs" not in kwargs:
            # We get here only on Prefill/Eager mode runs
            request_ids_to_seq_ids = kwargs["request_ids_to_seq_ids"]
            finished_requests_ids = kwargs["finished_requests_ids"]

            self._release_finished_requests(finished_requests_ids)
            state_indices = self._prepare_current_run_cache(
                request_ids_to_seq_ids, finished_requests_ids)

            state_indices_tensor = torch.as_tensor(state_indices,
                                                   dtype=torch.int32,
                                                   device="npu")
            cache_tensors = self.cache
        else:
            # CUDA graph capturing runs
            cache_tensors, state_indices_tensor = kwargs[
                "seqlen_agnostic_capture_inputs"]

        return MambaCacheParams(cache_tensors[0], cache_tensors[1],
                                state_indices_tensor)

    def _copy_cache(self, from_index: int, to_index: int):
        for cache_t in self.cache:
            cache_t[:, to_index].copy_(cache_t[:, from_index],
                                       non_blocking=True)

    def _assign_seq_id_to_cache_index(self, cur_rid: str, seq_id: int,
                                      finished_requests_ids) -> int:
        """
        Assign (req_id,seq_id) pair to a `destination_index` index, if
        already occupied, move the occupying index to a free index.
        """
        if cur_rid in finished_requests_ids:
            # set as pad, do not allocate destination index
            return PAD_SLOT_ID
        elif cur_rid not in self.cache_indices_mapping:
            destination_index = self.free_cache_indices.pop()
            self.cache_indices_mapping[cur_rid] = {seq_id: destination_index}
            return destination_index
        elif seq_id not in (seq_ids2indices :=
                            self.cache_indices_mapping[cur_rid]):
            # parallel sampling , where n > 1, assume prefill have
            # already happened, so we copy the
            # existing cache into the siblings seq_ids caches
            index_exists = next(iter(seq_ids2indices.values()))
            # case of decoding n>1, copy prefill cache to decoding indices
            destination_index = self.free_cache_indices.pop()
            self._copy_cache(from_index=index_exists,
                             to_index=destination_index)
            self.cache_indices_mapping[cur_rid][seq_id] = destination_index
            return destination_index
        else:
            return self.cache_indices_mapping[cur_rid][seq_id]

    def _prepare_current_run_cache(
            self, request_ids_to_seq_ids: dict[str, list[int]],
            finished_requests_ids: list[str]) -> list[int]:
        return [
            self._assign_seq_id_to_cache_index(req_id, seq_id,
                                               finished_requests_ids)
            for req_id, seq_ids in request_ids_to_seq_ids.items()
            for seq_id in seq_ids
        ]

    def _release_finished_requests(self,
                                   finished_seq_groups_req_ids: list[str]):
        for req_id in finished_seq_groups_req_ids:
            if req_id in self.cache_indices_mapping:
                for seq_id in self.cache_indices_mapping[req_id]:
                    self.free_cache_indices.append(
                        self.cache_indices_mapping[req_id][seq_id])
                self.cache_indices_mapping.pop(req_id)

class Mixer2RMSNormGated(nn.Module):

    def __init__(self,
                 full_hidden_size: int,
                 full_n_groups: int,
                 use_rms_norm: bool = True,
                 eps: float = 1e-6):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.full_hidden_size = full_hidden_size
        self.group_size = full_hidden_size // full_n_groups
        self.per_rank_hidden_size = full_hidden_size // self.tp_size
        self.n_groups = full_hidden_size // self.group_size

        self.variance_epsilon = eps
        self.use_rms_norm = use_rms_norm
        if self.use_rms_norm:
            # Register norm weight only if we're actually applying RMSNorm
            self.weight = nn.Parameter(torch.ones(self.per_rank_hidden_size))
            set_weight_attrs(self.weight,
                             {"weight_loader": sharded_weight_loader(0)})
        else:
            # Avoid checkpoint mismatch by skipping unused parameter
            self.register_parameter("weight", None)
        assert (self.full_hidden_size % self.tp_size == 0
                ), "Tensor parallel world size must divide hidden size."

    def forward(
        self,
        x: torch.Tensor,
        gate: torch.Tensor,
    ):
        # Three tensor-parallel cases:
        #   1. n_groups is 1
        #      In this case we parallelize along the reduction dim.
        #      Each rank computes a local sum of squares followed by AllReduce
        #   2. tp_size divides n_groups
        #      Each rank only reduces within its local group(s).
        #      No collective ops necessary.
        #   3. The general case can be pretty complicated so we AllGather
        #      the input and then redundantly compute the RMSNorm.
        input_dtype = x.dtype
        x = x * nn.functional.silu(gate.to(torch.float32))
        # x = x.to(torch.float32) * nn.functional.silu(gate.to(torch.float32))
        if not self.use_rms_norm:
            return x.to(input_dtype)

        if self.n_groups == 1:
            if self.tp_size > 1:
                # Compute local sum and then reduce to obtain global sum
                local_sums = x.pow(2).sum(dim=-1, keepdim=True)
                global_sums = tensor_model_parallel_all_reduce(local_sums)
                # Calculate the variance
                count = self.tp_size * x.shape[-1]
                variance = global_sums / count

            else:
                variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.variance_epsilon)
        else:
            redundant_tp: bool = self.n_groups % self.tp_size != 0
            if redundant_tp:
                # To handle the general case, redundantly apply the variance
                x = tensor_model_parallel_all_gather(x, -1)

            *prefix_dims, hidden_dim = x.shape
            group_count = hidden_dim // self.group_size
            x_grouped = x.view(*prefix_dims, group_count, self.group_size)
            variance = x_grouped.pow(2).mean(-1, keepdim=True)
            x_grouped = x_grouped * torch.rsqrt(variance +
                                           self.variance_epsilon)
                                           
            x = x_grouped.view(*prefix_dims, hidden_dim) # should use split num 

            if redundant_tp:
                start = self.per_rank_hidden_size * self.tp_rank
                end = start + self.per_rank_hidden_size
                x = x[..., start:end]

        return self.weight * x.to(input_dtype)


class Mamba2RMSNorm(nn.Module):

    def __init__(self, hidden_size, eps=1e-5, group_size=None, norm_before_gate=True, device=None, dtype=None, sequence_parallel: bool = True):
        """If group_size is not None, we do GroupNorm with each group having group_size elements.
        group_size=None is equivalent to group_size=hidden_size (i.e. there's only 1 group).
        """

        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        # self.weight = torch.nn.Parameter(torch.empty(hidden_size, **factory_kwargs))

        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.full_hidden_size = hidden_size
        #self.group_size = full_hidden_size // full_n_groups
        self.per_rank_hidden_size = self.full_hidden_size // self.tp_size

        self.weight = torch.nn.Parameter(torch.ones(self.per_rank_hidden_size))
        set_weight_attrs(self.weight, {"weight_loader": sharded_weight_loader(0)})

        self.register_parameter("bias", None)
        self.group_size = group_size  # do not // 8 , here should be intermediate_size // self.n_groups = 4096 // 32 = 128
        self.norm_before_gate = norm_before_gate
        self.reset_parameters()

        setattr(self.weight, 'sequence_parallel', sequence_parallel)

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)

    def _rms_norm_ref(self, x, weight, bias, z=None, eps=1e-6, group_size=None, norm_before_gate=True, upcast=True):
        dtype = torch.bfloat16
        N = x.shape[-1]
        weight = weight.float()
        bias = bias.float() if bias is not None else None
        # args = get_args()
        use_fused_rmsnorm = False # do not use torch_npu!!!!! TODO
        if upcast:
            x = x.float()
            z = z.float() if z is not None else z
        if z is not None and not norm_before_gate:
            x = x * nn.functional.silu(z)
        if group_size is None:
            if use_fused_rmsnorm:
                out = torch_npu.npu_rms_norm(x, weight, epsilon=eps)[0]
            else:
                rstd = 1 / torch.sqrt((x.square()).mean(dim=-1, keepdim=True) + eps)
                out = x * rstd * weight
            out = out + bias if bias is not None else out
        else:
            x_group = rearrange(x, "... (g d) -> ... g d", d=group_size)
            if use_fused_rmsnorm:
                out = torch_npu.npu_rms_norm(x_group, weight.view(-1, group_size), epsilon=eps)[0]
            else:
                rstd = 1 / torch.sqrt((x_group.square()).mean(dim=-1, keepdim=True) + eps)
                out = x_group * rstd * weight.view(-1, group_size)  
            out = rearrange(out, "... g d -> ... (g d)")
            if bias is not None:
                out = out + bias
        if z is not None and norm_before_gate:
            out *= nn.functional.silu(z)
        return out.to(dtype)        

    def forward(self, x, z=None):
        """If z is not None, we do norm(x) * silu(z) if norm_before_gate, else norm(x * silu(z))
        """
        return self._rms_norm_ref(x, self.weight, self.bias, z=z, eps=self.eps, group_size=self.group_size,
                            norm_before_gate=self.norm_before_gate)       


class MambaMixer2Hybrid(nn.Module):
    def __init__(self,
                hidden_size: int,
                xb_size: int,
                ssm_state_size: int,
                conv_kernel_size: int,
                intermediate_size: int,
                use_conv_bias: bool,
                use_bias: bool,
                n_groups: int = 1,
                rms_norm_eps: float = 1e-5,
                activation="silu",
                quant_config: Optional[QuantizationConfig] = None):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()

        self.xb_size = xb_size

        assert (n_groups % self.tp_size) == 0 or n_groups == 1, \
            (
                "If tensor parallel world size does not divide num_heads, "
                "then num_groups must equal 1."
            )

        assert self.tp_size == 1 or quant_config is None, \
            "Tensor parallel currently not supported for quantized models."

        self.ssm_state_size = ssm_state_size
        self.activation = activation

        self.intermediate_size = intermediate_size
        self.head_dim = intermediate_size // n_groups # 4096 / 32 = 128
        self.repeat_group=intermediate_size // xb_size
        num_heads = n_groups
        self.num_heads = num_heads

        self.n_groups = n_groups
        if n_groups % self.tp_size != 0:
            # - for TP we shard conv_dim by sharding on n_groups,
            # - but if n_groups cannot divide tp_size, we need to
            #   extend some extra groups
            self.n_groups = n_groups + extra_groups_for_head_shards(
                n_groups, self.tp_size)

        self.conv_dim = intermediate_size + xb_size + xb_size
        self.conv1d = ColumnParallelLinear(
            input_size=conv_kernel_size,
            output_size=self.conv_dim,
            bias=use_conv_bias,
            quant_config=None,
        )
        # unsqueeze to fit conv1d weights shape into the linear weights shape.
        # Can't do this in `weight_loader` since it already exists in
        # `ColumnParallelLinear` and `set_weight_attrs`
        # doesn't allow to override it
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)

        self.in_proj = ColumnParallelLinear(input_size=hidden_size,
                                            output_size=intermediate_size +
                                            self.conv_dim + self.num_heads,
                                            bias=use_bias,
                                            quant_config=quant_config)

        # - because in_proj is a concatenation of 3 weights, we
        #   need to interleave them before sharding
        # - use the custom weight loader mamba_v2_sharded_weight_loader
        #   for conv1d.bias, covn1d.weight and in_proj.weight
        # - need to set these settings, to assign the groups to the head shards
        group_shard_settings = (
            self.n_groups * self.ssm_state_size,  # expected model size
            (self.n_groups - n_groups) *
            self.ssm_state_size,  # extra dims assigned
            n_groups == 1,  # if there was only one group
        )
        intermediate_settings = (intermediate_size, 0, False)
        head_setings = (self.num_heads, 0, False)
        xb_settings = (self.xb_size, 0, False)

        # - the weight already has a "weight_loader" attribute
        #   which set_weight_attrs will raise if we do not
        #   delete before trying to override it
        # - ditto for the otther two weights below
        delattr(self.conv1d.bias, "weight_loader")
        set_weight_attrs(
            self.conv1d.bias, {
                "weight_loader":
                mamba_v2_sharded_weight_loader(
                    [
                        xb_settings,
                        xb_settings,
                        group_shard_settings,
                    ],
                    self.tp_size,
                    tp_rank,
                )
            })

        delattr(self.conv1d.weight, "weight_loader")
        set_weight_attrs(
            self.conv1d.weight, {
                "weight_loader":
                mamba_v2_sharded_weight_loader([
                    xb_settings,
                    xb_settings,
                    group_shard_settings,
                ], self.tp_size, tp_rank)
            })

        if quant_config is None:
            # - quant layers do not have a weight loader
            delattr(self.in_proj.weight, "weight_loader")
            set_weight_attrs(
                self.in_proj.weight,
                {
                    "weight_loader":
                    mamba_v2_sharded_weight_loader(
                        [
                            intermediate_settings,  # for gate
                            xb_settings,
                            xb_settings,
                            group_shard_settings,
                            head_setings,  # for dt
                        ],
                        self.tp_size,
                        tp_rank)
                })

        # - these are TPed by heads to reduce the size of the
        #   temporal shape
        self.A_log = nn.Parameter(
            torch.empty(
                divide(num_heads, self.tp_size),
                dtype=torch.float32,
            ))
        self.A = self.A_log
        self.D = nn.Parameter(torch.ones(num_heads // self.tp_size))
        self.dt_bias = nn.Parameter(torch.ones(num_heads // self.tp_size))

        set_weight_attrs(self.D, {"weight_loader": sharded_weight_loader(0)})
        a_weight_loader = composed_weight_loader(
            sharded_weight_loader(0), lambda x: -torch.exp(x.float()))
        set_weight_attrs(self.A, {"weight_loader": a_weight_loader})
        set_weight_attrs(self.dt_bias,
                         {"weight_loader": sharded_weight_loader(0)})

        self.out_proj = RowParallelLinear(intermediate_size,
                                          hidden_size,
                                          bias=use_bias,
                                          input_is_parallel=True,
                                          quant_config=quant_config)

        # self.norm = Mixer2RMSNormGated(intermediate_size,
        #                                n_groups,
        #                                eps=rms_norm_eps)
        
        self.norm = Mamba2RMSNorm(
                intermediate_size,
                eps=rms_norm_eps,
                group_size=intermediate_size // self.n_groups,
                norm_before_gate=False,
                device=torch_npu.npu.current_device(),
                dtype=torch.bfloat16,
            )


    def forward(
        self,
        hidden_states: torch.Tensor,
        mamba_cache_params: MambaCacheParams,
        mamba2_metadata: Mamba2Metadata,
    ):
        save_tensor_sequentially(hidden_states, "start_mamba", get_tensor_model_parallel_rank())
        # 0. attn_metadata
        attn_metadata = get_forward_context().attn_metadata
        prefill = attn_metadata.num_prefills > 0
        seq_len, _ = hidden_states.shape
        groups_time_state_size = self.n_groups * self.ssm_state_size

        num_prefills = attn_metadata.num_prefills  # request count
        num_decodes = attn_metadata.num_decode_tokens  # token count (=request)
        num_prefill_tokens = attn_metadata.num_prefill_tokens  # token count
        has_prefill = num_prefills > 0
        has_decode = num_decodes > 0

        # 1. Gated MLP's linear projection
        projected_states, _ = self.in_proj(hidden_states)
        gate, hidden_states_B_C, dt = torch.split(
            projected_states,
            [
                self.intermediate_size // self.tp_size,
                self.conv_dim // self.tp_size,
                self.num_heads // self.tp_size,
            ],
            dim=-1,
        )

        # dt_p, dt_d = torch.split(
        #     dt,
        #     [num_prefill_tokens, num_decodes],
        #     dim=0,
        # )

        # 2. Convolution sequence transformation
        conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0),
                                               self.conv1d.weight.size(2))
        

        if prefill:
            hidden_states_batched = batch_hidden_states(hidden_states_B_C, attn_metadata.seq_lens)
            hidden_states_B_C_batched, final_states_out = causal_conv1d_fn(
                x=hidden_states_batched.transpose(1, 2),
                weight=conv_weights,
                bias=self.conv1d.bias,
                initial_states=None,
                activation=self.activation,
                final_states_out=None,
                return_final_states=True,
                seq_len=attn_metadata.seq_lens
            )
            mamba_cache_params.conv_state[mamba_cache_params.state_indices_tensor] = final_states_out
            hidden_states_B_C = hidden_states_B_C_batched.transpose(1, 2)
        else:
            hidden_states_B_C, conv_state = causal_conv1d_update(
                x=hidden_states_B_C,
                conv_state=mamba_cache_params.conv_state[mamba_cache_params.state_indices_tensor],
                weight=conv_weights,
                bias=self.conv1d.bias,
                activation=self.activation,
                seq_len=attn_metadata.seq_lens
            )

            mamba_cache_params.conv_state[mamba_cache_params.state_indices_tensor] = conv_state


        # - get hidden_states, B and C after depthwise convolution.
        hidden_states, B, C = torch.split(
            hidden_states_B_C,
            [
                self.xb_size // self.tp_size,
                self.xb_size // self.tp_size,
                groups_time_state_size // self.tp_size,
            ],
            dim=-1,
        )

        save_tensor_sequentially(hidden_states, "after_conv", get_tensor_model_parallel_rank())

        # 3. State Space Model sequence transformation
        if prefill:
            # minic the GQA
            hidden_states = rearrange(hidden_states, "b l (xb_group dstate) -> b xb_group l dstate", dstate=self.ssm_state_size)
            hidden_states = repeat_kv(hidden_states, self.repeat_group)
            hidden_states = rearrange(hidden_states, "b g l p -> b l g p")
            
            B = rearrange(B, "b l (xb_group dstate) -> b xb_group l dstate", dstate=self.ssm_state_size)
            B = repeat_kv(B, self.repeat_group)
            B = rearrange(B, "b g l n -> b l g n")

            C = rearrange(C, "b l (g n) -> b l g n", g=B.shape[-2])

            flag = False

            if flag:
                chunk_size = 256
                initial_states = None
                if (mamba2_metadata.has_initial_states is not None
                        and mamba2_metadata.prep_initial_states):
                    # making a copy of the states
                    initial_states = torch.where(
                        mamba2_metadata.has_initial_states[:, None, None, None],
                        mamba_cache_params.ssm_state[state_indices_tensor_p], 0)
                scan_output, varlen_state = mamba_chunk_scan_combined(
                    C.reshape(1, num_prefill_tokens,
                                         self.num_heads // self.tp_size,
                                         self.head_dim),
                    dt.unsqueeze(0),
                    self.A,
                    B.reshape(1, num_prefill_tokens, self.n_groups // self.tp_size,
                             -1),
                    hidden_states.reshape(1, num_prefill_tokens, self.n_groups // self.tp_size,
                             -1),
                    chunk_size=chunk_size, # mamba2_metadata.chunk_size,
                    D=self.D,
                    z=None,
                    dt_bias=self.dt_bias,
                    seq_idx=mamba2_metadata.seq_idx,
                    chunk_indices=mamba2_metadata.chunk_indices,
                    chunk_offsets=mamba2_metadata.chunk_offsets,
                    cu_seqlens=attn_metadata.query_start_loc[:num_prefills + 1],
                    initial_states=initial_states,
                    return_varlen_states=True,
                    return_final_states=False,
                    dt_softplus=True,
                    dt_limit=(0.0, float("inf")),
                )
                a = 1

            scan_output, states = ssd_chunk_scan_combined(
                x=hidden_states,
                dt=batch_hidden_states(dt, attn_metadata.seq_lens),
                A=self.A,
                B=B,
                C=C,
                D=self.D,
                z=None,
                dt_bias=self.dt_bias,
                dt_softplus=True,
                chunk_size=256
            )
            # update ssm states
            mamba_cache_params.ssm_state[mamba_cache_params.state_indices_tensor] = states
            hidden_states = unbatch_hidden_states(scan_output.to(hidden_states.dtype), attn_metadata.seq_lens)
            hidden_states = hidden_states.view(seq_len, -1)

        else:
            # minic the GQA
            hidden_states = rearrange(hidden_states, "b (xb_group dstate) -> b xb_group dstate", dstate=self.ssm_state_size)
            x = torch.repeat_interleave(hidden_states, dim=1, repeats=self.repeat_group)

            B = rearrange(B, "b (xb_group dstate) -> b xb_group dstate", dstate=self.ssm_state_size)
            B = torch.repeat_interleave(B, dim=1, repeats=self.repeat_group)

            A = repeat(self.A, "h -> h p n", p=self.head_dim, n=self.ssm_state_size).to(dtype=torch.float32)
            dt = repeat(dt, "b h -> b h p", p=self.head_dim)
            dt_bias = repeat(self.dt_bias, "h -> h p", p=self.head_dim)
            D = repeat(self.D, "h -> h p", p=self.head_dim)
            C = rearrange(C, "b (g n) -> b g n", g=B.shape[-2])

            state = mamba_cache_params.ssm_state[mamba_cache_params.state_indices_tensor]
            batch, nheads, _, _ = state.shape
            ngroups = B.shape[1]

            if dt_bias is not None:
                dt = dt + dt_bias
            dt = F.softplus(dt)
            dA = torch.exp(rearrange(dt, "b h d -> b h d 1") * A)  # (batch, nheads, dim, dstate)
            B = repeat(B, "b g n -> b (g h) n", h=nheads // ngroups)  # (batch, nheads, dstate)
            C = repeat(C, "b g n -> b (g h) n", h=nheads // ngroups)  # (batch, nheads, dstate)
            dB = rearrange(dt, "b h d -> b h d 1") * rearrange(B, "b h n -> b h 1 n")  # (batch, nheads, dim, dstate)
            out = torch.einsum("bhdn,bhn->bhd", state.to(C.dtype), C)
            if D is not None:
                out += (x * D).to(out.dtype)
            hidden_states = out.view(batch, -1).to(x.dtype)
            state_new = state * dA + dB * rearrange(x, "b h d -> b h d 1")
            mamba_cache_params.ssm_state[mamba_cache_params.state_indices_tensor] = state_new.to(C.dtype)  # (batch, dim, dstate


        # 4. gated MLP

        save_tensor_sequentially(hidden_states, "before_norm", get_tensor_model_parallel_rank())
        hidden_states = self.norm(hidden_states.unsqueeze(0), gate.unsqueeze(0)).squeeze(0)
        save_tensor_sequentially(hidden_states, "after_norm", get_tensor_model_parallel_rank())

        # 5. Final linear projection
        out, _ = self.out_proj(hidden_states)
        save_tensor_sequentially(out, "after_proj", get_tensor_model_parallel_rank())
        return out


class Mamba2DecoderLayer(nn.Module):

    def __init__(self,
                 config: MambaConfig,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = "") -> None:
        super().__init__()
        self.config = config

        if config.d_inner is None:
            config.d_inner = config.expand * config.d_model

        self.mamba = MambaMixer2Hybrid(hidden_size=config.d_model,
                                       xb_size=config.d_xb,
                                       ssm_state_size=config.d_state,
                                       conv_kernel_size=config.d_conv,
                                       intermediate_size=config.d_inner,
                                       use_conv_bias=config.use_conv_bias,
                                       use_bias=config.use_bias,
                                       n_groups=config.ngroups,
                                       quant_config=quant_config)

        self.mlp = AscendSparseMoeBlock(config=config, # Qwen3MoeSparseMoeBlock
                                          quant_config=quant_config,
                                          prefix=f"{prefix}.mlp")

        self.input_layernorm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.d_model, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        mamba_cache_params: MambaCacheParams = None,
        mamba2_metadata: Mamba2Metadata = None
    ):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual) # inside norm hidden_states + residual

        hidden_states = self.mamba(hidden_states,mamba_cache_params, mamba2_metadata)
        hidden_states = hidden_states + residual

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


@support_torch_compile
class Qwen3MoeMamba2Model(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        mamba_config = MambaConfig(**config.mamba_config)

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=f"{prefix}.embed_tokens")

        def get_layer(prefix):
            idx = extract_layer_index(prefix)
            if idx in mamba_config.attn_layers:
                return AscendQwen3MoeDecoderLayer(config=config, # AscendQwen3MoeDecoderLayer
                                            cache_config=cache_config,
                                            quant_config=quant_config,
                                            prefix=prefix)
            else:
                return Mamba2DecoderLayer(config=mamba_config, 
                                          quant_config=quant_config, 
                                          prefix=prefix)

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            get_layer,
            prefix=f"{prefix}.layers",
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        mamba_cache_params: MambaCacheParams,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        attn_metadata = get_forward_context().attn_metadata
        
        mamba2_metadata = None
        mamba2_metadata = prepare_mamba2_metadata(
            chunk_size=256,
            attn_metadata=attn_metadata,
        )
        mamba_index = 0
        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            # print("layer {}".format(i))
            if isinstance(layer, AscendQwen3MoeDecoderLayer):
                hidden_states, residual = layer(
                    positions=positions, 
                    hidden_states=hidden_states, 
                    residual=residual)
            else:
                hidden_states, residual = layer(
                    hidden_states=hidden_states, 
                    residual=residual, 
                    mamba_cache_params=mamba_cache_params.at_layer_idx(mamba_index),
                    mamba2_metadata=mamba2_metadata)
                mamba_index += 1

        if get_tensor_model_parallel_rank() == 1:
            create_stop_flag()

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        return FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts)

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # Skip loading extra parameters for GPTQ/modelopt models.
        ignore_suffixes = (".bias", "_bias", ".k_scale", "_k_scale",
                           ".v_scale", "_v_scale", ".weight_scale",
                           "_weight_scale", ".input_scale", "_input_scale")

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        expert_params_mapping = self.get_expert_mapping()
        for name, loaded_weight in weights:
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if "mlp.experts" in name:
                    continue
                name = name.replace(weight_name, param_name)

                # Skip loading extra parameters for GPTQ/modelopt models.
                if name.endswith(ignore_suffixes) and name not in params_dict:
                    continue

                # Skip layers on other devices.
                if is_pp_missing_parameter(name, self):
                    continue
                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    # Skip layers on other devices.
                    if is_pp_missing_parameter(name, self):
                        continue
                    # Skip loading extra parameters for GPTQ/modelopt models.
                    if name.endswith(
                            ignore_suffixes) and name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param,
                                  loaded_weight,
                                  name,
                                  shard_id=shard_id,
                                  expert_id=expert_id)
                    break
                else:
                    # Skip loading extra parameters for GPTQ/modelopt models.
                    if name.endswith(
                            ignore_suffixes) and name not in params_dict:
                        continue
                    # Skip layers on other devices.
                    if is_pp_missing_parameter(name, self):
                        continue
                    # Remapping the name of FP8 kv-scale.
                    if name.endswith("kv_scale"):
                        remapped_kv_scale_name = name.replace(
                            ".kv_scale", ".attn.kv_scale")
                        if remapped_kv_scale_name not in params_dict:
                            logger.warning_once(
                                "Found kv scale in the checkpoint (e.g. %s), but not found the expected name in the model (e.g. %s). kv-scale is not loaded.",  # noqa: E501
                                name,
                                remapped_kv_scale_name,
                            )
                            continue
                        else:
                            name = remapped_kv_scale_name
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class Qwen3MoeMamba2ForCausalLM(nn.Module, HasInnerState, IsHybrid, SupportsPP, SupportsLoRA):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.vllm_config = vllm_config
        self.config = config
        self.quant_config = quant_config
        self.model_config = vllm_config.model_config
        self.mamba_config = MambaConfig(**config.mamba_config)
        assert not vllm_config.cache_config.enable_prefix_caching, \
            "Mamba does not support prefix caching"

        self.model = Qwen3MoeMamba2Model(vllm_config=vllm_config,
                                   prefix=maybe_prefix(prefix, "model"))
        self.lm_head = ParallelLMHead(config.vocab_size,
                                      config.hidden_size,
                                      quant_config=quant_config)
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)
        self.mamba_cache: Optional[MambaCacheManager] = None

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Union[torch.Tensor, IntermediateTensors]:

        if self.mamba_cache is None:
            num_mamba_layers = self.model_config.get_num_layers_by_block_type(
                self.vllm_config.parallel_config, LayerBlockType.mamba)

            self.mamba_cache = MambaCacheManager(
                self.vllm_config, self.lm_head.weight.dtype, num_mamba_layers,
                *self._get_mamba_cache_shape())

        mamba_cache_params = self.mamba_cache.current_run_tensors(**kwargs)

        hidden_states = self.model(input_ids, positions, mamba_cache_params,
                                   intermediate_tensors, inputs_embeds)
        return hidden_states

    def _get_mamba_cache_shape(
            self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        world_size = get_tensor_model_parallel_world_size()

        conv_state_shape, temporal_state_shape = None, None

        if self.mamba_config.d_inner is None:
            intermediate_size = self.mamba_config.expand * self.mamba_config.d_model
        else:
            intermediate_size = self.mamba_config.d_inner

        # if n_groups is not divisible by world_size, need to extend the shards
        # to ensure all groups needed by a head is sharded along with it
        n_groups = (
            self.mamba_config.ngroups +
            extra_groups_for_head_shards(self.mamba_config.ngroups, world_size))

        # - heads and n_groups are TP-ed
        conv_dim = (intermediate_size + 2 * self.mamba_config.d_xb)
        conv_state_shape = (
            divide(conv_dim, world_size),
            self.mamba_config.d_conv - 1,
        )

        # These are not TP-ed as they depend on A, dt_bias, D
        # - they are typically small
        #   e.g., (h_heads, d_head, d_state) = (128, 64, 128)
        temporal_state_shape = (
            divide(self.mamba_config.ngroups, world_size),
            self.mamba_config.d_model * self.mamba_config.expand // self.mamba_config.ngroups,
            self.mamba_config.d_state,
        )
        return conv_state_shape, temporal_state_shape

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()
