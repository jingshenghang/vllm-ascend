# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
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
# Adapted from vllm/model_executor/models/qwen3_moe.py
# This file is a part of the vllm-ascend project.

from typing import Optional, Union

import torch
from torch import nn
from transformers import PretrainedConfig
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.qwen3_moe import (Qwen3MoeAttention,
                                                  Qwen3MoeForCausalLM,
                                                  Qwen3MoeMLP, Qwen3MoeModel)
from vllm.model_executor.models.utils import (
    extract_layer_index, make_empty_intermediate_tensors_factory, make_layers,
    maybe_prefix)
from vllm.sequence import IntermediateTensors

from vllm_ascend.ops.fused_moe import AscendSparseMoeBlock
from vllm_ascend.ops.sequence_parallel import (MetadataForPadding,
                                               init_metadata_for_sp)
import os
import re
import torch

# 全局配置变量 - 集中管理路径
DEFAULT_BASE_DIR = "/home/ascend-vllm/mindspeed_vllm_tensor_align_0923"  # 默认基础目录
BASE_DIR = os.environ.get("VLLM_DEBUG_PATH", DEFAULT_BASE_DIR)  # 从环境变量获取，无则用默认
FLAG_FILENAME = os.path.join(BASE_DIR, "vllm_donot_write.txt")  # 停止标志文件路径

def check_stop_flag():
    """检查是否需要停止写入：文件存在且内容为1时返回True"""
    # 先检查文件是否存在
    if not os.path.exists(FLAG_FILENAME):
        return False
    
    # 读取文件内容并检查是否为1
    with open(FLAG_FILENAME, 'r') as f:
        content = f.read().strip()  # 去除空白字符和换行符
    
    return content == "1"


def create_stop_flag():

    debug_flag = os.environ.get('VLLM_IS_DEBUG', '0')
    
    # 仅当环境变量为'1'时执行逻辑
    if debug_flag != '1':
        return 0  # 非调试模式返回默认值0
    """
    创建或更新停止标志文件：
    - 不存在则创建并写入0
    - 存在且内容为0则更新为1
    - 存在且内容为1则不做操作
    """
    if os.path.exists(FLAG_FILENAME):
        # 文件存在，读取当前内容
        with open(FLAG_FILENAME, 'r') as f:
            content = f.read().strip()
        
        # 只有内容为0时才更新为1
        if content == "0":
            with open(FLAG_FILENAME, 'w') as f:
                f.write("1")
    else:
        # 文件不存在，创建并写入0
        with open(FLAG_FILENAME, 'w') as f:
            f.write("0")

def find_latest_sequence_number(name, rank_id):
    """
    查找当前文件夹中指定name和rank_id的最大序号
    name: 张量名称标识
    rank_id: 张量的rank标识
    """
    # 正则匹配 "vllm_{name}_rank_{rank_id}_数字.pt" 格式
    # 使用re.escape处理特殊字符，确保匹配准确性
    pattern = re.compile(
        f"^vllm_{re.escape(name)}_rank_{re.escape(str(rank_id))}_(\d+)\.pt$"
    )
    max_num = 0
    
    # 遍历基础目录下的所有文件
    for filename in os.listdir(BASE_DIR):
        file_path = os.path.join(BASE_DIR, filename)
        if os.path.isfile(file_path):
            match = pattern.match(filename)
            if match:
                num = int(match.group(1))
                if num > max_num:
                    max_num = num
    return max_num


def save_tensor_sequentially(tensor, name, rank_id, check_stop=True):
    """
    按name和rank_id分别递增保存张量，带停止标志检查
    - 若存在停止标志文件则停止写入
    - name: 张量名称，用于区分不同类型的张量
    - rank_id: 张量的rank标识，用于区分同类型张量的不同rank
    - 返回值: 保存的文件名，若未保存则返回None
    """
    
    debug_flag = os.environ.get('VLLM_IS_DEBUG', '0')
    
    # 仅当环境变量为'1'时执行逻辑
    if debug_flag != '1':
        return 0  # 非调试模式返回默认值0
    # 检查停止标志
    if check_stop and check_stop_flag():
        # print(f"检测到 {FLAG_FILENAME}，已停止写入")
        return None
    
    # 获取当前name和rank_id组合的最大序号并计算下一个序号
    latest_num = find_latest_sequence_number(name, rank_id)
    next_num = latest_num + 1
    
    # 生成新文件名并保存（格式：vllm_{name}_rank_{rank_id}_{num}.pt）
    new_filename = os.path.join(BASE_DIR, f"vllm_{name}_rank_{rank_id}_{next_num}.pt")
    torch.save(tensor.cpu(), new_filename)
    # print(f"已保存文件: {new_filename}")
    
    return new_filename


class AscendQwen3MoeDecoderLayer(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        vllm_config: Optional[VllmConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)
        self.self_attn = Qwen3MoeAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', False),
            head_dim=getattr(config, 'head_dim', None),
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )

        # `mlp_only_layers` in the config.
        layer_idx = extract_layer_index(prefix)
        mlp_only_layers = ([] if not hasattr(config, "mlp_only_layers") else
                           config.mlp_only_layers)
        if (layer_idx not in mlp_only_layers) and (
                config.num_experts > 0 and
            (layer_idx + 1) % config.decoder_sparse_step == 0):
            self.mlp = AscendSparseMoeBlock(config=config,
                                            quant_config=quant_config,
                                            prefix=f"{prefix}.mlp")
        else:
            self.mlp = Qwen3MoeMLP(hidden_size=config.hidden_size,
                                   intermediate_size=config.intermediate_size,
                                   hidden_act=config.hidden_act,
                                   quant_config=quant_config,
                                   prefix=f"{prefix}.mlp")
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

        self.enable_sequence_parallelism = (
            vllm_config.compilation_config.pass_config.
            enable_sequence_parallelism if vllm_config is not None else False)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        _metadata_for_padding: Optional[MetadataForPadding] = None,
    ) -> torch.Tensor:

        # To prevent precision issues during the decoder phase when only prefilling enables SP
        if not self.enable_sequence_parallelism:
            self.self_attn.o_proj.reduce_results = True
        else:
            self.self_attn.o_proj.reduce_results = not _metadata_for_padding.not_dummy_and_is_prefill if _metadata_for_padding is not None else True

        # Self Attention
        if residual is None:
            residual = hidden_states
            if _metadata_for_padding and _metadata_for_padding.not_dummy_and_is_prefill:
                residual = _metadata_for_padding.padding_slice(residual)

            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)

            if _metadata_for_padding and _metadata_for_padding.not_dummy_and_is_prefill:
                hidden_states = _metadata_for_padding.allgather_unpadding_aligned(
                    hidden_states)

        from vllm.distributed import (
            divide, get_tensor_model_parallel_world_size, get_tensor_model_parallel_rank,
            tensor_model_parallel_all_gather, tensor_model_parallel_all_reduce)

        save_tensor_sequentially(hidden_states, "before_attn", get_tensor_model_parallel_rank())

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        if _metadata_for_padding and _metadata_for_padding.not_dummy_and_is_prefill:
            hidden_states = _metadata_for_padding.padding_aligned_reduce_scatter(
                hidden_states)

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)

        save_tensor_sequentially(hidden_states, "before_mlp", get_tensor_model_parallel_rank())
        hidden_states = self.mlp(hidden_states,
                                 _metadata_for_padding=_metadata_for_padding)

        return hidden_states, residual


@support_torch_compile
class AscendQwen3MoeModel(Qwen3MoeModel):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=f"{prefix}.embed_tokens")
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: AscendQwen3MoeDecoderLayer(
                config=config,
                cache_config=cache_config,
                quant_config=quant_config,
                vllm_config=vllm_config,
                prefix=prefix),
            prefix=f"{prefix}.layers",
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        _metadata_for_padding: Optional[MetadataForPadding] = None,
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
        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
                _metadata_for_padding=_metadata_for_padding)
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })

        hidden_states, _ = self.norm(hidden_states, residual)

        if _metadata_for_padding and _metadata_for_padding.not_dummy_and_is_prefill:
            hidden_states = _metadata_for_padding.allgather_unpadding_aligned(
                hidden_states)

        return hidden_states


class CustomQwen3MoeForCausalLM(Qwen3MoeForCausalLM):
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
        "experts":
        ["experts.0.gate_proj", "experts.0.up_proj", "experts.0.down_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        SupportsPP.__init__(self)
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        self.model = AscendQwen3MoeModel(vllm_config=vllm_config,
                                         prefix=maybe_prefix(prefix, "model"))
        self.lm_head = ParallelLMHead(config.vocab_size,
                                      config.hidden_size,
                                      quant_config=quant_config,
                                      prefix=maybe_prefix(prefix, "lm_head"))
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)
        self.enable_sequence_parallelism = vllm_config.compilation_config.pass_config.enable_sequence_parallelism

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        _metadata_for_padding = init_metadata_for_sp(
            input_ids, self.enable_sequence_parallelism)
        hidden_states = self.model(input_ids, positions, intermediate_tensors,
                                   inputs_embeds, _metadata_for_padding)
        return hidden_states
