""" PyTorch Mixtral model."""
import inspect
import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from collections import OrderedDict

from transformers import PreTrainedModel

from config import MixtralConfig

class ClassInstantier(OrderedDict):
    def __getitem__(self, key):
        content = super().__getitem__(key)
        cls, kwargs = content if isinstance(content, tuple) else (content, {})
        return cls(**kwargs)

ACT2CLS = {
    "relu": nn.ReLU,
    "relu6": nn.ReLU6,
    "sigmoid": nn.Sigmoid,
    "silu": nn.SiLU,
    "swish": nn.SiLU,
    "tanh": nn.Tanh,
}
ACT2FN = ClassInstantier(ACT2CLS)



_CONFIG_FOR_DOC = "MixtralConfig"



class MixtralBlockSparseTop2MLP(nn.Module):
    def __init__(self, config: MixtralConfig):
        super().__init__()
        self.ffn_dim = config.intermediate_size
        self.hidden_dim = config.hidden_size

        self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.w2 = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
        self.w3 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)

        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        current_hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
        current_hidden_states = self.w2(current_hidden_states)
        return current_hidden_states


class MixtralBLockSparseTop2MLP(MixtralBlockSparseTop2MLP):
    def __init__(self, *args, **kwargs):
        logger.warning_once(
            "MixtralBLockSparseTop2MLP is deprecated by MixtralBlockSparseTop2MLP and will be removed in v4.40."
        )
        super().__init__(*args, **kwargs)


class MixtralSparseMoeBlock(nn.Module):
    """
    This implementation is
    strictly equivalent to standard MoE with full capacity (no
    dropped tokens). It's faster since it formulates MoE operations
    in terms of block-sparse operations to accomodate imbalanced
    assignments of tokens to experts, whereas standard MoE either
    (1) drop tokens at the cost of reduced performance or (2) set
    capacity factor to number of experts and thus waste computation
    and memory on padding.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok

        # gating
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)

        self.experts = nn.ModuleList([MixtralBlockSparseTop2MLP(config) for _ in range(self.num_experts)])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        # [20, 512]
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        #  [20,512] * [512,8]   --->   [20,8] 
        router_logits = self.gate(hidden_states)
        # print(f"router_logits is {router_logits}") 
        # print(f"router_logits1 shape is {router_logits.shape}")
        # [20,8]
        '''
            每个token对应的专家的概率
        '''
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        # print(f"router_logits2 shape is {router_logits.shape}")
        print(f"router_logits3  is {routing_weights}")
        # [20,2]
        # selected_experts对应每个token选的专家在[0,7]的索引 它是有梯度的
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        print(f"router_logits3 shape is {routing_weights.shape}")
        print(f"router_logits3  is {routing_weights}")

        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # print(f"router_logits4 shape is {router_logits.shape}")
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)
        # print(f"router_logits5 shape is {router_logits.shape}")

        # [20,512] 全0矩阵
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        # [8,2,20]
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            # idx是两行中的哪一行 top_x是每列的哪一列 参考test.ipynb文件说明
            idx, top_x = torch.where(expert_mask[expert_idx])

            # 
            if top_x.shape[0] == 0:
                continue

            # in torch it is faster to index using lists than torch tensors
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            # 这里相当于只取top_x_list的值，hidden_states是词向量[20,512],current_state只取专家0算的token(假设是5)--->[5,512] 专家0负责了5个token的计算
            current_state = hidden_states[None, top_x_list].reshape(-1, hidden_dim)
            # routing_weights是路由的参数(门控) 
            # 以第0个专家举例: 这里是上面算出来5个token的词向量送入的把专家0(就是个线性层) 再乘以  路由的权重 routing_weights->[20,2]
            print(f"routing_weights666{routing_weights.shape}")                      
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x_list, idx_list, None]
            print(f"专家:{expert_idx}对应的索引是-> {top_x}")
            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            # 沿着batch维度,添加的索引,current_hidden_states的索引对应的值添加到final_hidden_states
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        # print(f"router_logits6 shape is {router_logits.shape}")
        return final_hidden_states, router_logits



if __name__ == '__main__':
    MixConfig = MixtralConfig(vocab_size = 2048,hidden_size=512,intermediate_size = 14336 // 2)
    mix_block = MixtralSparseMoeBlock(config=MixConfig)

    batch_size,seq_length,hidden_dim = 2, 10, 512
    final_hidden_states = torch.randn(batch_size,seq_length,hidden_dim)

    final_hidden_states, router_logits = mix_block(final_hidden_states)
    print(final_hidden_states.shape,router_logits.shape)
