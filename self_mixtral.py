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


# ================================== mixtral block==================================
_CONFIG_FOR_DOC = "MixtralConfig"

class ClassInstantier(OrderedDict):
    def __getitem__(self, key):
        content = super().__getitem__(key)
        cls, kwargs = content if isinstance(content, tuple) else (content,{})
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

class MixtralBlockSparseTop2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ffn_dim = config.intermediate_size
        self.hidden_dim = config.hidden_size

        self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim, bias =False)
        self.w2 = nn.Linear(self.ffn_dim, self.hidden_dim, bias = False)
        self.w3 = nn.Linear(self.hidden_dim, self.ffn_dim, bias = False)

        self.act_fn = ACT2FN[config.hidden_act]


    def forward(self,hidden_states):
        # 过w1
        x = self.act_fn(self.w1(hidden_states) * self.w3(hidden_states))
        x = self.w2(x)
        return x

class MixtralBLockSparseTop2MLP(MixtralBlockSparseTop2MLP):
    def __init__(self, *args, **kwargs):
        print(f'"MixtralBLockSparseTop2MLP is deprecated by MixtralBlockSparseTop2MLP and will be removed in v4.40."')
        super().__init__(*args, **kwargs)
    
class MixtralSparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok    
        self.gate = nn.Linear(self.hidden_dim,self.num_experts,bias=False)
        self.experts = nn.ModuleList([MixtralBLockSparseTop2MLP(config) for _ in range(self.num_experts)])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 取出shape  [batch,seq_len,hidden_dim] 假设是 [2,10,512]
        batch_size, seq_len, hidden_dim = hidden_states.shape
        # batch和seq_len变一个维度 [b,s,h] -> [b*s,h] [20,512]
        hidden_states = hidden_states.view(-1, hidden_dim)
        '''
            gate shape: [hidden_dim,num_experts] [512,8]
            [b*s,h] -> [b*s,h]   [20,8] 每个token都对应8个expert 
        '''
        router_logits = self.gate(hidden_states)
        # softmax 注意是在专家维度，路由到8个专家的概率值被归一化 shape: [20,8]
        routing_weights = F.softmax(router_logits, dim = 1,dtype=torch.float)
        
        # 取出前两个概率值高的expert 还是在专家维度选两个 shape: [20,2]
        # routing_weights: [20,2] seleted_experts: [20,2] topk这个函数是不连续的说明不可导 但是取出来的routing_weights是有梯度的，bp的时候更新这个参数
        routing_weights,seleted_experts = torch.topk(routing_weights,self.top_k,dim=-1)
        
        # 对weight进行平均求和 routing_weights: [20,2]
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True).to(hidden_states.dtype)
        # 初始化矩阵 final_hidden_states: [20,512]
        final_hidden_states = torch.zeros(
            (batch_size * seq_len, hidden_dim),dtype=hidden_states.dtype
        )
        # 获取mask 为了计算效率,具体做法是遍历每个专家，把每个专家负责的token一次计算完
        # seleted_experts: [20,2] 此时one_hot加了个维度8 --> [20,2,8] 由于我们需要遍历专家进行计算--> [8,2,20]
        expert_mask = nn.functional.one_hot(seleted_experts,num_classes=self.num_experts)
        # [20,2,8] ---> [8,2,20]
        expert_mask = expert_mask.permute(2,1,0)
        # 开始遍历专家
        for expert_idx in range(self.num_experts):
            # 取出索引为expert_idx的专家,self.experts是个list
            expert_layer = self.experts[expert_idx]
            # 通过where函数拿到需要被第0个专家计算的token的索引 注意我们这里假设是20个token，idx代表的是20个token的索引
            # 比如20个token中，第5、7、9个token被选中了，以及top_x是代表idx在哪一行，前面我们是取了top2个专家出来
            idx,top_x = torch.where(expert_mask[expert_idx])

            # 转为list
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()

            # 把原始hidden_states的值通过top_x_list取出来，可以把hidden_states理解成词向量: [20,512]
            # 注意这里用None加了一个维度，实际还是从 [20,512] 20里面取，也就是取哪一个token
            current_state = hidden_states[None,top_x_list].reshape(-1,hidden_dim)
            # 注意这里把原始词向量的某些token的词向量取出来，过了专家层再与路由权重相乘
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x_list,idx_list]
            # 添加到final_hidden_states 第0个维度 索引 添加的权重
            final_hidden_states.index_add_(0,top_x,current_hidden_states)
        final_hidden_states.reshape(batch_size,seq_len,hidden_dim)
        return final_hidden_states,router_logits
