import math

import paddle
import paddle.nn.functional as F
from paddle import nn
from paddle.distributed import fleet
from paddle.autograd import PyLayer

from paddlenlp.transformers.segment_parallel_utils import ReshardLayer
from .no_parallel_model import (
    LlamaRMSNorm, LlamaLMHead, LlamaMLP, prepare_casual_attention_mask
)


def scaled_dot_product_attention(query_states, key_states, value_states, attention_mask, segment_parallel_degree, reshard_layer):
    bsz, q_len, num_heads, head_dim = query_states.shape
    
    # [bsz, seq_len, head_num // sep_degree, head_dim] -> [bsz, head_num // sep_degree, seq_len, head_dim]
    query_states = paddle.transpose(query_states, [0, 2, 1, 3])
    key_states = paddle.transpose(key_states, [0, 2, 1, 3])
    value_states = paddle.transpose(value_states, [0, 2, 1, 3])
    
    attn_weights = paddle.matmul(query_states / math.sqrt(head_dim), key_states.transpose([0, 1, 3, 2]))
    attn_weights += attention_mask
    with paddle.amp.auto_cast(False):
        attn_weights = F.softmax(attn_weights, axis=-1, dtype="float32").astype(query_states.dtype)
    attn_output = paddle.matmul(attn_weights, value_states)
    
    # [bsz, head_num // sep_degree, seq_len, head_dim] -> [bsz, seq_len, head_num // sep_degree, head_dim]
    attn_output = attn_output.transpose([0, 2, 1, 3])
    
    
    if reshard_layer is not None:
        # [bsz, seq_len, head_num // sep_degree, head_dim] -> [bsz, seq_len // sep_degree, head_num, head_dim]
        attn_output = reshard_layer(
            attn_output,
            split_axis=1,
            concat_axis=2,
        )
        q_len = q_len // segment_parallel_degree
        num_heads = num_heads * segment_parallel_degree

    # -> [bsz, seq_len // sep_degree, head_num * head_dim]
    attn_output = attn_output.reshape([bsz, q_len, head_dim * num_heads])
    return attn_output


class LlamaAttention(nn.Layer):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        
        self.q_proj = nn.Linear(self.hidden_size, self.config.num_key_value_heads * self.head_dim, bias_attr=False)
        self.k_proj = nn.Linear(self.hidden_size, self.config.num_key_value_heads * self.head_dim, bias_attr=False)
        self.v_proj = nn.Linear(self.hidden_size, self.config.num_key_value_heads * self.head_dim, bias_attr=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias_attr=False)
        
        if config.segment_parallel_degree > 1:
            assert self.num_key_value_heads % config.segment_parallel_degree == 0
            assert self.num_heads % config.segment_parallel_degree == 0
            self.reshard_layer = ReshardLayer()
        else:
            self.reshard_layer = None
        
    def forward(
        self,
        hidden_states,    # [bs, seq_len // sep_degree, num_head * head_dim]
        attention_mask
    ):  
        query_states = self.q_proj(hidden_states)   # [bs, seq_len // sep_degree, num_head * head_dim]
        key_states = self.k_proj(hidden_states) 
        value_states = self.v_proj(hidden_states)
        if self.reshard_layer is not None:
            # [bs, seq_len // sep_degree, num_head * head_dim] -> [bs, seq_len, num_head * head_dim // sep_degree]
            query_states = self.reshard_layer(
                query_states,
                split_axis=2,
                concat_axis=1,
            )
            key_states = self.reshard_layer(
                key_states,
                split_axis=2,
                concat_axis=1,
            )
            value_states = self.reshard_layer(
                value_states,
                split_axis=2,
                concat_axis=1,
            )
            
            # [bs, seq_len, num_head * head_dim // sep_degree] -> [bs, seq_len, num_head // sep_degree, head_dim]
            query_states = paddle.reshape(query_states, [0, self.config.seq_length, -1, self.head_dim])
            key_states = paddle.reshape(key_states, [0, self.config.seq_length, -1, self.head_dim])
            value_states = paddle.reshape(value_states, [0, self.config.seq_length, -1, self.head_dim])
        else:
            target_query_shape = [0, 0, self.num_heads, self.head_dim]
            target_key_value_shape = [0, 0, self.num_key_value_heads, self.head_dim]
            query_states = query_states.reshape(shape=target_query_shape)   # [bs, seq_len, num_head, head_dim]
            key_states = key_states.reshape(shape=target_key_value_shape)
            value_states = value_states.reshape(shape=target_key_value_shape)
        attn_output = scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attention_mask,
            self.config.segment_parallel_degree,
            self.reshard_layer
        )  # [bs, seq_len, num_head * head_dim]
        attn_output = self.o_proj(attn_output)
        return attn_output


class LlamaDecoderLayer(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config)
        self.post_attention_layernorm = LlamaRMSNorm(config)

    def forward(
        self,
        hidden_states,
        attention_mask
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        outputs = self.self_attn(hidden_states, attention_mask)
        hidden_states = residual + outputs

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        outputs = residual + hidden_states
        return outputs
    

class LlamaPretrainingCriterion(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.loss_func = paddle.nn.CrossEntropyLoss(reduction="none")
        
    def forward(self, prediction_scores, masked_lm_labels):
        with paddle.amp.auto_cast(False):
            # prediction_scores: [bsz, seq_len, vocab_size]
            # masked_lm_labels: [bsz, seq_len]
            masked_lm_loss = self.loss_func(prediction_scores.astype("float32"), masked_lm_labels.unsqueeze(2))
            if self.config.segment_parallel_degree > 1:
                _hcg = fleet.get_hybrid_communicate_group()
                masked_lm_loss = ConcatMaskedLoss.apply(masked_lm_loss, axis=1, group=_hcg.get_sep_parallel_group())

            masked_lm_loss = masked_lm_loss[masked_lm_loss > 0]
            
            loss = paddle.mean(masked_lm_loss)
        return loss


class ConcatMaskedLoss(PyLayer):
    @staticmethod
    def forward(ctx, inp, axis, group):
        inputs = []
        paddle.distributed.all_gather(inputs, inp, group=group)
        with paddle.no_grad():
            cat = paddle.concat(inputs, axis=axis)
        ctx.args_axis = axis
        ctx.args_group = group
        return cat

    @staticmethod
    def backward(ctx, grad):
        axis = ctx.args_axis
        group = ctx.args_group
        with paddle.no_grad():
            grads = paddle.split(grad, paddle.distributed.get_world_size(group), axis=axis)
        grad = grads[paddle.distributed.get_rank(group)]
        return grad

class SimpleLlama(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.layers = nn.LayerList(
            [LlamaDecoderLayer(config) for i in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config)
        self.lm_head = LlamaLMHead(config)
        self.criterion = LlamaPretrainingCriterion(config)
     
    def forward(
        self,
        input_ids,          # [bs, seq_len // sep_degree]
        labels,             # [bs, seq_len // sep_degree]
    ):  
        hidden_states = self.embed_tokens(input_ids)    # [bs, seq_len // sep_degree, hidden_size]
        attention_mask = prepare_casual_attention_mask(hidden_states.shape[0], hidden_states.dtype, self.config)    ## [bs, 1, seq_len, seq_len]
        for _, (decoder_layer) in enumerate(self.layers):
            hidden_states = decoder_layer(hidden_states, attention_mask)
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)            # [bs, seq_len // sep_degree, vocab_size]
        loss = self.criterion(logits, labels)
        return loss