import math

import paddle
import paddle.nn.functional as F

from paddle import nn
from paddle.incubate.nn.functional import swiglu
from paddle.distributed.fleet.meta_parallel import get_rng_state_tracker


def prepare_casual_attention_mask(batch_size, dtype, config):
    mask = paddle.tril(paddle.ones((config.seq_length, config.seq_length), dtype="bool"))
    mask = mask[None, None, :, :].expand([batch_size, 1, config.seq_length, config.seq_length])
    mask = paddle.where(mask, 0.0, paddle.finfo(dtype).min).astype(dtype)
    mask.stop_gradient = True
    return mask

def scaled_dot_product_attention(query_states, key_states, value_states, attention_mask):
    bsz, q_len, num_heads, head_dim = query_states.shape
    
    # [bsz, seq_len, head_num, head_dim] -> [bsz, head_num, seq_len, head_dim]
    query_states = paddle.transpose(query_states, [0, 2, 1, 3])
    key_states = paddle.transpose(key_states, [0, 2, 1, 3])
    value_states = paddle.transpose(value_states, [0, 2, 1, 3])
    
    attn_weights = paddle.matmul(query_states / math.sqrt(head_dim), key_states.transpose([0, 1, 3, 2]))
    attn_weights += attention_mask
    with paddle.amp.auto_cast(False):
        attn_weights = F.softmax(attn_weights, axis=-1, dtype="float32").astype(query_states.dtype)
    attn_output = paddle.matmul(attn_weights, value_states)
    
    # [bsz, head_num, seq_len, head_dim] -> [bsz, seq_len, head_num, head_dim]
    attn_output = attn_output.transpose([0, 2, 1, 3])
    # [bsz, seq_len, head_num, head_dim] -> [bsz, seq_len, head_num * head_dim]
    attn_output = attn_output.reshape([bsz, q_len, head_dim * num_heads])
    return attn_output

class LlamaRMSNorm(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.weight = paddle.create_parameter(
            shape=[self.hidden_size],
            dtype=paddle.get_default_dtype())
        self.variance_epsilon = config.rms_norm_eps
        self.config = config

    def forward(self, hidden_states):
        with paddle.amp.auto_cast(False):
            hidden_states = hidden_states.astype("float32")
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = paddle.rsqrt(variance + self.variance_epsilon) * hidden_states

        if self.weight.dtype in [paddle.float16, paddle.bfloat16]:
            hidden_states = paddle.cast(hidden_states, self.weight.dtype)
        return hidden_states * self.weight

class LlamaMLP(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias_attr=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias_attr=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias_attr=False)

    def forward(self, x):
        x = swiglu(self.gate_proj(x), self.up_proj(x))
        out = self.down_proj(x)
        return out


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
        
    def forward(
        self,
        hidden_states,    # [bs, seq_len, num_head * head_dim]
        attention_mask
    ):
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        target_query_shape = [0, 0, self.num_heads, self.head_dim]
        target_key_value_shape = [0, 0, self.num_key_value_heads, self.head_dim]
        query_states = query_states.reshape(shape=target_query_shape)   # [bs, seq_len, num_head, head_dim]
        key_states = key_states.reshape(shape=target_key_value_shape)
        value_states = value_states.reshape(shape=target_key_value_shape)
        
        attn_output = scaled_dot_product_attention(query_states, key_states, value_states, attention_mask)  # [bs, seq_len, num_head * head_dim]
        
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

class LlamaLMHead(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        vocab_size = config.vocab_size
        self.weight = self.create_parameter(shape=[config.hidden_size, vocab_size])
        
    def forward(self, hidden_states):
        logits = paddle.matmul(hidden_states, self.weight, transpose_y=False)
        return logits

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
            masked_lm_loss = masked_lm_loss[masked_lm_loss > 0]
            loss = paddle.mean(masked_lm_loss)
        return loss
            
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
        input_ids,          # [bs, seq_len]
        labels,             # [bs, seq_len]
    ):  
        hidden_states = self.embed_tokens(input_ids)    # [bs, seq_len, hidden_size]
        attention_mask = prepare_casual_attention_mask(hidden_states.shape[0], hidden_states.dtype, self.config)    ## [bs, 1, seq_len, seq_len]
        for _, (decoder_layer) in enumerate(self.layers):
            hidden_states = decoder_layer(hidden_states, attention_mask)
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)            # [bs, seq_len, vocab_size]
        loss = self.criterion(logits, labels)
        return loss
    
