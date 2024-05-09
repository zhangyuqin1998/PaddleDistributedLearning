import paddle

from paddle import nn
from paddle.incubate.nn.functional import swiglu
from paddle.distributed import fleet
from paddle.distributed.fleet.utils.sequence_parallel_utils import (
        ColumnSequenceParallelLinear,
        GatherOp,
        RowSequenceParallelLinear,
        ScatterOp,
        mark_as_sequence_parallel_parameter,
    )
from paddle.distributed.fleet.layers.mpu.random import get_rng_state_tracker

from .no_parallel_model import (
    scaled_dot_product_attention, prepare_casual_attention_mask
)


class LlamaRMSNorm(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.weight = paddle.create_parameter(
            shape=[self.hidden_size],
            dtype=paddle.get_default_dtype())
        self.variance_epsilon = config.rms_norm_eps
        self.config = config

        if config.sequence_parallel:
            mark_as_sequence_parallel_parameter(self.weight)

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
        
        if config.sequence_parallel:
            ColumnParallelLinear = ColumnSequenceParallelLinear
            RowParallelLinear = RowSequenceParallelLinear
        else:
            ColumnParallelLinear = fleet.meta_parallel.ColumnParallelLinear
            RowParallelLinear = fleet.meta_parallel.RowParallelLinear
        
        if config.tensor_parallel_degree > 1:
            self.gate_proj = ColumnParallelLinear(
                self.hidden_size,
                self.intermediate_size,
                gather_output=False,
                has_bias=False,
            )
            self.up_proj = ColumnParallelLinear(
                self.hidden_size,
                self.intermediate_size,
                gather_output=False,
                has_bias=False,
            )
            self.down_proj = RowParallelLinear(
                self.intermediate_size,
                self.hidden_size,
                input_is_parallel=True,
                has_bias=False,
            )
        else:
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
        
        if config.sequence_parallel:
            ColumnParallelLinear = ColumnSequenceParallelLinear
            RowParallelLinear = RowSequenceParallelLinear
        else:
            ColumnParallelLinear = fleet.meta_parallel.ColumnParallelLinear
            RowParallelLinear = fleet.meta_parallel.RowParallelLinear
        
        if config.tensor_parallel_degree > 1:
            self.q_proj = ColumnParallelLinear(
                    self.hidden_size,
                    self.hidden_size,
                    has_bias=False,
                    gather_output=False,
                )
            self.k_proj = ColumnParallelLinear(
                    self.hidden_size,
                    self.config.num_key_value_heads * self.head_dim,
                    has_bias=False,
                    gather_output=False,
                )
            self.v_proj = ColumnParallelLinear(
                self.hidden_size,
                self.config.num_key_value_heads * self.head_dim,
                has_bias=False,
                gather_output=False,
            )
            self.o_proj = RowParallelLinear(
                self.hidden_size,
                self.hidden_size,
                has_bias=False,
                input_is_parallel=True,
            )
        else:
            self.q_proj = nn.Linear(self.hidden_size, self.config.num_key_value_heads * self.head_dim, bias_attr=False)
            self.k_proj = nn.Linear(self.hidden_size, self.config.num_key_value_heads * self.head_dim, bias_attr=False)
            self.v_proj = nn.Linear(self.hidden_size, self.config.num_key_value_heads * self.head_dim, bias_attr=False)
            self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias_attr=False)
            
            
        if config.tensor_parallel_degree > 1:
            self.num_heads = self.num_heads // config.tensor_parallel_degree
            self.num_key_value_heads = config.num_key_value_heads // config.tensor_parallel_degree
        
    def forward(
        self,
        hidden_states,  # [bs * seq_len // mp_degree, num_head * head_dim ]
        attention_mask
    ):
        query_states = self.q_proj(hidden_states)   # [bs * seq_len, num_head * head_dim // mp_degree]
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        if self.config.sequence_parallel:
            # [bs * seq_len, num_head * head_dim // mp_degree] -> [bs, seq_len, num_head, head_dim // mp_degree]
            target_query_shape = [-1, self.config.seq_length, self.num_heads, self.head_dim]
            target_key_value_shape = [-1, self.config.seq_length, self.num_key_value_heads, self.head_dim]
        else:
            target_query_shape = [0, 0, self.num_heads, self.head_dim]
            target_key_value_shape = [0, 0, self.num_key_value_heads, self.head_dim]   # [bs, seq_len, num_head // mp_degree, head_dim]
        query_states = query_states.reshape(shape=target_query_shape)
        key_states = key_states.reshape(shape=target_key_value_shape)
        value_states = value_states.reshape(shape=target_key_value_shape)
        
        attn_output = scaled_dot_product_attention(query_states, key_states, value_states, attention_mask)  # [bs, seq_len, num_head * head_dim // mp_degree]
        
        if self.config.sequence_parallel:
            # [bs, seq_len, num_head * head_dim // mp_degree] -> [bs * seq_len, num_head * head_dim // mp_degree]
            attn_output = attn_output.reshape([query_states.shape[0] * self.config.seq_length, self.num_heads * self.head_dim])

        attn_output = self.o_proj(attn_output)    # [bs * seq_len // mp_degree, num_head * head_dim]
        return attn_output

class LlamaLMHead(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if config.tensor_parallel_degree > 1:
            vocab_size = config.vocab_size // config.tensor_parallel_degree
        else:
            vocab_size = config.vocab_size

        if vocab_size != config.vocab_size:
            with get_rng_state_tracker().rng_state():
                self.weight = self.create_parameter(shape=[config.hidden_size, vocab_size])
        else:
            self.weight = self.create_parameter(shape=[config.hidden_size, vocab_size])
        
        if config.tensor_parallel_degree > 1:
            # 手动切分参数时, 需要设置is_distributed为True, distributed_model中广播参数时就会把这些切分的参数跳过
            self.weight.is_distributed = True
            self.weight.split_axis = 1  # 用于正确保存分布式模型
        
    def forward(self, hidden_states):
        if self.config.sequence_parallel:
            hidden_states = GatherOp.apply(hidden_states)
            seq_length = self.config.seq_length
            hidden_states = paddle.reshape_(hidden_states, [-1, seq_length, self.config.hidden_size])

        if self.config.tensor_parallel_degree > 1:
            hcg = fleet.get_hybrid_communicate_group()
            hidden_states = paddle.distributed.collective._c_identity(hidden_states, group=hcg.get_model_parallel_group())  # 正向什么也不做, 反向时做all-reduce
        logits = paddle.matmul(hidden_states, self.weight, transpose_y=False)
        return logits
    
    
class LlamaPretrainingCriterion(paddle.nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config

        if config.tensor_parallel_degree > 1:
            self.loss_func = fleet.meta_parallel.ParallelCrossEntropy()
        else:
            self.loss_func = paddle.nn.CrossEntropyLoss(reduction="none")

    def forward(self, prediction_scores, masked_lm_labels):
        with paddle.amp.auto_cast(False):
            masked_lm_loss = self.loss_func(prediction_scores.astype("float32"), masked_lm_labels.unsqueeze(2))
            masked_lm_loss = masked_lm_loss[masked_lm_loss > 0]
            loss = paddle.mean(masked_lm_loss)

        return loss
    

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


class SimpleLlama(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if config.tensor_parallel_degree > 1 and config.vocab_size % config.tensor_parallel_degree == 0:
            self.embed_tokens = fleet.meta_parallel.VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size
            )
        else:
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

        if self.config.sequence_parallel:
            bs, seq_len, hidden_size = hidden_states.shape
            # [bs, seq_len, num_head * head_dim] -> [bs * seq_len, num_head * head_dim]
            hidden_states = paddle.reshape_(hidden_states, [bs * seq_len, hidden_size])
            # [bs * seq_len // mp_degree, num_head * head_dim]
            hidden_states = ScatterOp.apply(hidden_states)
        for _, (decoder_layer) in enumerate(self.layers):
            hidden_states = decoder_layer(hidden_states, attention_mask)
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)            # [bs, seq_len, vocab_size // mp_degree]
        loss = self.criterion(logits, labels)
        return loss