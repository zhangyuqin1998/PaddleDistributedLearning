from paddle import nn

from paddle.distributed.fleet.meta_parallel import (
    LayerDesc,
    PipelineLayer,
)
from paddle.distributed.fleet.layers.mpu.random import get_rng_state_tracker

from .no_parallel_model import (
    LlamaDecoderLayer,
    LlamaRMSNorm,
    LlamaLMHead,
    LlamaPretrainingCriterion,
    prepare_casual_attention_mask
)


class LlamaEmbeddingPipe(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
        )
    
    def forward(self, input_ids):
        hidden_states = self.embed_tokens(input_ids)
        # 如果只有1个返回值, 框架会把它变成一个直接的tensor
        # 如果有多个返回值, 框架会把他们变成一个tuple
        return hidden_states

class PrepareCasualAttentionMaskPipe(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
    def forward(self, input):
        hidden_states = input
        attention_mask = prepare_casual_attention_mask(hidden_states.shape[0], hidden_states.dtype, self.config)
        return hidden_states, attention_mask
    
    
class LlamaDecoderLayerPipe(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.layer = LlamaDecoderLayer(config)
    def forward(self, input):
        hidden_states, attention_mask = input[0], input[1]
        hidden_states = self.layer(hidden_states, attention_mask)
        return hidden_states, attention_mask

class LlamaLlamaRMSNormPipe(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.layer = LlamaRMSNorm(config)
        
    def forward(self, input):
        hidden_states, attention_mask = input[0], input[1]
        hidden_states = self.layer(hidden_states)
        return hidden_states

class SimpleLlamaPipe(PipelineLayer):
    def __init__(self, config, **kwargs):
        self.config = config
        decs = []
        decs.append(LayerDesc(LlamaEmbeddingPipe, config=config))
        decs.append(LayerDesc(PrepareCasualAttentionMaskPipe, config=config))
        for _ in range(config.num_hidden_layers):
            decs.append(LayerDesc(LlamaDecoderLayerPipe, config=config))
        decs.append(LayerDesc(LlamaLlamaRMSNormPipe, config=config))
        decs.append(LayerDesc(LlamaLMHead, config=config))
        PipelineLayer.__init__(self, layers=decs,loss_fn=LlamaPretrainingCriterion(config), **kwargs)