
from no_parallel_train import (
    DefaultConfig, get_simple_optimizer, RandomDataset
)
from models.no_parallel_model import SimpleLlama

from dataclasses import dataclass, field
from typing import List, Optional

from paddle.io import Sampler

from paddlenlp.trainer import (
    PdArgumentParser,
    Trainer,
    TrainingArguments
)

@dataclass
class MyArguments:
    vocab_size: int = field(
        default=100, metadata={"help": "This is for XXX"}
    )
    hidden_size: int = field(
        default=256, metadata={"help": "This is for XXX"}
    )
    num_attention_heads: int = field(
        default=8, metadata={"help": "This is for XXX"}
    )
    num_key_value_heads: int = field(
        default=8, metadata={"help": "This is for XXX"}
    )
    num_hidden_layers:int = field(
        default=12, metadata={"help": "This is for XXX"}
    )
    intermediate_size: int = field(
        default=512, metadata={"help": "This is for XXX"}
    )
    epoch: int = field(
        default=1, metadata={"help": "This is for XXX"}
    )
    batch_size: int = field(
        default=2, metadata={"help": "This is for XXX"}
    )
    batch_num: int = field(
        default=5, metadata={"help": "This is for XXX"}
    )
    seq_length: int = field(
        default=512, metadata={"help": "This is for XXX"}
    )


def main():
    parser = PdArgumentParser((MyArguments, TrainingArguments))
    my_args, training_args = parser.parse_args_into_dataclasses()
    
    config = DefaultConfig()
    config.vocab_size = my_args.vocab_size
    config.hidden_size = my_args.hidden_size
    config.num_attention_heads = my_args.num_attention_heads
    config.num_key_value_heads = my_args.num_key_value_heads
    config.num_hidden_layers = my_args.num_hidden_layers
    config.intermediate_size = my_args.intermediate_size
    config.seq_length = my_args.seq_length
    
    training_args.per_device_train_batch_size = my_args.batch_size
    
    model = SimpleLlama(config)
    optimizer = get_simple_optimizer(config, parameter_list=model.parameters())
    dataset = RandomDataset(config)
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset, optimizers=(optimizer, None))
    _ = trainer.train()
    
if __name__ == "__main__":
    main()