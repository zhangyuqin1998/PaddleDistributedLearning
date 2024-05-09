from paddle.distributed import fleet

from no_parallel_pretrain import (
    get_simple_optimizer, create_pretrained_dataset, set_seed
)
from models.paddlenlp_model import SimpleLlama, ModelConfig
from models.paddlenlp_model_pp import SimpleLlamaPipe, ModelConfigPipe

from dataclasses import dataclass, field

from paddlenlp.transformers import AutoTokenizer
from paddlenlp.trainer import (
    PdArgumentParser,
    Trainer,
    TrainingArguments
)

@dataclass
class ModelArguments:
    hidden_size: int = field(default=256)
    num_attention_heads: int = field(default=8)
    num_key_value_heads: int = field(default=8)
    num_hidden_layers:int = field(default=12)
    intermediate_size: int = field(default=512)
    sequence_parallel: bool = field(default=False)

@dataclass
class DataArguments:
    split: str = field(default="949,50,1", metadata={"help": "Train/valid/test data split."})
    max_seq_length: int = field(default=1024)
    data_impl: str = field(default="mmap", metadata={"help": "The format of the preprocessed data."})
    
def main():
    parser = PdArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    data_file = ["/work/PaddleNLP/llm/llama/data/llama_openwebtext_100k"]
    
    tokenizer = AutoTokenizer.from_pretrained("facebook/llama-7b")

    if training_args.pipeline_parallel_degree > 1:
        config = ModelConfigPipe()
    else:
        config = ModelConfig()
    config.vocab_size = tokenizer.vocab_size
    config.hidden_size = model_args.hidden_size
    config.num_attention_heads = model_args.num_attention_heads
    config.num_key_value_heads = model_args.num_key_value_heads
    config.num_hidden_layers = model_args.num_hidden_layers
    config.intermediate_size = model_args.intermediate_size
    config.sequence_parallel = model_args.sequence_parallel
    config.tensor_parallel_degree = training_args.tensor_parallel_degree
    config.segment_parallel_degree = training_args.sep_parallel_degree

    set_seed(42)
    
    # 如果要使用分布式能力，就需要对这个模型进行完整的分布式改造，从mp到pp到sep
    if training_args.pipeline_parallel_degree > 1:
        cls = SimpleLlamaPipe
        hcg = fleet.get_hybrid_communicate_group()
        model = cls(config, num_stages=training_args.pipeline_parallel_degree, topology=hcg._topo)
    else:
        cls = SimpleLlama
        model = cls(config)

    optimizer = get_simple_optimizer(parameter_list=model.parameters())
    train_dataset, valid_dataset, test_dataset, data_collator = create_pretrained_dataset(data_args, training_args, data_file)
    
    # model.save_pretrained(training_args.output_dir)
    # model = cls.from_pretrained(training_args.output_dir)
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        optimizers=(optimizer, None)
    )
    
    _ = trainer.train()
    
if __name__ == "__main__":
    main()