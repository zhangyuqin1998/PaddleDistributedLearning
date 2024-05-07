import paddle
from paddle.distributed import fleet
from paddle.distributed.fleet.utils import sequence_parallel_utils

from paddlenlp.transformers import AutoTokenizer
from paddlenlp.trainer import set_seed

from models.model_parallel_with_sequence_parallel_model import SimpleLlama
from no_parallel_pretrain import (
    ModelConfig, DataConfig, TrainerConfig, SimpleTrainer, get_simple_optimizer, create_pretrained_dataset
)


dist_strategy = fleet.DistributedStrategy()
dist_strategy.hybrid_configs = {
    "mp_degree": 4,
}


def print_rank_0(*args, **kwargs):
    if paddle.distributed.get_rank() == 0:
        print(*args, **kwargs)


# python -m paddle.distributed.launch --gpus=0,1,2,3 --log_dir logs model_parallel_with_sequence_parallel_pretrain.py
if __name__ == "__main__":
    fleet.init(is_collective=True, strategy=dist_strategy)
    
    data_file = ["/work/PaddleNLP/llm/llama/data/llama_openwebtext_100k"]
    tokenizer = AutoTokenizer.from_pretrained("facebook/llama-7b")
    
    model_args, data_args, traing_args = ModelConfig(), DataConfig(), TrainerConfig()
    model_args.vocab_size = tokenizer.vocab_size
    model_args.tensor_parallel_degree = dist_strategy.hybrid_configs["mp_degree"]
    model_args.sequence_parallel = True

    set_seed(42)

    model = SimpleLlama(model_args)
    optimizer = get_simple_optimizer(parameter_list=model.parameters())
    train_dataset, valid_dataset, test_dataset, data_collator = create_pretrained_dataset(data_args, traing_args, data_file)

    model = fleet.distributed_model(model)
    optimizer = fleet.distributed_optimizer(optimizer)

    if model_args.sequence_parallel:
        sequence_parallel_utils.register_sequence_parallel_allreduce_hooks(model, 1, False)
        
    trainer = SimpleTrainer(model, traing_args, data_collator, train_dataset, optimizer)
    trainer.train()