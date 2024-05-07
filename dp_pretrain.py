import copy
import random
import numpy as np

import paddle
from paddle.distributed import fleet
from paddle.io import DataLoader, BatchSampler
from paddlenlp.transformers import AutoTokenizer

from models.no_parallel_model import SimpleLlama
from no_parallel_pretrain import (
    ModelConfig, DataConfig, TrainerConfig, MyTrainer, get_simple_optimizer, create_pretrained_dataset, set_seed
)


dist_strategy = fleet.DistributedStrategy()
dist_strategy.hybrid_configs = {
    "dp_degree": 4,
    "mp_degree": 1,
    "pp_degree": 1,
}


# python -m paddle.distributed.launch --gpus=0,1,2,3 --log_dir logs dp_pretrain.py
if __name__ == "__main__":
    fleet.init(is_collective=True, strategy=dist_strategy)
    
    data_file = ["/work/PaddleNLP/llm/llama/data/llama_openwebtext_100k"]
    tokenizer = AutoTokenizer.from_pretrained("facebook/llama-7b")
    
    model_args, data_args, traing_args = ModelConfig(), DataConfig(), TrainerConfig()
    traing_args.per_device_train_batch_size //= dist_strategy.hybrid_configs["dp_degree"]
    model_args.vocab_size = tokenizer.vocab_size
    
    set_seed(42)
    
    model = SimpleLlama(model_args)
    optimizer = get_simple_optimizer(parameter_list=model.parameters())
    train_dataset, valid_dataset, test_dataset, data_collator = create_pretrained_dataset(data_args, traing_args, data_file)

    model = fleet.distributed_model(model)
    optimizer = fleet.distributed_optimizer(optimizer)

    trainer = MyTrainer(model, traing_args, data_collator, train_dataset, optimizer, dist_strategy.hybrid_configs["dp_degree"])
    trainer.train()