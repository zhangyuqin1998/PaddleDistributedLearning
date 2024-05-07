import copy
import random
import numpy as np

import paddle
from paddle.distributed import fleet
from paddle.distributed.sharding import group_sharded_parallel

from paddlenlp.transformers import AutoTokenizer

from models.no_parallel_model import SimpleLlama
from no_parallel_pretrain import (
    ModelConfig, DataConfig, TrainerConfig, get_simple_optimizer, create_pretrained_dataset, set_seed
)
from data_parallel_pretrain import (
    DataParallelTrainer, print_rank_0
)

dist_strategy = fleet.DistributedStrategy()
dist_strategy.hybrid_configs = {
    "sharding_degree": 4,
}


# python -m paddle.distributed.launch --gpus=0,1,2,3 --log_dir logs sharding_parallel_pretrain.py
if __name__ == "__main__":
    fleet.init(is_collective=True, strategy=dist_strategy)
    
    data_file = ["/work/PaddleNLP/llm/llama/data/llama_openwebtext_100k"]
    tokenizer = AutoTokenizer.from_pretrained("facebook/llama-7b")
    
    model_args, data_args, traing_args = ModelConfig(), DataConfig(), TrainerConfig()
    traing_args.per_device_train_batch_size //= dist_strategy.hybrid_configs["sharding_degree"]
    model_args.vocab_size = tokenizer.vocab_size
    
    set_seed(42)
    
    model = SimpleLlama(model_args)
    optimizer = get_simple_optimizer(parameter_list=model.parameters())
    train_dataset, valid_dataset, test_dataset, data_collator = create_pretrained_dataset(data_args, traing_args, data_file)

    hcg = fleet.get_hybrid_communicate_group()
    # wrap GroupSharded model, optimizer and scaler. level1='os', level2='os_g', level3='p_g_os'
    model, optimizer, scaler = group_sharded_parallel(model, optimizer, level="p_g_os", group=hcg.get_sharding_parallel_group())

    trainer = DataParallelTrainer(model, traing_args, data_collator, train_dataset, optimizer, sharding_parallel_degree=dist_strategy.hybrid_configs["sharding_degree"])
    trainer.train()
