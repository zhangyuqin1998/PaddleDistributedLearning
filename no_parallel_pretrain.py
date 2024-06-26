import copy
import random
import numpy as np

import paddle
from paddle.distributed import fleet
from paddle.io import DataLoader, DistributedBatchSampler
from paddlenlp.transformers import AutoTokenizer
from paddlenlp.trainer import set_seed
from paddlenlp.data.causal_dataset import (
    build_train_valid_test_datasets,
    check_data_split,
)
from paddlenlp.datasets import load_dataset

from models.no_parallel_model import SimpleLlama


def create_pretrained_dataset(
    data_args,
    training_args, 
    data_file
):
    # check_data_split(config.split, config.do_train, config.do_eval, config.do_predict)
    
    train_val_test_num_samples = [
        training_args.per_device_train_batch_size
        * training_args.max_steps,
        0,
        0
    ]

    # Build the datasets.
    train_dataset, valid_dataset, test_dataset = build_train_valid_test_datasets(
        data_prefix=data_file,
        data_impl=data_args.data_impl,
        splits_string=data_args.split,
        train_val_test_num_samples=train_val_test_num_samples,
        seq_length=data_args.max_seq_length,
        seed=42,
        skip_warmup=False
    )

    from paddlenlp.data import Stack

    def _collate_data(data, stack_fn=Stack()):
        tokens_ = stack_fn([x["text"] for x in data])

        labels = copy.deepcopy(tokens_)[:, 1:]
        tokens = tokens_[:, :-1]

        return {
            "input_ids": tokens,
            "labels": labels,
        }

    return train_dataset, valid_dataset, test_dataset, _collate_data

def get_simple_optimizer(parameter_list=None):
    optimizer = paddle.optimizer.Momentum(
        learning_rate=0.1,
        momentum=0.9,
        weight_decay=paddle.regularizer.L2Decay(1e-4),
        parameters=parameter_list,
        grad_clip=paddle.nn.ClipGradByGlobalNorm(1.0)
    )
    return optimizer

class ModelConfig:
    def __init__(self):
        self.vocab_size = -1
        self.hidden_size = 256
        self.num_attention_heads = 8
        self.num_key_value_heads = 8
        self.num_hidden_layers = 12
        self.intermediate_size = 512
        self.rms_norm_eps = 1e-6
        
        self.seq_length = 1024
        self.tensor_parallel_degree = 1
        self.segment_parallel_degree = 1
        self.sequence_parallel = False


class DataConfig:
    def __init__(self):
        self.data_impl = "mmap"
        self.split = "949,50,1"
        self.max_seq_length = 1024
        
class TrainerConfig:
    def __init__(self):   
        self.num_train_epochs = 1
        self.per_device_train_batch_size = 4
        self.max_steps = 2000
        
        self.logging_steps = 10        
        # self.eval_iters = 10
        # self.test_iters = 100
        
        # self.do_train = True
        # self.do_eval = False
        # self.do_predict = False

def print_rank_0(*args, **kwargs):
    if paddle.distributed.get_rank() == 0:
        print(*args, **kwargs)


class SimpleTrainer:
    def __init__(self, model, config, data_collator, train_dataset, optimizer, data_parallel_degree=1, sharding_parallel_degree=1):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.train_dataset = train_dataset
        self.data_collator = data_collator
        self.data_parallel_degree = data_parallel_degree
        self.sharding_parallel_degree = sharding_parallel_degree
        
        try:
            hcg = fleet.get_hybrid_communicate_group()
            self.data_parallel_rank = max(hcg.get_data_parallel_group().rank, 0)
            self.sharding_parallel_rank = max(hcg.get_sharding_parallel_group().rank, 0)
        except:
            self.data_parallel_rank = 0
            self.sharding_parallel_rank = 0

    
    def get_dataset_rank(self):
        return max(self.sharding_parallel_degree, 1) * self.data_parallel_rank + self.sharding_parallel_rank

    def get_num_replicas(self):
        return max(self.sharding_parallel_degree, 1) * max(self.data_parallel_degree, 1)
        
    def train(self):
        global_step = 0
        sampler = DistributedBatchSampler(self.train_dataset, batch_size=self.config.per_device_train_batch_size, num_replicas=self.get_num_replicas(), rank=self.get_dataset_rank(), shuffle=False)

        train_loader = DataLoader(self.train_dataset, batch_sampler=sampler, collate_fn=self.data_collator)

        for eop in range(self.config.num_train_epochs):
            self.model.train()
            for batch_id, data in enumerate(train_loader()):
                global_step += 1
                if global_step == self.config.max_steps:
                    return
                input_ids, labels = data["input_ids"], data["labels"]
                loss = self.model(input_ids, labels)
                loss.backward()

                self.optimizer.step()
                self.model.clear_gradients()
                if global_step % self.config.logging_steps == 0:
                    print_rank_0(f"epoch: {eop}, global_step: {global_step}, loss: {loss.numpy()}")


# python no_parallel_train.py
if __name__ == "__main__":
    data_file = ["/work/PaddleNLP/llm/llama/data/llama_openwebtext_100k"]
    tokenizer = AutoTokenizer.from_pretrained("facebook/llama-7b")  # 因为数据集已经被tokenizer处理过了，所以这里的tokenizer其实没用到，只是为了拿到vocab_size
    
    model_args, data_args, traing_args = ModelConfig(), DataConfig(), TrainerConfig()
    model_args.vocab_size = tokenizer.vocab_size
    
    set_seed(42)
    
    model = SimpleLlama(model_args)
    optimizer = get_simple_optimizer(parameter_list=model.parameters())
    train_dataset, valid_dataset, test_dataset, data_collator = create_pretrained_dataset(data_args, traing_args, data_file)

    trainer = SimpleTrainer(model, traing_args, data_collator, train_dataset, optimizer)
    trainer.train()