import paddle
from paddle.distributed import fleet
from paddle.io import DataLoader, DistributedBatchSampler

from paddlenlp.transformers import AutoTokenizer
from paddlenlp.transformers.segment_parallel_utils import split_inputs_sequence_dim
from paddlenlp.trainer import set_seed

from models.segment_parallel_model import SimpleLlama
from no_parallel_pretrain import (
    ModelConfig, DataConfig, TrainerConfig, get_simple_optimizer, create_pretrained_dataset
)

def print_rank_0(*args, **kwargs):
    if paddle.distributed.get_rank() == 0:
        print(*args, **kwargs)

class SegmentParallelTrainer:
    def __init__(self, model, config, data_collator, train_dataset, optimizer, data_parallel_degree=1, sharding_parallel_degree=1, segment_parallel_degree=1):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.train_dataset = train_dataset
        self.data_collator = data_collator
        self.data_parallel_degree = data_parallel_degree
        self.sharding_parallel_degree = sharding_parallel_degree
        
        self.segment_parallel_degree = segment_parallel_degree
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
                if self.segment_parallel_degree > 1:
                    input_ids = split_inputs_sequence_dim(input_ids)
                    labels = split_inputs_sequence_dim(labels)
                    
                loss = self.model(input_ids, labels)
                loss.backward()

                self.optimizer.step()
                self.model.clear_gradients()
                
                if global_step % self.config.logging_steps == 0:
                    print_rank_0(f"epoch: {eop}, global_step: {global_step}, loss: {loss.numpy()}")

dist_strategy = fleet.DistributedStrategy()
dist_strategy.hybrid_configs = {
    "sep_degree": 4,
    "dp_degree": 1,
}

# python -m paddle.distributed.launch --gpus=0,1,2,3 --log_dir logs segment_parallel_pretrain.py
if __name__ == "__main__":
    fleet.init(is_collective=True, strategy=dist_strategy)
    
    data_file = ["/work/PaddleNLP/llm/llama/data/llama_openwebtext_100k"]
    tokenizer = AutoTokenizer.from_pretrained("facebook/llama-7b")
    
    model_args, data_args, traing_args = ModelConfig(), DataConfig(), TrainerConfig()
    model_args.vocab_size = tokenizer.vocab_size
    model_args.segment_parallel_degree = dist_strategy.hybrid_configs["sep_degree"]

    set_seed(42)

    model = SimpleLlama(model_args)
    optimizer = get_simple_optimizer(parameter_list=model.parameters())
    train_dataset, valid_dataset, test_dataset, data_collator = create_pretrained_dataset(data_args, traing_args, data_file)

    model = fleet.distributed_model(model)
    optimizer = fleet.distributed_optimizer(optimizer)

    trainer = SegmentParallelTrainer(model, traing_args, data_collator, train_dataset, optimizer, segment_parallel_degree=dist_strategy.hybrid_configs["sep_degree"])
    trainer.train()