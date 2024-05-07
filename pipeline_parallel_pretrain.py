import paddle
from paddle.distributed import fleet
from paddle.distributed.fleet.utils import sequence_parallel_utils
from paddle.io import DataLoader, DistributedBatchSampler, BatchSampler

from paddlenlp.transformers import AutoTokenizer
from paddlenlp.trainer import set_seed

from models.pipeline_parallel_model import SimpleLlamaPipe
from no_parallel_pretrain import (
    ModelConfig, DataConfig, TrainerConfig, get_simple_optimizer, create_pretrained_dataset, print_rank_0
)


class PipelineTrainer:
    def __init__(self, model, config, data_collator, train_dataset, optimizer, data_parallel_degree=1, sharding_parallel_degree=1):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.train_dataset = train_dataset
        self.data_collator = data_collator
        self.data_parallel_degree = data_parallel_degree
        self.sharding_parallel_degree = sharding_parallel_degree
        
    def train(self):
        global_step = 0
        sampler = BatchSampler(self.train_dataset, batch_size=self.config.per_device_train_batch_size, shuffle=False)

        train_loader = DataLoader(self.train_dataset, batch_sampler=sampler, collate_fn=self.data_collator)

        for eop in range(self.config.num_train_epochs):
            self.model.train()
            for batch_id, data in enumerate(train_loader()):
                global_step += 1
                if global_step == self.config.max_steps:
                    return
                input_ids, labels = data["input_ids"], data["labels"]

                loss = self.model.train_batch([input_ids, labels], optimizer=self.optimizer)

                if global_step % self.config.logging_steps == 0:
                    print_rank_0(f"epoch: {eop}, global_step: {global_step}, loss: {loss.numpy()}")

dist_strategy = fleet.DistributedStrategy()
dist_strategy.hybrid_configs = {
    "dp_degree": 1,
    "mp_degree": 1,
    "pp_degree": 4,
}

dist_strategy.pipeline_configs = {
    "accumulate_steps": 1,
    "micro_batch_size": 4
}

# python -m paddle.distributed.launch --gpus=0,1,2,3 --log_dir logs pipeline_parallel_pretrain.py
if __name__ == "__main__":
    fleet.init(is_collective=True, strategy=dist_strategy)
    
    data_file = ["/work/PaddleNLP/llm/llama/data/llama_openwebtext_100k"]
    tokenizer = AutoTokenizer.from_pretrained("facebook/llama-7b")
    
    model_args, data_args, traing_args = ModelConfig(), DataConfig(), TrainerConfig()
    model_args.vocab_size = tokenizer.vocab_size

    set_seed(42)

    hcg = fleet.get_hybrid_communicate_group()
    model = SimpleLlamaPipe(model_args, num_stages=dist_strategy.hybrid_configs["pp_degree"], topology=hcg._topo)
    optimizer = get_simple_optimizer(parameter_list=model.parameters())
    train_dataset, valid_dataset, test_dataset, data_collator = create_pretrained_dataset(data_args, traing_args, data_file)

    model = fleet.distributed_model(model)
    optimizer = fleet.distributed_optimizer(optimizer)

    if model_args.sequence_parallel:
        sequence_parallel_utils.register_sequence_parallel_allreduce_hooks(model, 1, False)
        
    trainer = PipelineTrainer(model, traing_args, data_collator, train_dataset, optimizer)
    trainer.train()