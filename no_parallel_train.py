import copy

import paddle
from paddle.io import Dataset, DataLoader, BatchSampler

from models.no_parallel_model import SimpleLlama

def get_simple_optimizer(config, parameter_list=None):
    optimizer = paddle.optimizer.Momentum(
        learning_rate=config.base_lr,
        momentum=config.momentum_rate,
        weight_decay=paddle.regularizer.L2Decay(config.l2_decay),
        parameters=parameter_list)
    return optimizer

class RandomDataset(Dataset):
    def __init__(self, config):
        self.num_samples = config.batch_size * config.batch_num
        self.config = config

    def __getitem__(self, idx):
        input_ids = paddle.randint(low=0, high=self.config.vocab_size, shape=[self.config.seq_length + 1])
        labels = copy.deepcopy(input_ids)[1:]
        input_ids = input_ids[:-1]
        return {
            "input_ids": input_ids,
            "labels": labels,
        }

    def __len__(self):
        return self.num_samples

class DefaultConfig:
    def __init__(self):
        self.vocab_size = 100
        self.hidden_size = 256
        self.num_attention_heads = 8
        self.num_key_value_heads = 8
        self.num_hidden_layers = 12
        self.intermediate_size = 512
        self.rms_norm_eps = 1e-6
        
        self.num_train_epochs = 1
        self.batch_size = 2
        self.batch_num = 5
        self.seq_length = 512
        
        self.base_lr = 0.1
        self.momentum_rate = 0.9
        self.l2_decay = 1e-4

# python no_parallel_train.py
if __name__ == "__main__":
    config = DefaultConfig()
    model = SimpleLlama(config)
    optimizer = get_simple_optimizer(config, parameter_list=model.parameters())

    dataset = RandomDataset(config)
    sampler = BatchSampler(dataset, batch_size=config.batch_size, shuffle=False, drop_last=True)
    train_loader = DataLoader(dataset, batch_sampler=sampler, num_workers=1)

    for eop in range(config.num_train_epochs):
        model.train()

        for batch_id, data in enumerate(train_loader()):
            input_ids, labels = data["input_ids"], data["labels"]
            loss = model(input_ids, labels)
            loss.backward()
            
            optimizer.step()
            model.clear_gradients()
            
            print(f"epoch: {eop}, batch_id: {batch_id}, loss: {loss.numpy()}")