### Steps to follow for distributed training  in Pytorch

1. import within train method

```from torch.distributed import init_process_group, destroy_process_group```

2. initialize the group

`init_process_group(backend='nccl')`

3. Update config

```
config.local_rank = int(os.environ['LOCAL_RANK'])
config.global_rank = int(os.environ['RANK'])
if config.local_rank == 0:
    logging.info("Starting training process %d" % config.global_rank)
    logging.info("Config: %s" % config)
```
4. set device

```
torch.cuda.set_device(config.local_rank)
config.device = torch.device('cuda')
```
5. Update dataloader, disable shuffling in train_dl and use distributed sampler

```
from torch.utils.data.distributed import DistributedSampler
sampler= DistributedSampler(train_dataset, shuffle=True)
train_dataloader = DataLoader(train_dataset, batch_size=configs.training.batch_size, shuffle=False,drop_last=True, sampler=sampler)
```

6. Wrap the model with the Distributed Data Parallel
```
from torch.nn.parallel import DistributedDataParallel
model = YourModel()
model= DistributedDataParallel(model, device_ids=[config.local_rank])
```
7. Run the model
```
torchrun --nproc_per_node=2 --nnodes=1 --rdzv_id=456 --rdzv_backend=c10d --rdzv_endpoint=localhost:48123 main.py  \
# other args
#nproc_per_node=  number of gpu in each node
#nnodes = number of cluster

```