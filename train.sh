torchrun --nproc_per_node=2 --nnodes=1 --rdzv_id=456 --rdzv_backend=c10d --rdzv_endpoint=localhost:48123 main.py  \
 --config=/home/cidar/Distributed_pytorch/configs/Unet/unet_ds.py \
 --mode='train'  \
 --workdir=/home/cidar/Distributed_pytorch/work_dir