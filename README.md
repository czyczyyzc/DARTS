# DARTS
 Code for DARTS: DIFFERENTIABLE ARCHITECTURE SEARCH（Support for pytorch_1.6 and distributed training）

## To replicate the result of this paper:

### train
 python main.py --unrolled --gpu_ids=0

### distributed training (using 2 gpus)
 python -m torch.distributed.launch --nproc_per_node=2 main.py --unrolled --gpu_ids=0,1

### test
 python main.py --resume --evaluate --gpu_ids=0
