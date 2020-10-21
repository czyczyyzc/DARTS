# DARTS
 Code for DARTS: DIFFERENTIABLE ARCHITECTURE SEARCH（Support for pytorch_1.6 and distributed training）

## To replicate the result of this paper:

### train for search
 python main_search.py --unrolled --gpu-ids=0

### train for searched architecture
 python main.py --auxiliary --gpu-ids=0

### distributed training for search (using 2 gpus)
 python -m torch.distributed.launch --nproc_per_node=2 main_search.py --unrolled --gpu-ids=0,1

### distributed training for searched architecture (using 2 gpus)
 python -m torch.distributed.launch --nproc_per_node=2 main.py --auxiliary --gpu-ids=0,1

### test
 python main.py --resume --evaluate --gpu_ids=0
