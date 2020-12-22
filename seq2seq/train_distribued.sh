#export BS=4; CUDA_VISIBLE_DEVICES=0,1,2,3  USE_TF=0   
python -m torch.distributed.launch --nproc_per_node=4 --master_port=9918  finetune_t5_trainer.py temp_configs/xla-lr-3e-02-r-8-l-true.json
python -m torch.distributed.launch --nproc_per_node=4 --master_port=9918  finetune_t5_trainer.py temp_configs/xla-lr-3e-02-r-8-l-false.json
python -m torch.distributed.launch --nproc_per_node=4 --master_port=9918  finetune_t5_trainer.py temp_configs/xla-lr-3e-03-r-8-l-true.json
python -m torch.distributed.launch --nproc_per_node=4 --master_port=9918  finetune_t5_trainer.py temp_configs/xla-lr-3e-03-r-8-l-false.json

python -m torch.distributed.launch --nproc_per_node=4 --master_port=9918  finetune_t5_trainer.py temp_configs/xla-lr-3e-04-r-8-l-false.json
python -m torch.distributed.launch --nproc_per_node=4 --master_port=9918  finetune_t5_trainer.py temp_configs/xla-lr-2e-05-r-8-l-true.json


python -m torch.distributed.launch --nproc_per_node=4 --master_port=9918  finetune_t5_trainer.py temp_configs/xla-lr-3e-04-r-8-l-true.json
python -m torch.distributed.launch --nproc_per_node=4 --master_port=9918  finetune_t5_trainer.py temp_configs/xla-lr-2e-05-r-8-l-false.json


python -m torch.distributed.launch --nproc_per_node=4 --master_port=9918  finetune_t5_trainer.py temp_configs/xla-lr-3e-05-r-8-l-true.json
python -m torch.distributed.launch --nproc_per_node=4 --master_port=9918  finetune_t5_trainer.py temp_configs/xla-lr-3e-05-r-8-l-false.json

