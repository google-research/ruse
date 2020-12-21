export BS=4; CUDA_VISIBLE_DEVICES=0,1,2,3  USE_TF=0   python -m torch.distributed.launch --nproc_per_node=4 --master_port=9918  finetune_t5_trainer.py configs/experiments/test.json
