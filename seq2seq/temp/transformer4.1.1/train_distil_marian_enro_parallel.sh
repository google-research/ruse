# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



export BS=4; CUDA_VISIBLE_DEVICES=0,1  USE_TF=0   python -m torch.distributed.launch --nproc_per_node=2 --master_port=9910  finetune_trainer.py --model_name_or_path t5-small --output_dir output_dir --adam_eps 1e-06 --data_dir wmt_en_ro --do_train --freeze_embeds --label_smoothing 0.1 --learning_rate 3e-5 --logging_first_step --logging_steps 1000 --max_source_length 128 --max_target_length 128 --num_train_epochs 1 --overwrite_output_dir --per_device_train_batch_size $BS --sortish_sampler  --task translation --val_max_target_length 128 --warmup_steps 500 --n_train 500 

