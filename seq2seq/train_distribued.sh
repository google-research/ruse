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



export BS=4; CUDA_VISIBLE_DEVICES=0,1  USE_TF=0   python -m torch.distributed.launch --nproc_per_node=2 --master_port=9910  finetune_t5_trainer.py \
	--model_name_or_path t5-small \
	--tokenizer_name t5-small \
	--learning_rate  1e-2 \
	--output_dir outputs/test \
	--max_source_length 128 \
	--max_target_length 128 \
	--val_max_target_length 128 \
		--test_max_target_length 128 \
		--num_train_epochs 10 \
		--warmup_steps 500 \
		--eval_steps 200 \
		--overwrite_output_dir  \
		--tasks  scitail boolq \
		--eval_tasks rte boolq \
		--sampling  \
		--label_smoothing 0.1 \
		--per_device_train_batch_size 64 \
		--per_device_eval_batch_size 64 \
		--save_steps 20 \
		--logging_first_step \
		--logging_steps 200 \
		--save_total_limit 1 \
		--train_adapters \
		--adapter_config_name parametric-meta-adapter \
		--temperature 10 \
		--do_eval  \
		--predict_with_generate \
		--n_train  10 \
		--task_embedding_dir test_data/task_embeddings/n-train-all \
		--task_embedding_dim  512 \
		--n_val 10 \
		--n_train 10 \
		--do_finetune \
		--do_train  \
		--n_finetune 100 \
		--eval_output_dir outputs/eval_test \
		--reduction_factor 16 \
		--non_linearity relu \
		--train_task_embeddings \
		--projected_task_embedding_dim  512 \
		--unfreeze_lm_head \
		--unfreeze_layer_norms \
                "$@"
