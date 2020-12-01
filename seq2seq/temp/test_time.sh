start=`date +%s`
python finetune_t5_trainer.py configs/mrpc_adapter_local.json
end=`date +%s`

runtime=$((end-start))
echo $runtime
