start=`date +%s`
python finetune_t5_trainer.py configs/experiments/test.json
end=`date +%s`

runtime=$((end-start))
echo $runtime
