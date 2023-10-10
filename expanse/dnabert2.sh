lr=3e-5
data_path="/expanse/lustre/projects/nji102/sgriesmer/DNABERT"
seed=42

for	data in 0 1 2 3 4
  do 
        python train.py \
            --model_name_or_path zhihan1996/DNABERT-2-117M \
            --data_path  $data_path/GUE/tf/$data \
            --kmer -1 \
            --run_name DNABERT2_${vocab}_${lr}_tf_${data}_seed${seed} \
            --model_max_length 30 \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 64 \
            --gradient_accumulation_steps 1 \
            --learning_rate ${lr} \
            --num_train_epochs 3 \
            --fp16 \
            --save_steps 200 \
            --output_dir $data_path/output/dnabert2 \
            --evaluation_strategy steps \
            --eval_steps 200 \
            --warmup_steps 30 \
            --logging_steps 100000 \
            --overwrite_output_dir True \
            --log_level info \
            --find_unused_parameters False
done
