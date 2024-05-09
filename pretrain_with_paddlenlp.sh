master=127.0.0.1
port=36677

distributed_log_dir=paddlenlp_log
rm -rf $distributed_log_dir

python -u  -m paddle.distributed.launch \
    --master $master:$port \
    --gpus "0,1,2,3" \
    --log_dir "./$distributed_log_dir" \
    pretrain_with_paddlenlp.py \
    --output_dir "output" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --max_steps 2000 \
    --logging_steps 50 \
    --disable_tqdm true \
    --sequence_parallel false \
    --tensor_parallel_degree 4 \
    --sep_parallel_degree 1 \
    --sharding_parallel_degree 1 \
    --pipeline_parallel_degree 1 \

