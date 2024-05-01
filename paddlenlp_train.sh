master=127.0.0.1
port=36677

distributed_log_dir=paddlenlp_log
rm -rf $distributed_log_dir

python -u  -m paddle.distributed.launch \
    --master $master:$port \
    --gpus "0" \
    --log_dir "./$distributed_log_dir" \
    paddlenlp_train.py \
    --output_dir "output" \
