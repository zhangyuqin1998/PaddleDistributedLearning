环境配置:
# 1.拉取paddle推荐的镜像
docker pull registry.baidubce.com/paddlepaddle/paddle:2.6.1-gpu-cuda12.0-cudnn8.9-trt8.6
# 2.进入镜像
docker run xxxx
# 3.安装 paddlenlp
pip install --upgrade paddlenlp
# 4.数据集准备
wget https://bj.bcebos.com/paddlenlp/models/transformers/llama/data/llama_openwebtext_100k_ids.npy
wget https://bj.bcebos.com/paddlenlp/models/transformers/llama/data/llama_openwebtext_100k_idx.npz
