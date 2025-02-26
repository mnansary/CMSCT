#pytorch
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
# bytetrack reqs
pip install -r requirements.txt
pip install -e dependencies/ByteTrack/
# tensor-rt
pip install nvidia-pyindex
pip install nvidia-tensorrt==99.0.0
pip install git+https://github.com/NVIDIA-AI-IOT/torch2trt.git

# install repo
pip install -e .