# Project
This is code of project. Implement FlashAttention1 & 2 from scratch & automated selecting num_heads of attention with less exe time. A way to improve further is to change original softmax operation into approximation one.

Due to use jupiter, all codes are all in the same .ipynb, but before running it, still need to use terminal to install some packages.

Require for this packaging

Tensor flow Version should be 2.16.0   pip3 install --upgrade tensorflow
keras._version_ should be 3.3.3      pip3 install --upgrade keras

Install Python (>= 3.8).
Install Pytorch >= 1.12.1).

pip3 install packaging

e.g. ninja --version then echo $? should return exit code 0

If not (sometimes ninja --version then echo $? returns a nonzero exit code), uninstall then reinstall ninja (pip3 uninstall -y ninja && pip3 install ninja). 

# this is for import flashAttention original code
pip3 install flash-attn --no-build-isolation

MAX_JOBS=4 pip install flash-attn --no-build-isolation //RAM < 96G & many CPU

pip3 install "git+https://github.com/openai/triton.git#egg=triton&subdirectory=python"


Existed test is on A100 GPU
