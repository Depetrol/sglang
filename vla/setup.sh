echo "Setting up SGLang environment..."
source $HOME/miniconda3/etc/profile.d/conda.sh
$HOME/miniconda3/bin/conda create -n sglang-stable python=3.10 -y
conda activate sglang-stable
cd ~
git clone https://github.com/depetrol/sglang-stable
cd sglang-stable
pip install --upgrade pip
pip install -e "python[all]"
pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/
pip install timm===0.9.10 openpyxl
pip install json_numpy numpy transformers