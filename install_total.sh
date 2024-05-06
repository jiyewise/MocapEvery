pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
# pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu117_pyt1131/download.html
pip install pytorch3d
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.13.1+cu117.html

pip install -r requirements.txt
# install fairmotion (modified)
cd fairmotion
pip install -e .
cd ..