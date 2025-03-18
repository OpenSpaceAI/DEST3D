conda create --name dest python=3.8 -y
conda activate dest

pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

pip install -r requirements.txt

cd pointnet2
python setup.py install --user
cd ..