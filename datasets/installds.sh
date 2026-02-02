# Sen1Floods11 (446 samples)
source .venv/bin/activate

echo "Downloading Sen1Floods11 Training Data"
python ./datasets/download_sen1.py
tar -xvf ./datasets/sen1floods11_v1.1.tar.gz

# Timor ML4FLood
echo "Downloading Timor Test Set..."
git xet install
git clone https://huggingface.co/datasets/azzenn4/Timor_ML4FLood
