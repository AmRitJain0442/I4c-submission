#!/bin/bash
# Runs on a fresh GCP Deep Learning VM: stage data, install deps, start training.
# Usage: bash remote_setup.sh "<train.py args>"
set -e
cd ~

if [ ! -d data/train ]; then
  gsutil cp gs://kla-restore-tribe-v2/train.zip gs://kla-restore-tribe-v2/Test_NoisyLR.zip .
  mkdir -p data
  unzip -q -o train.zip -d data -x "__MACOSX/*" "*.DS_Store"
  unzip -q -o Test_NoisyLR.zip -d data -x "__MACOSX/*" "*.DS_Store"
fi

gsutil cp gs://kla-restore-tribe-v2/code.tar.gz .
tar xzf code.tar.gz

pip install -q lpips timm scikit-image

nohup python train.py --data_root data $1 > train_run.log 2>&1 &
echo "training started: $1"
