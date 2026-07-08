#!/bin/bash

source /home/leonardo.barreto/Documents/pesquisa/mestrado/audiodiff-restoration-project/python-envs/venv-mrcqtdiff/bin/activate

conf=conf_inference78rpm.yaml

name="MR-CQTDiff_44k_MAESTRO"
PATH_EXPERIMENT=experiments/$name

#choose model and checkpoint
network=MR-cqtdiff_44k
ckpt="experiments/MR-CQTDiff_44k_MAESTRO/len1_5-bat5-lr2-sched2_3-diffBern-3090-1600000.pt"

#choose noisy audio file
noisy_audio="/home/leonardo.barreto/Documents/pesquisa/mestrado/audiodiff-restoration-project/audio-examples/sampling_testing/noisy_sample.wav"
output_file="/home/leonardo.barreto/Documents/pesquisa/mestrado/audiodiff-restoration-project/audio-examples/sampling_test_output.wav"


mkdir -p $PATH_EXPERIMENT

python inference_78rpm.py --config-name=$conf \
  network=$network \
  checkpoint=$ckpt \
  model_dir=$PATH_EXPERIMENT \
  noisy_audio=$noisy_audio \
  output_file=$output_file \



