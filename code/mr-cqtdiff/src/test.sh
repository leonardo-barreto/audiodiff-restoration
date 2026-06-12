#!/bin/bash

source /home/leonardo.barreto/Documents/pesquisa/mestrado/audiodiff-restoration-project/python-envs/venv-mrcqtdiff/bin/activate

#test_script=test_unconditional.py
test_script=test_conditional.py

conf=conf_SamplerTesting.yaml

name="MR-CQTDiff_44k_MAESTRO"
PATH_EXPERIMENT=experiments/$name

#choose model and dataset
network=MR-cqtdiff_44k
ckpt="experiments/MR-CQTDiff_44k_MAESTRO/len1_5-bat5-lr2-sched2_3-diffBern-3090-1600000.pt"


mkdir -p $PATH_EXPERIMENT

python $test_script --config-name=$conf \
  model_dir=$PATH_EXPERIMENT \
  tester.checkpoint=$ckpt \
  #tester.unconditional.num_samples=1 \
  #tester.overriden_name="2D_OpenSinger_HD_CQT_waveform_IS2_100k_v2" \



