#!/bin/bash

export HYDRA_FULL_ERROR=1 

#module load mamba
#source activate /scratch/work/molinee2/conda_envs/env2024
source /home/leonardo.barreto/Documents/pesquisa/mestrado/audiodiff-restoration-project/python-envs/venv-mrcqtdiff/bin/activate

# main config
#conf=conf_FMA.yaml
conf=conf_MAESTRO.yaml

network="MR-cqtdiff_44k"
#network="ncsnpp_44k"
#network="cqtdiff+_44k"

python train.py --config-name=$conf \
  network=$network \

