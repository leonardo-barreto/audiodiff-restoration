#!/bin/bash

source /home/leonardo.barreto/Documents/pesquisa/mestrado/audiodiff-restoration-project/python-envs/venv-mrcqtdiff/bin/activate

conf=conf_inference78rpm.yaml

name="MR-CQTDiff_44k_MAESTRO"
PATH_EXPERIMENT=experiments/$name

#choose model and checkpoint
network=MR-cqtdiff_44k
ckpt="experiments/MR-CQTDiff_44k_MAESTRO/len1_5-bat5-lr2-sched2_3-diffBern-3090-1600000.pt"

#choose conditioner (dps or mm)
conditioner=mm

#choose number of diffusion steps (T) for inference
T=140

#choose conditioner parameters
xi=0.35
maxiter=1
measurement_noise_sigma=0.0

#choose noisy audio file and output folder
base_path=/home/leonardo.barreto/Documents/pesquisa/mestrado/audiodiff-restoration-project/audio-examples
noisy_audio=$base_path/panel-1/noisy/midsum_short.wav
restored_dir=$base_path/panel-1/restored/conditioner_tests
mkdir -p "$restored_dir"

#build parameter tags
network_tag=MRCQTDiff
conditioner_tag=$(echo "$conditioner" | tr '[:lower:]' '[:upper:]')
xi_tag=$(echo "$xi" | tr -d '.')
noisy_name=$(basename "$noisy_audio")
noisy_stem=${noisy_name%.*}
output_name="${noisy_stem}_${network_tag}_${conditioner_tag}_T${T}_xi${xi_tag}"

if [ "${conditioner,,}" = "mm" ]; then
  output_name="${output_name}_maxiter${maxiter}"
fi

output_file="${restored_dir}/${output_name}.wav"
run_times_file="${restored_dir}/run_times.txt"

mkdir -p "$PATH_EXPERIMENT"

#fixed-gain estimate is used in these inference tests
start_seconds=$(date +%s)

conditioner_overrides=(
  "conditioner.xi=$xi"
  "conditioner.infer_xi=False"
)

if [ "${conditioner,,}" = "mm" ]; then
  conditioner_overrides+=(
    "conditioner.maxiter=$maxiter"
    "conditioner.measurement_noise_sigma=$measurement_noise_sigma"
  )
fi

python inference_78rpm.py --config-name=$conf \
  network=$network \
  conditioner=$conditioner \
  checkpoint=$ckpt \
  model_dir=$PATH_EXPERIMENT \
  noisy_audio=$noisy_audio \
  output_file=$output_file \
  tester.sampling_params.T=$T \
  inference.use_gain=True \
  inference.use_range=False \
  "${conditioner_overrides[@]}"

run_status=$?
end_seconds=$(date +%s)
elapsed_seconds=$((end_seconds - start_seconds))
elapsed_minutes=$((elapsed_seconds / 60))
elapsed_remainder=$((elapsed_seconds % 60))

printf '%s: %dm %02ds\n' "$output_name.wav" "$elapsed_minutes" "$elapsed_remainder" >> "$run_times_file"

exit "$run_status"
