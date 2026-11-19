source /home/leonardo.barreto/Documents/pesquisa/mestrado/audiodiff-restoration-project/python-envs/venv-mrcqtdiff/bin/activate

#choose noisy audio file
base_path=/home/leonardo.barreto/Documents/pesquisa/mestrado/audiodiff-restoration-project/audio-examples
#noisy_audio=$base_path/sampling_testing/noisy_sample.wav
#output_file=$base_path/RG_test_output.wav
noisy_audio=$base_path/panel-1/noisy/midsum_short.wav
output_file=$base_path/panel-1/restored/midsum_short_CQTDiff_T50.wav

python inference.py \
    noisy_audio=$noisy_audio \
    output_file=$output_file \
    device="cuda:1" \
    inference.T=50 \
    inference.xi=0.35 \
    inference.use_gain=True \
    inference.use_range=False

output_file=$base_path/panel-1/restored/midsum_short_CQTDiff_T140.wav

python inference.py \
    noisy_audio=$noisy_audio \
    output_file=$output_file \
    device="cuda:1" \
    inference.T=140 \
    inference.xi=0.35 \
    inference.use_gain=True \
    inference.use_range=False