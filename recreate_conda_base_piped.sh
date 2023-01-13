base_env_name='conjtest'
piped_env_name=$base_env_name'_piped'

echo $piped_env_name

conda env remove --name $piped_env_name

conda create --name $piped_env_name --clone $base_env_name
eval "$(conda shell.bash hook)"
conda activate $piped_env_name

pip install hausdorff
