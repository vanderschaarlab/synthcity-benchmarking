# dataset=brazil dataset
echo "python -u experiments/synthcity_bench_model_arg_augment.py --model=ctgan"
python -u experiments/synthcity_bench_model_arg_augment.py --model=ctgan
echo "python -u experiments/synthcity_bench_model_arg_augment.py --model=ddpm"
python -u experiments/synthcity_bench_model_arg_augment.py --model=ddpm
echo "python -u experiments/synthcity_bench_model_arg_augment.py --model=tvae"
python -u experiments/synthcity_bench_model_arg_augment.py --model=tvae
echo "python -u experiments/synthcity_bench_model_arg_augment.py --model=radialgan"
python -u experiments/synthcity_bench_model_arg_augment.py --model=radialgan
echo "python -u experiments/synthcity_bench_model_arg_augment.py --model=goggle"
python -u experiments/synthcity_bench_model_arg_augment.py --model=goggle
