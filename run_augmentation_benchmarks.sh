# dataset=brazil dataset
echo "python -u synthcity_bench_model_arg_augment.py --model=ctgan"
python -u synthcity_bench_model_arg_augment.py --model=ctgan
echo "python -u synthcity_bench_model_arg_augment.py --model=fflows"
python -u synthcity_bench_model_arg_augment.py --model=fflows
echo "python -u synthcity_bench_model_arg_augment.py --model=timevae"
python -u synthcity_bench_model_arg_augment.py --model=timevae
