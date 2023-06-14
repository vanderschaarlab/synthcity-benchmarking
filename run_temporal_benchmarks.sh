# dataset=googlestocks
echo "python -u synthcity_bench_model_arg_time.py --dataset=googlestocks --model=timegan"
python -u synthcity_bench_model_arg_time.py --dataset=googlestocks --model=timegan
echo "python -u synthcity_bench_model_arg_time.py --dataset=googlestocks --model=fflows"
python -u synthcity_bench_model_arg_time.py --dataset=googlestocks --model=fflows
echo "python -u synthcity_bench_model_arg_time.py --dataset=googlestocks --model=timevae"
python -u synthcity_bench_model_arg_time.py --dataset=googlestocks --model=timevae

# dataset=sine
echo "python -u synthcity_bench_model_arg_time.py --dataset=sine --model=timegan"
python -u synthcity_bench_model_arg_time.py --dataset=sine --model=timegan
echo "python -u synthcity_bench_model_arg_time.py --dataset=sine --model=fflows"
python -u synthcity_bench_model_arg_time.py --dataset=sine --model=fflows
echo "python -u synthcity_bench_model_arg_time.py --dataset=sine --model=timevae"
python -u synthcity_bench_model_arg_time.py --dataset=sine --model=timevae

# dataset=pbc
echo "python -u synthcity_bench_model_arg_time.py --dataset=pbc --model=timegan"
python -u synthcity_bench_model_arg_time.py --dataset=pbc --model=timegan
echo "python -u synthcity_bench_model_arg_time.py --dataset=pbc --model=fflows"
python -u synthcity_bench_model_arg_time.py --dataset=pbc --model=fflows
echo "python -u synthcity_bench_model_arg_time.py --dataset=pbc --model=timevae"
python -u synthcity_bench_model_arg_time.py --dataset=pbc --model=timevae
