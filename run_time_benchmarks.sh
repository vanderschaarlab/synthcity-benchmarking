# dataset=sine
echo "python -u experiments/synthcity_bench_model_arg_time.py --dataset=sine --model=timegan"
python -u experiments/synthcity_bench_model_arg_time.py --dataset=sine --model=timegan
echo "python -u experiments/synthcity_bench_model_arg_time.py --dataset=sine --model=fflows"
python -u experiments/synthcity_bench_model_arg_time.py --dataset=sine --model=fflows
echo "python -u experiments/synthcity_bench_model_arg_time.py --dataset=sine --model=timevae"
python -u experiments/synthcity_bench_model_arg_time.py --dataset=sine --model=timevae

# dataset=googlestocks
echo "python -u experiments/synthcity_bench_model_arg_time.py --dataset=googlestocks --model=timegan"
python -u experiments/synthcity_bench_model_arg_time.py --dataset=googlestocks --model=timegan
echo "python -u experiments/synthcity_bench_model_arg_time.py --dataset=googlestocks --model=fflows"
python -u experiments/synthcity_bench_model_arg_time.py --dataset=googlestocks --model=fflows
echo "python -u experiments/synthcity_bench_model_arg_time.py --dataset=googlestocks --model=timevae"
python -u experiments/synthcity_bench_model_arg_time.py --dataset=googlestocks --model=timevae
