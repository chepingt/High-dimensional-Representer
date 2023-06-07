#!/bin/bash

n_runs=40
n_test=40
dataset=ml-1m

for model in MF YoutubeDNN ; do
	for method in negative positive;  do
		exp_dir=exp/delcurve/$dataset/$model/$method
		mkdir -p $exp_dir
		python3 del_curve_cf.py --gpu 0  --config config/$model-$dataset.yaml --exp_dir $exp_dir \
						--dataset $dataset  --seed 0 --method $method \
						--n_runs $n_runs --n_test $n_test | tee $exp_dir/run.log 

	done
done

exit 1