#/bin/bash

for dataset in rcv1 gisette ; do
	for method in positive negative; do
		exp_dir=exp/l1_delcurve/$dataset/$method
		mkdir -p $exp_dir
		python3 -u del_curve_l1_classification.py --method $method --dataset $dataset \
				--seed 0 --C 1 --n_runs 40 --n_test 40 | tee $exp_dir/run.log 
	done
done

for method in positive negative; do
	exp_dir=exp/l1_delcurve/news20/$method
	mkdir -p $exp_dir
	python3 -u del_curve_l1_classification.py --method $method --dataset news20 \
				--seed 0 --C 10 --n_runs 40 --n_test 40 | tee $exp_dir/run.log 
done