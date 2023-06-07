# High-dimensional-Representer
This code corresponds to our ICML paper titled "[Representer Point Selection for Explaining Regularized High-dimensional Models](https://arxiv.org/abs/2305.20002)".


## Instructions
To replicate the experiments described in our paper, please adhere to the following steps:

1. Download the required datasets by executing the following command:
```
bash download.sh
```
2. Run the deletion curve diagnostic experiment on L1-regularized binary classifiers using the following command:
```
bash run_del_curve_l1_classification.sh
```
3. Run the deletion curve diagnostic experiment on collaborative filtering models using the following command:
```
bash run_del_curve_cf.sh
```



## Citation

If you find this code useful, please cite the following paper.

```
@article{tsai2023representer,
  title={Representer Point Selection for Explaining Regularized High-dimensional Models},
  author={Tsai, Che-Ping and Zhang, Jiong and Chien, Eli and Yu, Hsiang-Fu and Hsieh, Cho-Jui and Ravikumar, Pradeep},
  journal={arXiv preprint arXiv:2305.20002},
  year={2023}
}
```
