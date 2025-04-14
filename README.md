# relief


### Set Up

<details>
<summary>Instructions for Brown's OSCAR</summary>

Before doing the below steps, make sure to:
```
module load miniconda3/23.11.0s
conda init bash
```
More instructions available on [CCV's website](https://docs.ccv.brown.edu/oscar/software/miniconda)
</details>
</br>

Use the following steps to set up your python environment:
```
conda create -p ./relief.env --file requirements.txt
conda activate relief.env
conda install --channel conda-forge pygmt
pip install --file other_reqs.txt
```
After the setup, the environment will be active.
Deactivate the environment with `conda deactivate`.
Activate it again at anytime while in the repo with `conda activate relief.env`
