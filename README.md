# One-shot Active Learning from Relational Data


## Requirements

- This project requires python 3.6
- Python package requiremnts is listed in _requirements.txt_

## Preparing environment

- We highly recomment using virtual environment for this part of the project. To create a virtual environment with Anaconda and activating it, run the following:

```
    conda create -n osal python=3.6
    conda activate osal
```

- Then install all the dependencies though pip:

```
    pip install -r requirements.txt
```

## Usage

To reproduce the results in MLG paper, run the following: 

```
    sh mlg20.sh <num_trials> <budget>
```

- Use the following values for the parameters: 
    - num_trials = 5
    - budget = 224

In order to run a single experiment, execute following:

```
    python experiment.py -config configs/<clf>_<ds>.json -nt <nt> -b <b> -algos <algos>
```

- Parameter description:
    - \<clf>    : classfier name (e.g. wvrn, cc, sgc, gsage)
    - \<ds>     : dataset name (e.g. citeseer, cora, hateful, pubmed)
    - \<nt>     : number of trials (e.g. 1, 5)
    - \<b>      : active learning budget (e.g. 32, 224)
    - \<algos>  : space separated names of sampling algo (e.g. rs, fp, ffs .. )

- A simple run:

```
    python experiment.py -config configs/wvrn_cora.json -nt 1 -b 32 -algos rs fp
```

---
## Citation
```
@inproceedings{ahsan-mlg20,
  title={Effectiveness of Sampling Strategies for One-shot Active Learning from Relational Data},
  author={Ahsan, Ragib and Zheleva, Elena},
  booktitle={Proceedings of the 16th International Workshop on Mining and Learning with Graphs (MLG)},
  year={2020}
}
```