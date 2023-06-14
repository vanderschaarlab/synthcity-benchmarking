# Synthcity: a benchmark framework for diverse use cases of tabular synthetic data

## Abstract

![image](https://github.com/vanderschaarlab/synthcity/raw/main/docs/arch.png "Synthcity Architecture")

Synthcity is an open-source software package for innovative use cases of synthetic data in ML fairness, privacy and augmentation across diverse tabular data modalities, including static data, regular and irregular time series, data with censoring, multi-source data, composite data, and more. Synthcity provides the practitioners with a single access point to cutting edge research and tools in synthetic data. It also offers the community a playground for rapid experimentation and prototyping, a one-stop-shop for SOTA benchmarks, and an opportunity for extending research impact. The library can be accessed on GitHub ([this https URL](https://github.com/vanderschaarlab/synthcity)) and pip ([this https URL](https://pypi.org/project/synthcity/)). We warmly invite the community to join the development effort by providing feedback, reporting bugs, and contributing code.

## :rocket: Installation

First set up your virtual environment. It is recommended to use a python 3.9 environment, which can be created as follows using conda:

```
conda create -n [the-name-of-your-env] python=3.9
```

The activate your environment with:

```
conda activate [the-name-of-your-env]
```

Clone the repository:
```
git clone https://github.com/vanderschaarlab/synthcity-benchmarking.git
```

Please then install all the requirements for the repository. This can be done with the following:
```
pip install -r requirements.txt
```

## :robot: Data
First download the relevant data files and place them in the data folder. This is required for the standard and augmentation benchmark experiments. The temporal experiments load datasets from synthcity dynamically.

The data for the standard benchmarks can be obtained by running the following command from the root directory:
```
bash get_data.sh
```

The data used for the augmentation benchmark experiment is the SIVEP-Gripe public dataset, which contains anonymized records of COVID-19 patients in Brazil. See the following paper for more details:

```
Pedro Baqui, Valerio Marra, Ahmed M Alaa, Ioana Bica, Ari Ercole, and Mihaela van der Schaar.
Comparing covid-19 risk factors in brazil using machine learning: the importance of socioeconomic,
demographic and structural factors. Scientific reports, 11(1):1–10, 2021
```

The pre-processing used for this dataset is as described in `/SIVEP-Gripe public dataset.md` in the root directory.

To run an experiment 'as-is' the data must be saved in the folder '/data'. This is automatic when running the bash script above for standard experiments. For the augmentation experiement the data must be obtained and saved to `/data/augmentation`



## :boom: Usage
To run the experiments first make sure the data is in the correct location as described above.

The benchmark scripts can be used in two ways 1) via notebook, 2) via terminal.

### :book: Notebooks
#### Standard static experiment

Open the notebook and run the cells. Edit the final cell to adjust which dataset/models you want to run.

#### :gear: Terminal

The full set of experiments (all models and all datasets) can be run using the below three commands. This may take some considerable time. If this is not desirable, it is recommended to use the notebooks.

Standard static:
```
bash run_static_benchmarks.sh
```

Static Augmentation:
```
bash run_augmentation_benchmarks.sh
```

Temporal:
```
bash run_temporal_benchmarks.sh
```

## Citation
If our paper or code helped you in your own research, please cite our work as:

```
@inproceedings{qian2023synthcity,
  title={Synthcity: a benchmark framework for diverse use cases of tabular synthetic data},
  author={Qian, Zhaozhi and Davis, Rob and van der Schaar Mihaela},
  booktitle={Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks},
  year={2023}
}
```
