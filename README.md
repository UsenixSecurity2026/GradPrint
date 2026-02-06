# GradPrint

This repository contains the Adversarial Android malware detection system GradPrint.

Due to GitHub’s file-size limits, the full project (including the dataset and model weight files) is available on Zenodo: [https://zenodo.org/uploads/18507012].



## Quick Start

GradPrint is a detection method that implements defense at the feature level. 
Therefore, the basis of the Quick Start is a dataset of attack data based on attack methods(we provided here).



### Installation

1. Install any packages by `environment.yml`. 
  
```bash
   conda env create -f environment.yml
```

2. conda activate 
```bash
  conda activate GradPrint
```

### Configuration

The next step is to fill out configuration in `settings.py`.

Due to the multiple moving parts, we try to resolve most paths to their absolute form to reduce slip-ups.

* `_project_path`: The absolute path to the root folder of the project (e.g., `/tmp/GradPrint/`). **Please adjust the project paths according to your project structure.**

The helper functions `_project()` can be used to resolve files in these directories given their relative paths.

Experimental settings:

* `data_source`: The path where the data loaded from different attacker methods[`AdvDroidZero & BagAmmo .../train_sample & test sample & adv sample`] (e.g., `_project('dataset/')`)
* `base_clf_dir`: The path where the classifier model provided by Attacker AdvDroidZero (only AdvDroidZero need, train target model need, other attacker has provided in dataset)(
  e.g., `_project('base_clf_models/')`)
* `model_save_dir`: The path of target model saved (
  e.g., `_project('checkpoints/')`)
* `GradPrint_result_dir`: The path of running results (
  e.g., `_project('results/GradPrint/')`)


### Dataset

To facilitate quick reproduction, we have provided a dataset of adversarial samples generated under the AdvDroidZero attack method in this GitHub project, located in `dataset/AdvDroidZero`. 

For other attack datasets & attack checkpoints, you can obtain them through Zenodo: [Zenodo](https://zenodo.org/uploads/14713949). 

1. The **HIV_JSMA** and **HIV_CW** datasets, as well as the **BagAmmo** dataset, utilize the same **train_sample** and **test_sample**.
2. Only the checkpoints and dataset directories in Google Drive need to be added in this github project.


> **Note:**  
> If you want to implement the attack dataset from the source APK, you can refer to the following resources:  
> - Source APK: [AMDs](https://github.com/Maruko912/AMDs)  
> - AdvDroidZero Attacker: [AdvDroidZero-Access-Instructions](https://github.com/gnipping/AdvDroidZero-Access-Instructions)





### Usage

As well as the configuration settings, there are a number of command line arguments:

```
$ python detection_GradPrint.py -h
usage: detection_GradPrint.py [-h] [-R RUN_TAG] [--mode MOAD_LOAD]
               [--detection DETECTION] [--classifier CLASSIFIER]
               [--attacker ATTACKER] [--Random_sample] [--performance_fixed_param]
```

#### Detection Performance 

This stage is to valid adversarial sample detection performance. 

Example:
```shell
  python detection_GradPrint.py -R detection_GradPrint \
    --mode load \
    --detection mamadroid \
    --classifier rf \
    --attacker AdvDroidZero \
    --Random_sample \
    --performance_fixed_param
```

More shell command lines can be seen in file **command_lines**. 

## Note

The open-source version of GradPrint was rapidly prepared, so there might be some bugs.
Please feel free to reach out with any questions or report any issues you encounter.
We are here to assist and improve the repository.


## Licensing

For ethical considerations, _the remainder of the repository_ is covered by a similar license but which also restricts
the use of the code to academic purposes and which specifically prohibits commercial applications.

> Any redistribution or use of this software must be limited to the purposes of non-commercial scientific research or
> non-commercial education. Any other use, in particular any use for commercial purposes, is prohibited. This includes,
> without limitation, incorporation in a commercial product, use in a commercial service, or production of other
> artefacts for commercial purposes.
