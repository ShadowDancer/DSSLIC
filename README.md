# DSSLIC: Deep Semantic Segmentation-based Layered Image Compression


## Datasets
-Cityscapes https://www.cityscapes-dataset.com/
-ADE20K https://groups.csail.mit.edu/vision/datasets/ADE20K/
-VOC2012

Quickstart:
- create env

```
conda env create -f environment.yml
conda activate DSSLIC
```


- download & unpack ADE20K
- copy config.template.yml to config.yml
- set dataset path in config file
- run scripts from src directory (PYCHARM: right click src and "Mark directory as / Sources root")
- run scripts with project root as cwd (PYCHARM: run configuration settings) 
- `python source/train.py`