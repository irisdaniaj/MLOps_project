Object detection with YOLOv8
==============================

This repository contains the group(Iris Jimenez, Nicolò Campagnoli, Meimingwei Li) project work for the MLOps course at LMU Munich. 


**1 Overall goal**: The goal is to perform object segmentation on images. We will not train the model from scratch but rather fine-tune it on our dataset. \
**2 Framework:** As a framework [PyTorch Image Models](https://github.com/huggingface/pytorch-image-models) was used. It contains models, scripts and pre trained for a lot of state-of-the-art image models within computer vision.\
**3 Data:** We train our model on the [VisDrone 2019](https://github.com/VisDrone/VisDrone-Dataset) dataset. The VisDrone2019 dataset is collected by the AISKYEYE team at Lab of Machine Learning and Data Mining , Tianjin University, China. The benchmark dataset consists of 288 video clips formed by 261,908 frames and 10,209 static images, captured by various drone-mounted cameras, covering a wide range of aspects including location (taken from 14 different cities separated by thousands of kilometers in China), environment (urban and country), objects (pedestrian, vehicles, bicycles, etc.), and density (sparse and crowded scenes).  \
**4 Deep Learning model used:**  We used the [Ultralytics YOLOv8 model](https://github.com/ultralytics/ultralytics).

# How to install 
Clone the repository 
```
git clone https://github.com/irisdaniaj/MLOps_project.git
```
We suggest to install the requirements.txt preferebly in a new conda environment 
```
pip install -r requirements.txt
``` 
# How to run 
Download the dataset by running from the command line
```
python src/data/make_dataset.py 
```
then load the pretrained model 
```
python src/models/train_model.py
```
Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
