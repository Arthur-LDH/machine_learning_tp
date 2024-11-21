# ynov

Ce projet a été généré par le template Data de l'ADS  https://github.com/France-Travail/gabarit et largement modifié et simplifié.

## 0. Installation

- (optionnel) Nous vous conseillons d'utiliser un environnement virtuel python avec poetry

	- `pip install poetry`

	- `poetry init`

	- `poetry add pandas` to add packages to your project 

	- `poetry install --all-extras` to install all the packages needed in your project


## 1. notebook use 

Add your notebooks in the folder ynov-notebooks

Install Jupyter 

- `poetry add jupyter`

Launch Jupyter

- `jupyter-notebooks`

Organize your notebook in differents folders and keep them clean. 
Try to build function that can be re-usable. You will write them in the folder ynov and load them in your notebooks. 
It will be easier to maintain and will make your notebook cleaner.

## 2. run scripts

Example with the split train test valid script 

- `poetry run 0_split_train_valid_test.py -f dataset.csv --perc_train 60 --perc_valid 20 --perc_test 20`

