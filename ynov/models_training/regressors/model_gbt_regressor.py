#!/usr/bin/env python3

## Modèle Gradient Boosting Tree
# Auteurs : Agence dataservices
# Date : 28/10/2019
#
# Classes :
# - ModelGBTRegressor -> Modèle pour prédictions via Gradient Boosting Tree - Regression


import os
import json
import logging
import numpy as np
import pandas as pd
import dill as pickle
from datetime import datetime
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from ynov import utils
from ynov.models_training import utils_models
from ynov.models_training.model_pipeline import ModelPipeline
from ynov.models_training.model_regressor import ModelRegressorMixin


class ModelGBTRegressor(ModelRegressorMixin, ModelPipeline):
    '''Modèle pour prédictions via Gradient Boosting Tree - Regression'''

    _default_name = 'model_gbt_regressor'

    def __init__(self, gbt_params: dict = {}, **kwargs):
        '''Initialisation de la classe (voir ModelPipeline, ModelClass & ModelRegressorMixin pour arguments supplémentaires)

        Kwargs:
            gbt_params (dict) : paramètres pour la Gradient Boosting Tree
        '''
        # Init.
        super().__init__(**kwargs)

        # Get logger (must be done after super init)
        self.logger = logging.getLogger(__name__)

        # Gestion modèles
        self.gbt = GradientBoostingRegressor(**gbt_params)
        # On def. une pipeline pour compatibilité autres modèles
        self.pipeline = Pipeline([('gbt', self.gbt)])

    def save(self, json_data: dict = None):
        '''Sauvegarde du modèle

        Kwargs:
            json_data (dict): configuration à ajouter pour la sauvegarde JSON
        Raises:
            TypeError: si l'objet json_data n'est pas du type dict
        '''
        # Save model
        if json_data is None:
            json_data = {}

        # Pas besoin de sauvegarder les params des steps de la pipeline, déjà fait dans model_pipeline

        # Save
        super().save(json_data=json_data)

    def reload_from_standalone(self, configuration_path: str, model_pipeline_path: str, preprocess_pipeline_path: str, **kwargs):
        '''Fonction pour recharger un modèle à partir de sa configuration et de sa pipeline
        - /!\\ Exploratoire /!\\ -

        Args:
            configuration_path (str): path to configuration
            model_pipeline_path (str): path to standalone pipeline
            preprocess_pipeline_path (str): path to preprocess pipeline
        Raises:
            ValueError : si configuration_path est à None
            ValueError : si model_pipeline_path est à None
            FileNotFoundError : si l'objet configuration_path n'est pas un fichier existant
            FileNotFoundError : si l'objet model_pipeline_path n'est pas un fichier existant
        '''
        if configuration_path is None:
            raise ValueError("L'argument configuration_path ne peut pas être à None")
        if model_pipeline_path is None:
            raise ValueError("L'argument model_pipeline_path ne peut pas être à None")
        if preprocess_pipeline_path is None:
            raise ValueError("L'argument preprocess_pipeline_path ne peut pas être à None")
        if not os.path.exists(configuration_path):
            raise FileNotFoundError(f"Le fichier {configuration_path} n'existe pas")
        if not os.path.exists(model_pipeline_path):
            raise FileNotFoundError(f"Le fichier {model_pipeline_path} n'existe pas")
        if not os.path.exists(preprocess_pipeline_path):
            raise FileNotFoundError(f"Le fichier {preprocess_pipeline_path} n'existe pas")

        # Load confs
        with open(configuration_path, 'r', encoding='utf-8') as f:
            configs = json.load(f)

        # Set class vars
        # self.model_name = # On décide de garder le nom créé
        self.model_type = configs['model_type'] if 'model_type' in configs.keys() else self.model_type
        self.x_col = configs['x_col'] if 'x_col' in configs.keys() else self.x_col
        self.y_col = configs['y_col'] if 'y_col' in configs.keys() else self.y_col
        self.columns_in = configs['columns_in'] if 'columns_in' in configs.keys() else self.columns_in
        self.mandatory_columns = configs['mandatory_columns'] if 'mandatory_columns' in configs.keys() else self.mandatory_columns
        # self.model_dir = # On décide de garder le dossier créé
        self.level_save = configs['level_save'] if 'level_save' in configs.keys() else self.level_save
        self.nb_fit = configs['nb_fit'] if 'nb_fit' in configs.keys() else 1 # On considère 1 unique fit par défaut
        self.trained = configs['trained'] if 'trained' in configs.keys() else True # On considère trained par défaut

        # Reload pipeline model
        with open(model_pipeline_path, 'rb') as f:
            self.pipeline = pickle.load(f)

        # Reload pipeline elements
        self.gbt = self.pipeline['gbt']

        # Reload pipeline preprocessing
        with open(preprocess_pipeline_path, 'rb') as f:
            self.preprocess_pipeline = pickle.load(f)


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.error("Ce script ne doit pas être exécuté, il s'agit d'un package.")