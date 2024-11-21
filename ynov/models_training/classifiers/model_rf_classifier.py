#!/usr/bin/env python3

## Modèle Random Forest
# Auteurs : Agence dataservices
# Date : 28/10/2019
#
# Classes :
# - ModelRFClassifier -> Modèle pour prédictions via Random Forest - Classification


import os
import json
import logging
import numpy as np
import pandas as pd
import dill as pickle
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from ynov import utils
from ynov.models_training import utils_models
from ynov.models_training.model_pipeline import ModelPipeline
from ynov.models_training.model_classifier import ModelClassifierMixin


class ModelRFClassifier(ModelClassifierMixin, ModelPipeline):
    '''Modèle pour prédictions via Random Forest - Classification'''

    _default_name = 'model_rf_classifier'

    def __init__(self, rf_params: dict = {}, multiclass_strategy: str = None, **kwargs):
        '''Initialisation de la classe (voir ModelPipeline, ModelClass & ModelClassifierMixin pour arguments supplémentaires)

        Kwargs:
            rf_params (dict) : paramètres pour le Random Forest
            multiclass_strategy (str): stratégie multiclass, 'ovr' (OneVsRest), ou 'ovo' (OneVsOne). Si None, on laisse l'algo tel quel.
        Raises:
            ValueError: si multiclass_strategy n'est pas 'ovo' ou 'ovr' (si pas None)
        '''
        if multiclass_strategy is not None and multiclass_strategy not in ['ovo', 'ovr']:
            raise ValueError(f"La valeur de 'multiclass_strategy' doit être 'ovo' ou 'ovr' (pas {multiclass_strategy})")
        # Init.
        super().__init__(**kwargs)

        # Get logger (must be done after super init)
        self.logger = logging.getLogger(__name__)

        # Gestion modèles
        self.rf = RandomForestClassifier(**rf_params)
        self.multiclass_strategy = multiclass_strategy

        # On ne gère pas le multilabel / mutliclass
        if not self.multi_label:
            # Si pas multiclass, pas d'impact
            if multiclass_strategy == 'ovr':
                self.pipeline = Pipeline([('rf', OneVsRestClassifier(self.rf))])
            elif multiclass_strategy == 'ovo':
                self.pipeline = Pipeline([('rf', OneVsOneClassifier(self.rf))])
            else:
                self.pipeline = Pipeline([('rf', self.rf)])

        # Le RandomForest supporte nativement le multi_label
        if self.multi_label:
            self.pipeline = Pipeline([('rf', self.rf)])

    @utils.trained_needed
    def predict_proba(self, x_test: pd.DataFrame, **kwargs):
        '''Prédictions probabilité sur test
            'ovo' ne peut pas prédire de probas. Par défaut on retourne 1. si classe pred, sinon 0.

        Args:
            x_test (pd.DataFrame): DataFrame sur laquelle faire les prédictions
        Returns:
            (?): array of shape = [n_samples]
        '''
        # Utilisation fonction super() de la classe pipeline si != 'ovo' ou multi label
        if self.multi_label or self.multiclass_strategy != 'ovo':
            return super().predict_proba(x_test=x_test, **kwargs)
        else:
            preds = self.pipeline.predict(x_test)
            # Format ['a', 'b', 'c', 'a', ..., 'b']
            # Transform to "proba"
            transform_dict = {col: [0. if _ != i else 1. for _ in range(len(self.list_classes))] for i, col in enumerate(self.list_classes)}
            probas = np.array([transform_dict[x] for x in preds])
        return probas

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

        json_data['multiclass_strategy'] = self.multiclass_strategy

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
            ValueError : si preprocess_pipeline_path est à None
            FileNotFoundError : si l'objet configuration_path n'est pas un fichier existant
            FileNotFoundError : si l'objet model_pipeline_path n'est pas un fichier existant
            FileNotFoundError : si l'objet preprocess_pipeline_path n'est pas un fichier existant
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
        # Can't set int as keys in json, so need to cast it after realoading
        # dict_classes keys are always ints
        if 'dict_classes' in configs.keys():
            configs['dict_classes'] = {int(k): v for k, v in configs['dict_classes'].items()}
        elif 'list_classes' in configs.keys():
            configs['dict_classes'] = {i: col for i, col in enumerate(configs['list_classes'])}

        # Set class vars
        # self.model_name = # On décide de garder le nom créé
        self.model_type = configs['model_type'] if 'model_type' in configs.keys() else self.model_type
        self.x_col = configs['x_col'] if 'x_col' in configs.keys() else self.x_col
        self.y_col = configs['y_col'] if 'y_col' in configs.keys() else self.y_col
        self.columns_in = configs['columns_in'] if 'columns_in' in configs.keys() else self.columns_in
        self.mandatory_columns = configs['mandatory_columns'] if 'mandatory_columns' in configs.keys() else self.mandatory_columns
        # self.model_dir = # On décide de garder le dossier créé
        self.list_classes = configs['list_classes'] if 'list_classes' in configs.keys() else self.list_classes
        self.dict_classes = configs['dict_classes'] if 'dict_classes' in configs.keys() else self.dict_classes
        self.multi_label = configs['multi_label'] if 'multi_label' in configs.keys() else self.multi_label
        self.level_save = configs['level_save'] if 'level_save' in configs.keys() else self.level_save
        self.nb_fit = configs['nb_fit'] if 'nb_fit' in configs.keys() else 1 # On considère 1 unique fit par défaut
        self.trained = configs['trained'] if 'trained' in configs.keys() else True # On considère trained par défaut
        self.multiclass_strategy = configs['multiclass_strategy'] if 'multiclass_strategy' in configs.keys() else self.multiclass_strategy

        # Reload pipeline model
        with open(model_pipeline_path, 'rb') as f:
            self.pipeline = pickle.load(f)

        # Gestion multilabel ou multiclass
        if not self.multi_label and self.multiclass_strategy is not None:
            self.rf = self.pipeline['rf'].estimator
        else:
            self.rf = self.pipeline['rf']

        # Reload pipeline preprocessing
        with open(preprocess_pipeline_path, 'rb') as f:
            self.preprocess_pipeline = pickle.load(f)


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.error("Ce script ne doit pas être exécuté, il s'agit d'un package.")