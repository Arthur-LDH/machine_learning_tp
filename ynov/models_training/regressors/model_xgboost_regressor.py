#!/usr/bin/env python3

## Modèle Xgboost
# Auteurs : Agence dataservices
# Date : 28/10/2019
#
# Classes :
# - ModelXgboostRegressor -> Modèle pour prédictions via Xgboost - Regression


import os
import json
import logging
import numpy as np
import pandas as pd
import dill as pickle
from datetime import datetime
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from ynov import utils
from ynov.models_training import utils_models
from ynov.models_training.model_class import ModelClass
from ynov.models_training.model_regressor import ModelRegressorMixin


class ModelXgboostRegressor(ModelRegressorMixin, ModelClass):
    '''Modèle pour prédictions via Xgboost - Regression'''

    _default_name = 'model_xgboost_regressor'

    def __init__(self, xgboost_params: dict = {}, early_stopping_rounds: int = 5, validation_split: float = 0.2, **kwargs):
        '''Initialisation de la classe (voir ModelClass & ModelRegressorMixin pour arguments supplémentaires)

        Kwargs:
            xgboost_params (dict): paramètres pour la Xgboost
                -> https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBRegressor
            early_stopping_rounds (int):
            validation_split (float): fraction validation split.
                Utile seulement si pas de jeu de validation en entrée du fit.
        '''
        # Init.
        super().__init__(**kwargs)

        # Get logger (must be done after super init)
        self.logger = logging.getLogger(__name__)

        # Set paramas
        self.xgboost_params = xgboost_params
        self.early_stopping_rounds = early_stopping_rounds
        self.validation_split = validation_split

        # Set objective (if not in params) & init. model
        if 'objective' not in self.xgboost_params.keys():
            self.xgboost_params['objective'] = 'reg:squarederror'
             # list of objectives https://xgboost.readthedocs.io/en/latest/parameter.html#learning-task-parameters
        self.model = XGBRegressor(**self.xgboost_params)

    def fit(self, x_train, y_train, x_valid=None, y_valid=None, with_shuffle: bool = True, **kwargs):
        '''Entrainement du modèle
           **kwargs permet la comptabilité avec les modèles keras

        Args:
            x_train (?): array-like, shape = [n_samples, n_features]
            y_train (?): array-like, shape = [n_samples, n_features]
            x_valid (?): array-like, shape = [n_samples, n_features]
            y_valid (?): array-like, shape = [n_samples, n_features]
        Kwargs:
            with_shuffle (boolean): si x, y doivent être mélangés avant le fit
                Experimental: fonctionnement à vérifier en fonction différents formats x, y
        Raises:
            RuntimeError: si on essaie d'entrainer un modèle déjà fit
        '''
        # TODO: voir si on peut continuer l'entrainement d'un xgboost
        if self.trained:
            self.logger.error("Il n'est pas prévu de pouvoir réentrainer un modèle de type xgboost")
            self.logger.error("Veuillez entrainer un nouveau modèle")
            raise RuntimeError("Impossible de réentrainer un modèle de type pipeline sklearn")

        # On check le format des entrants
        x_train, y_train = self._check_input_format(x_train, y_train, fit_function=True)
        # Si validation, on check aussi le format (mais fit_function à None)
        if y_valid is not None and x_valid is not None:
            x_valid, y_valid = self._check_input_format(x_valid, y_valid, fit_function=False)
        # Sinon, on split random
        else:
            self.logger.warning(f"Attention, pas de jeu de validation. On va donc split le jeu de training (fraction valid = {self.validation_split})")
            x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=self.validation_split)

        # Shuffle x, y if wanted
        if with_shuffle:
            p = np.random.permutation(len(x_train))
            x_train = np.array(x_train)[p]
            y_train = np.array(y_train)[p]
        # Else still transform to numpy array
        else:
            x_train = np.array(x_train)
            y_train = np.array(y_train)

        # Also get x_valid & y_valid as numpy
        x_valid = np.array(x_valid)
        y_valid = np.array(y_valid)


        # Set eval set and train
        eval_set = [(x_train, y_train), (x_valid, y_valid)] # If there’s more than one item in eval_set, the last entry will be used for early stopping.
        prior_objective = self.model.objective
        self.model.fit(x_train, y_train, eval_set=eval_set, early_stopping_rounds=self.early_stopping_rounds, verbose=True)
        post_objective = self.model.objective
        if prior_objective != post_objective:
            self.logger.warning("ATTENTION: la fonction d'objectif à automatiquement été changée par XGBOOST")
            self.logger.warning(f"Avant: {prior_objective}")
            self.logger.warning(f"Après: {post_objective}")

        # Set trained
        self.trained = True
        self.nb_fit += 1

    @utils.trained_needed
    def predict(self, x_test: pd.DataFrame, return_proba: bool = False, **kwargs):
        '''Prédictions sur test

        Args:
            x_test (pd.DataFrame): DataFrame sur laquelle faire les prédictions
        Kwargs:
            return_proba (boolean): pour compatibilité autres models. Retourne une erreur si à True.
        Raises:
            ValueError: si return_proba à True
        Returns:
            (?): array of shape = [n_samples]
        '''
        # Manage errros
        if return_proba == True:
            raise ValueError(f"Les modèles de type model_xgboost_regressor ne gère pas les probabilités")
        # On check le format des entrants
        x_test, _ = self._check_input_format(x_test)
        # Attention, "The method returns the model from the last iteration"
        # Mais : "Predict with X. If the model is trained with early stopping, then best_iteration is used automatically."
        y_pred = self.model.predict(x_test)
        return y_pred

    @utils.trained_needed
    def predict_proba(self, x_test):
        '''Prédictions probabilité sur test

        Args:
            x_test (?): array-like or sparse matrix of shape = [n_samples]
        Raises:
            ValueError: si modèle pas classifier
        Returns:
            (?): array of shape = [n_samples]
        '''
        # Pour compatibilité
        raise ValueError(f"Les modèles de type model_xgboost_regressor n'implémente pas la fonction predict_proba")

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

        json_data['librairie'] = 'xgboost'
        json_data['xgboost_params'] = self.xgboost_params
        json_data['early_stopping_rounds'] = self.early_stopping_rounds
        json_data['validation_split'] = self.validation_split

        # Save xgboost standalone
        if self.level_save in ['MEDIUM', 'HIGH']:
            if self.trained:
                save_path = os.path.join(self.model_dir, f'{self.model_name}.model')
                self.model.save_model(save_path)
            else:
                self.logger.warning("Impossible de sauvegarder le XGboost en standalone car pas encore fitted")

        # Save
        super().save(json_data=json_data)

    def reload_from_standalone(self, configuration_path: str, xgboost_path: str, preprocess_pipeline_path: str, **kwargs):
        '''Fonction pour recharger un modèle à partir de sa configuration et de sa pipeline
        - /!\\ Exploratoire /!\\ -

        Args:
            configuration_path (str): path to configuration
            xgboost_path (str): path to standalone xgboost
            preprocess_pipeline_path (str): path to preprocess pipeline
        Raises:
            ValueError : si configuration_path est à None
            ValueError : si xgboost_path est à None
            FileNotFoundError : si l'objet configuration_path n'est pas un fichier existant
            FileNotFoundError : si l'objet xgboost_path n'est pas un fichier existant
        '''
        if configuration_path is None:
            raise ValueError("L'argument configuration_path ne peut pas être à None")
        if xgboost_path is None:
            raise ValueError("L'argument xgboost_path ne peut pas être à None")
        if preprocess_pipeline_path is None:
            raise ValueError("L'argument preprocess_pipeline_path ne peut pas être à None")
        if not os.path.exists(configuration_path):
            raise FileNotFoundError(f"Le fichier {configuration_path} n'existe pas")
        if not os.path.exists(xgboost_path):
            raise FileNotFoundError(f"Le fichier {xgboost_path} n'existe pas")
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
        # Specific params xgboost
        self.xgboost_params = configs['xgboost_params'] if 'xgboost_params' in configs.keys() else self.xgboost_params
        self.early_stopping_rounds = configs['early_stopping_rounds'] if 'early_stopping_rounds' in configs.keys() else self.early_stopping_rounds
        self.validation_split = configs['validation_split'] if 'validation_split' in configs.keys() else self.validation_split
        # self.model_dir = # On décide de garder le dossier créé
        self.level_save = configs['level_save'] if 'level_save' in configs.keys() else self.level_save
        self.nb_fit = configs['nb_fit'] if 'nb_fit' in configs.keys() else 1 # On considère 1 unique fit par défaut
        self.trained = configs['trained'] if 'trained' in configs.keys() else True # On considère trained par défaut

        # Reload xgboost model
        self.model.load_model(xgboost_path)  # load data

        # Reload pipeline preprocessing
        with open(preprocess_pipeline_path, 'rb') as f:
            self.preprocess_pipeline = pickle.load(f)

    # Idée optimisation bayésienne
    # def fit_with_optim(self, x_train, y_train, x_valid, y_valid, number_init_optim=10 ,number_optim=40, **kwargs):
    #     '''Entrainement du modèle avec optimisation bayesienne
    #     ------------ https://github.com/fmfn/BayesianOptimization -----------
    #         **kwargs permet la comptabilité avec les modèles keras
    #     Args:
    #         x_train (?): array-like or sparse matrix of shape = [n_samples, n_features]
    #         y_train (?): array-like, shape = [n_samples, n_features]
    #     Raises:
    #         RuntimeError: si on essaie d'entrainer un modèle déjà fit
    #         ValueError: si le type de modèle n'est pas classifier ou regressor
    #     '''
    #     if self.trained:
    #         self.logger.error("Il n'est pas prévu de pouvoir réentrainer un modèle de type pipeline sklearn")
    #         self.logger.error("Veuillez entrainer un nouveau modèle")
    #         raise RuntimeError("Impossible de réentrainer un modèle de type pipeline sklearn")
    #
    #     # On check le format des entrants
    #     x_train, y_train = self._check_input_format(x_train, y_train, fit_function=True)
    #
    #
    #     d_train = xgb.DMatrix(data=x_train, label=y_train, feature_names=self.x_col)
    #     d_valid = xgb.DMatrix(data=x_valid, label=y_valid, feature_names=self.x_col)
    #
    #     def model_to_optim(max_depth,subsample,colsample_bytree,gamma,eta):
    #         '''Entrainement du modèle
    #         Args:
    #             x_train (?): array-like or sparse matrix of shape = [n_samples, n_features]
    #             y_train (?): array-like, shape = [n_samples, n_features]
    #         Raises:
    #             RuntimeError: si on essaie d'entrainer un modèle déjà fit
    #             ValueError: si le type de modèle n'est pas classifier ou regressor
    #         '''
    #         if self.trained:
    #             self.logger.error("Il n'est pas prévu de pouvoir réentrainer un modèle de type pipeline sklearn")
    #             self.logger.error("Veuillez entrainer un nouveau modèle")
    #             raise RuntimeError("Impossible de réentrainer un modèle de type pipeline sklearn")
    #
    #         self.xgboost_params["max_depth"] = int(max_depth)
    #         self.xgboost_params["subsample"] = subsample
    #         self.xgboost_params["colsample_bytree"] = colsample_bytree
    #         self.xgboost_params["gamma"] = gamma
    #         self.xgboost_params["eta"] = eta
    #         print(self.xgboost_params)
    #
    #         watchlist = [(d_train, 'train'), (d_valid, 'valid')]
    #         self.model = xgb.train(dtrain=d_train, num_boost_round=self.num_boost_round, evals=watchlist,
    #                     early_stopping_rounds=self.early_stopping_rounds, verbose_eval=self.verbose_eval, params=self.xgboost_params)
    #
    #         return - self.model.best_score
    #
    #     # Bounded region of parameter space
    #     pbounds = {'max_depth': (3, 15), 'subsample': (0.1, 0.99),
    #                'colsample_bytree': (0.1, 0.99), 'gamma': (0.1, 10000),
    #                'eta':(0.1,0.2)}
    #
    #     optimizer = BayesianOptimization(
    #         f=model_to_optim,
    #         pbounds=pbounds,
    #         random_state=1,
    #     )
    #
    #     optimizer.maximize(
    #         init_points=number_init_optim,
    #         n_iter=number_optim,
    #     )
    #
    #     self.xgboost_params["max_depth"] = int(optimizer.max["params"]["max_depth"])
    #     self.xgboost_params["subsample"] = optimizer.max["params"]["subsample"]
    #     self.xgboost_params["colsample_bytree"] = optimizer.max["params"]["colsample_bytree"]
    #     self.xgboost_params["gamma"] = optimizer.max["params"]["gamma"]
    #     self.xgboost_params["eta"] = optimizer.max["params"]["eta"]
    #
    #     watchlist = [(d_train, 'train'), (d_valid, 'valid')]
    #     self.model = xgb.train(dtrain=d_train, num_boost_round=self.num_boost_round, evals=watchlist,
    #                  early_stopping_rounds=self.early_stopping_rounds, verbose_eval=self.verbose_eval, params=self.xgboost_params)
    #
    #     # Set trained
    #     self.trained = True
    #     self.nb_fit += 1
    #
    #     #print(self.model.best_score)
    #     return self.xgboost_params, optimizer.res


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.error("Ce script ne doit pas être exécuté, il s'agit d'un package.")