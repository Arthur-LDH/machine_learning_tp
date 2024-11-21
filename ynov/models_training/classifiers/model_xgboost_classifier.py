#!/usr/bin/env python3

## Modèle Xgboost
# Auteurs : Agence dataservices
# Date : 28/10/2019
#
# Classes :
# - ModelXgboostClassifier -> Modèle pour prédictions via Xgboost - Classification


import os
import json
import logging
import numpy as np
import pandas as pd
import dill as pickle
from datetime import datetime
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.utils.validation import _check_fit_params, _deprecate_positional_args
from sklearn.utils.multiclass import check_classification_targets
from sklearn.base import is_classifier
from joblib import Parallel, delayed
from sklearn.multioutput import _fit_estimator
from ynov import utils
from ynov.models_training import utils_models
from ynov.models_training.model_class import ModelClass
from ynov.models_training.model_classifier import ModelClassifierMixin


class ModelXgboostClassifier(ModelClassifierMixin, ModelClass):
    '''Modèle pour prédictions via Xgboost - Classification'''

    _default_name = 'model_xgboost_classifier'

    def __init__(self, xgboost_params: dict = {}, early_stopping_rounds: int = 5, validation_split: float = 0.2, **kwargs):
        '''Initialisation de la classe (voir ModelClass & ModelClassifierMixin pour arguments supplémentaires)

        Kwargs:
            xgboost_params (dict): paramètres pour la Xgboost
                -> https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBClassifier
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
            self.xgboost_params['objective'] = 'binary:logistic'
             # list of objectives https://xgboost.readthedocs.io/en/latest/parameter.html#learning-task-parameters
        # ATTENTION, si multiclass, backup AUTOMATIQUE sur multi:softprob (par xgboost)
        # https://stackoverflow.com/questions/57986259/multiclass-classification-with-xgboost-classifier
        self.model = XGBClassifier(**self.xgboost_params)

        # Si multilabel, on utilise MultiOutputClassifier
        if self.multi_label:
            self.model = MyMultiOutputClassifier(self.model)

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

        # Récupération des colonnes en entrées pour la suite
        if hasattr(y_train, 'columns'):
            original_list_classes = list(y_train.columns)
        else:
            original_list_classes = None

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
        prior_objective = self.model.objective if not self.multi_label else self.model.estimator.objective
        self.model.fit(x_train, y_train, eval_set=eval_set, early_stopping_rounds=self.early_stopping_rounds, verbose=True)
        post_objective = self.model.objective if not self.multi_label else self.model.estimator.objective
        if prior_objective != post_objective:
            self.logger.warning("ATTENTION: la fonction d'objectif à automatiquement été changée par XGBOOST")
            self.logger.warning(f"Avant: {prior_objective}")
            self.logger.warning(f"Après: {post_objective}")

        # Set list classes
        if not self.multi_label:
            self.list_classes = list(self.model.classes_)
        else:
            if original_list_classes is not None:
                self.list_classes = original_list_classes
            else:
                self.logger.warning(
                    "Impossible de lire l'information sur le nom des colonnes de y_train -> la transformation inverse ne sera pas possible"
                )
                # On créé quand même une liste de classes pour être compatible avec les autres fonctions
                self.list_classes = [str(_) for _ in range(pd.DataFrame(y_train).shape[1])]

        # Set dict_classes based on list classes
        self.dict_classes = {i: col for i, col in enumerate(self.list_classes)}

        # Set trained
        self.trained = True
        self.nb_fit += 1

    @utils.trained_needed
    def predict(self, x_test: pd.DataFrame, return_proba: bool = False, **kwargs):
        '''Prédictions sur test

        Args:
            x_test (pd.DataFrame): DataFrame sur laquelle faire les prédictions
        Kwargs:
            return_proba (boolean): si la fonction doit retourner les probas au lieu des classes (comptabilité keras)
        Returns:
            (?): array of shape = [n_samples]
        '''
        # Si on veut des probas, on utilise predict_proba
        if return_proba:
            return self.predict_proba(x_test, **kwargs)
        # Sinon, on retourne les predictions :
        else:
            # On check le format des entrants
            x_test, _ = self._check_input_format(x_test)
            # Attention, "The method returns the model from the last iteration"
            # Mais : "Predict with X. If the model is trained with early stopping, then best_iteration is used automatically."
            y_pred = self.model.predict(x_test)
            return y_pred

    @utils.trained_needed
    def predict_proba(self, x_test: pd.DataFrame, **kwargs):
        '''Prédictions probabilité sur test

        Args:
            x_test (pd.DataFrame): DataFrame sur laquelle faire les prédictions
        Returns:
            (?): array of shape = [n_samples]
        '''
        # On check le format des entrants
        x_test, _ = self._check_input_format(x_test)

        #
        probas = np.array(self.model.predict_proba(x_test))
        # Si utilisation de MultiOutputClassifier -> retourne proba de 0 et proba de 1 pour tous les éléments et toutes les classes
        # Correction dans le cas où on détecte une shape > 2 (i.e. égale à 3)
        # Rappel : on ne gère pas le multilabel multiclass
        if len(probas.shape) > 2:
            probas = np.swapaxes(probas[:, :, 1], 0, 1)
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

        json_data['librairie'] = 'xgboost'
        json_data['xgboost_params'] = self.xgboost_params
        json_data['early_stopping_rounds'] = self.early_stopping_rounds
        json_data['validation_split'] = self.validation_split

        # Save xgboost standalone
        if self.level_save in ['MEDIUM', 'HIGH']:
            if not self.multi_label:
                if self.trained:
                    save_path = os.path.join(self.model_dir, f'{self.model_name}.model')
                    self.model.save_model(save_path)
                else:
                    self.logger.warning("Impossible de sauvegarder le XGboost en standalone car pas encore fitted")
            else:
                # Si multilabel, on utilise un multioutput, et on fit plusieurs xgboost au final (cf. strategy sklearn)
                # Du coup on ne peut pas sauvegarder un seul xgboost, donc on sauvegarde en pkl
                # Problème : on ne sera pas compatible avec les montées de versions :'(
                save_path = os.path.join(self.model_dir, f"{self.model_name}.pkl")
                with open(save_path, 'wb') as f:
                    pickle.dump(self.model, f)

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
            ValueError : si preprocess_pipeline_path est à None
            FileNotFoundError : si l'objet configuration_path n'est pas un fichier existant
            FileNotFoundError : si l'objet xgboost_path n'est pas un fichier existant
            FileNotFoundError : si l'objet preprocess_pipeline_path n'est pas un fichier existant
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
        # Specific params xgboost
        self.xgboost_params = configs['xgboost_params'] if 'xgboost_params' in configs.keys() else self.xgboost_params
        self.early_stopping_rounds = configs['early_stopping_rounds'] if 'early_stopping_rounds' in configs.keys() else self.early_stopping_rounds
        self.validation_split = configs['validation_split'] if 'validation_split' in configs.keys() else self.validation_split
        # self.model_dir = # On décide de garder le dossier créé
        self.list_classes = configs['list_classes'] if 'list_classes' in configs.keys() else self.list_classes
        self.dict_classes = configs['dict_classes'] if 'dict_classes' in configs.keys() else self.dict_classes
        self.multi_label = configs['multi_label'] if 'multi_label' in configs.keys() else self.multi_label
        self.level_save = configs['level_save'] if 'level_save' in configs.keys() else self.level_save
        self.nb_fit = configs['nb_fit'] if 'nb_fit' in configs.keys() else 1 # On considère 1 unique fit par défaut
        self.trained = configs['trained'] if 'trained' in configs.keys() else True # On considère trained par défaut

        # Reload xgboost model
        if not self.multi_label:
            self.model.load_model(xgboost_path)
        else:
            with open(xgboost_path, 'rb') as f:
                self.model = pickle.load(f)

        # Reload pipeline preprocessing
        with open(preprocess_pipeline_path, 'rb') as f:
            self.preprocess_pipeline = pickle.load(f)


# Problème : On veut utiliser le MultiOutputClassifier pour les cas multilabels, mais le comportement par
# défaut est d'envoyer les mêmes paramètres à chaque fit. Cependant, pour le XGboost, on veut valider
# sur le bon label (erreur si on fait rien, car pas bon format).
# Solution : on modifie la classe MultiOutputClassifier pour faire en sorte d'envoyer le bon subset de y_valid
# a chaque fit
# From : https://stackoverflow.com/questions/66785587/how-do-i-use-validation-sets-on-multioutputregressor-for-xgbregressor
# From : https://github.com/scikit-learn/scikit-learn/blob/2beed5584/sklearn/multioutput.py#L293
# From : https://github.com/scikit-learn/scikit-learn/blob/2beed55847ee70d363bdbfe14ee4401438fba057/sklearn/multioutput.py#L64
class MyMultiOutputClassifier(MultiOutputClassifier):

    @_deprecate_positional_args
    def __init__(self, estimator, *, n_jobs=None):
        super().__init__(estimator, n_jobs=n_jobs)

    def fit(self, X, y, sample_weight=None, **fit_params):
        """ Fit the model to data.
        Fit a separate model for each output variable.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Data.
        y : {array-like, sparse matrix} of shape (n_samples, n_outputs)
            Multi-output targets. An indicator matrix turns on multilabel
            estimation.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.
            Only supported if the underlying regressor supports sample
            weights.
        **fit_params : dict of string -> object
            Parameters passed to the ``estimator.fit`` method of each step.
            .. versionadded:: 0.23
        Returns
        -------
        self : object
        """
        if not hasattr(self.estimator, "fit"):
            raise ValueError("The base estimator should implement"
                             " a fit method")

        X, y = self._validate_data(X, y,
                                   force_all_finite=False,
                                   multi_output=True, accept_sparse=True)

        if is_classifier(self):
            check_classification_targets(y)

        if y.ndim == 1:
            raise ValueError("y must have at least two dimensions for "
                             "multi-output regression but has only one.")

        if (sample_weight is not None and
                not has_fit_parameter(self.estimator, 'sample_weight')):
            raise ValueError("Underlying estimator does not support"
                             " sample weights.")

        fit_params_validated = _check_fit_params(X, fit_params)

        # New : extract eval_set
        if 'eval_set' in fit_params_validated.keys():
            eval_set = fit_params_validated.pop('eval_set')
            self.estimators_ = Parallel(n_jobs=self.n_jobs)(
                delayed(_fit_estimator)(
                    self.estimator, X, y[:, i], sample_weight,
                    **fit_params_validated,
                    eval_set=[(X_test, Y_test[:, i]) for X_test, Y_test in eval_set])
                for i in range(y.shape[1]))
        # Pas d'eval_set
        else:
            self.estimators_ = Parallel(n_jobs=self.n_jobs)(
                delayed(_fit_estimator)(
                    self.estimator, X, y[:, i], sample_weight,
                    **fit_params_validated)
                for i in range(y.shape[1]))
        return self


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.error("Ce script ne doit pas être exécuté, il s'agit d'un package.")