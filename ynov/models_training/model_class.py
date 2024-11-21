#!/usr/bin/env python3

## Définition d'une classe parent pour les modèles
# Auteurs : Agence dataservices
# Date : 07/04/2021
#
# Classes :
# - ModelClass -> Classe parent modèle


import os
import re
import json
import dill as pickle
import logging
import numpy as np
import pandas as pd
from typing import List
from datetime import datetime
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from ynov import utils
from ynov.preprocessing import preprocess
from ynov.models_training import utils_models
from ynov.monitoring.model_logger import ModelLogger


class ModelClass:
    '''Classe parent pour les modèles'''

    _default_name = 'none'

    # Not implemented :
    # -> fit
    # -> predict
    # -> predict_proba
    # -> inverse_transform
    # -> get_and_save_metrics

    def __init__(self, model_dir: str = None, model_name: str = None, x_col: list = None, y_col=None,
                 preprocess_pipeline: ColumnTransformer = None, level_save: str = 'HIGH'):
        '''Initialisation de la classe parent

        Kwargs:
            model_dir (str): dossier où sauvegarder le modèle
            model_name (str): nom du modèle
            x_col (list): nom des colonnes utilisées pour l'apprentissage - x
            y_col (str ou int ou list si multi-label): nom de la ou des colonnes utilisées pour l'apprentissage - y
            preprocess_pipeline (ColumnTransformer): pipeline de preprocessing utilisée. Si None -> pas de preprocessing !
            level_save (str): Niveau de sauvegarde
                LOW: statistiques + configurations + logger keras - /!\\ modèle non réutilisable /!\\ -
                MEDIUM: LOW + hdf5 + pkl + plots
                HIGH: MEDIUM + predictions
        Raises:
            TypeError: si l'objet x_col list
            TypeError: si l'objet y_col n'est pas du type str ou int ou list
            ValueError : si l'objet level_save n'est pas une option valable (['LOW', 'MEDIUM', 'HIGH'])
        '''
        if x_col is not None and type(x_col) != list:
            raise TypeError('L\'objet x_col doit être du type list.')
        if y_col is not None and type(y_col) not in (str, int, list):
            raise TypeError('L\'objet y_col doit être du type str ou int ou list.')
        if level_save not in ['LOW', 'MEDIUM', 'HIGH']:
            raise ValueError(f"L'objet level_save ({level_save}) n'est pas une option valide (['LOW', 'MEDIUM', 'HIGH'])")

        # Get logger
        self.logger = logging.getLogger(__name__)

        # Type de model -> 'classifier' ou 'regressor' en fonction de l'application
        self.model_type = None

        # Nom modèle
        if model_name is None:
            self.model_name = self._default_name
        else:
            self.model_name = model_name

        # Nom des colonne utilisées
        self.x_col = x_col
        self.y_col = y_col
        if x_col is None:
            self.logger.warning("Attention, l'attribut x_col n'est pas renseigné ! Fonctionnement non garanti.")
        if y_col is None:
            self.logger.warning("Attention, l'attribut y_col n'est pas renseigné ! Fonctionnement non garanti.")

        # Dossier modèle
        if model_dir is None:
            self.model_dir = self._get_model_dir()
        else:
            if not os.path.isdir(model_dir):
                os.mkdir(model_dir)
            self.model_dir = os.path.abspath(model_dir)

        # Pipeline de preprocessing
        self.preprocess_pipeline = preprocess_pipeline
        if self.preprocess_pipeline is not None:
            try:
                check_is_fitted(self.preprocess_pipeline)
            except NotFittedError as e:
                self.logger.error("La pipeline de preprocessing n'est pas fit !")
                self.logger.error(repr(e))
                raise NotFittedError()
            # On récupère les colonnes associées (un check si fitted est réalisé)
            self.columns_in, self.mandatory_columns = utils_models.get_columns_pipeline(self.preprocess_pipeline)
        else:
            # On ne peut pas définir une pipeline "no_preprocess" puisqu'il faudrait la fit
            # Du coup, on se rattrape au permier fit
            self.logger.warning("Attention, aucune pipeline de preprocessing renseignée !")
            self.columns_in, self.mandatory_columns = None, None

        # Other options
        self.level_save = level_save

        # is trained ?
        self.trained = False
        self.nb_fit = 0

    def fit(self, x_train, y_train):
        '''Entrainement du modèle

        Args:
            x_train (?): array-like or sparse matrix of shape = [n_samples, n_features]
            y_train (?): array-like, shape = [n_samples, n_features]
        '''
        raise NotImplementedError("'fit' needs to be overrided")

    def predict(self, x_test: pd.DataFrame, **kwargs):
        '''Prédictions sur test

        Args:
            x_test (pd.DataFrame): DataFrame sur laquelle faire les prédictions
        Returns:
            (?): array of shape = [n_samples]
        '''
        raise NotImplementedError("'predict' needs to be overrided")

    def predict_proba(self, x_test: pd.DataFrame, **kwargs):
        '''Prédictions probabilité sur test -

        Args:
            x_test (pd.DataFrame): DataFrame sur laquelle faire les prédictions
        Returns:
            (?): array of shape = [n_samples]
        '''
        raise NotImplementedError("'predict_proba' needs to be overrided")

    def inverse_transform(self, y):
        '''Fonction pour obtenir format final de prédiction
            - Classification : classes depuis preds
            - Regression : valeurs (fonction identité)

        Args:
            y (?): array-like, shape = [n_samples, n_features]
        Returns:
            (?): array of shape = [n_samples, ?]
        '''
        raise NotImplementedError("'inverse_transform' needs to be overrided")

    def get_and_save_metrics(self, y_true, y_pred, df_x=None, series_to_add: List[pd.Series] = None, type_data: str = '', model_logger=None):
        '''Fonction pour obtenir et sauvegarder les métriques d'un modèle

        Args:
            y_true (?): array-like, shape = [n_samples, n_features]
            y_pred (?): array-like, shape = [n_samples, n_features]
        Kwargs:
            df_x (?): DataFrame en entrée de la prédiction
            series_to_add (list): liste de pd.Series à ajouter à la dataframe
            type_data (str): type du dataset (validation, test, ...)
            model_logger (ModelLogger): classe custom pour logger les métriques dans ML Flow
        Raises:
            TypeError: si l'objet series_to_add n'est pas du type list, et composé d'éléments de type pd.Series
        Returns:
            pd.DataFrame: la df qui contient les statistiques
        '''
        raise NotImplementedError("'get_and_save_metrics' needs to be overrided")

    def save(self, json_data: dict = None):
        '''Sauvegarde du modèle

        Kwargs:
            json_data (dict): configuration à ajouter pour la sauvegarde JSON
        '''

        # Gestion paths
        pkl_path = os.path.join(self.model_dir, f"{self.model_name}.pkl")
        pipeline_pkl_path = os.path.join(self.model_dir, "preprocess_pipeline.pkl")
        conf_path = os.path.join(self.model_dir, "configurations.json")

        # Sauvegarde model & pipeline preprocessing si level_save > 'LOW'
        if self.level_save in ['MEDIUM', 'HIGH']:
            # TODO: utiliser dill pour ne plus avoir de problème de "can't pickle ..." ?
            with open(pkl_path, 'wb') as f:
                pickle.dump(self, f)
            # Utile pour reload_from_standalone, sinon sauvegardée en tant qu'attribut de la classe
            with open(pipeline_pkl_path, 'wb') as f:
                pickle.dump(self.preprocess_pipeline, f)

        # Save configuration JSON
        json_dict = {
            'mainteners': 'Agence DataServices',
            'date': datetime.now().strftime("%d/%m/%Y - %H:%M:%S"),  # Pas la même que le nom du dossier du coup
            'package_version': utils.get_package_version(),
            'model_name': self.model_name,
            'model_dir': self.model_dir,
            'model_type': self.model_type,
            'trained': self.trained,
            'nb_fit': self.nb_fit,
            'x_col': self.x_col,
            'y_col': self.y_col,
            'columns_in': self.columns_in,
            'mandatory_columns': self.mandatory_columns,
            'level_save': self.level_save,
            'librairie': None,
        }
        # Merge json_data if not None
        if json_data is not None:
            # On priorise json_data !
            json_dict = {**json_dict, **json_data}

        # Save conf
        with open(conf_path, 'w', encoding='utf-8') as f:
            json.dump(json_dict, f, indent=4, cls=utils.NpEncoder)

        # Now, save a proprietes file for artifactory export
        self._save_proprietes_artifactory(json_dict)

    def _save_proprietes_artifactory(self, json_dict: dict = {}):
        '''Fonction pour préparer un fichier de conf pour un futur export sur l'artifactory

        Kwargs:
            json_dict: configurations à save
        '''

        # Gestion paths
        proprietes_path = os.path.join(self.model_dir, "proprietes.json")
        artifactory_vanilla_instructions = os.path.join(utils.get_ressources_path(), 'artifactory_instructions.md')
        artifactory_specific_instructions = os.path.join(self.model_dir, "artifactory_instructions.md")

        # First, we define a list of "allowed" properties
        allowed_properties = ["mainteners", "date", "package_version", "model_name", "list_classes",
                              "librairie", "fit_time"]
        # Now we filter these properties
        final_dict = {k: v for k, v in json_dict.items() if k in allowed_properties}
        # Save
        with open(proprietes_path, 'w', encoding='utf-8') as f:
            json.dump(final_dict, f, indent=4, cls=utils.NpEncoder)

        # Add instructions to upload a model to artifactory
        with open(artifactory_vanilla_instructions, 'r', encoding='utf-8') as f:
            content = f.read()
        new_content = content.replace('model_dir_path_identifier', os.path.abspath(self.model_dir))
        with open(artifactory_specific_instructions, 'w', encoding='utf-8') as f:
            f.write(new_content)

    def _get_model_dir(self):
        '''Fonction pour récupérer un dossier où sauvegarder le modèle

        Returns:
            str: path vers le dossier
        '''
        models_dir = utils.get_models_path()
        subfolder = os.path.join(models_dir, self.model_name)
        folder_name = datetime.now().strftime(f"{self.model_name}_%Y_%m_%d-%H_%M_%S")
        model_dir = os.path.join(subfolder, folder_name)
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)
        return model_dir

    def _check_input_format(self, x_input, y_input = None, fit_function: bool = False):
        '''Fonction pour vérifier l'intégrité des entrants d'une fonction
        On check le bon nombre de colonnes et on reorder
        Warnings si :
            - Pas les bonnes colonnes
            - Colonnes pas dans le bon ordre
        Si fit & x_col et/ou y_col not defined -> warning & on utilise les colonnes en entrées
        On en profite aussi pour set pipeline, columns_in et mandatory_columns si à None

        Args:
            x_input (?): array-like, shape = [n_samples, n_features]
        Kwargs:
            y_input (?): array-like, shape = [n_samples, n_features]
                Obligatoire si fit_function
            fit_function (bool): s'il s'agit d'une fonction de fit
        Raises:
            AttributeError: si fit_function == True, mais y_input à None
            ValueError: si un des inputs n'a pas le bon nombre de colonnes
        Returns:
            ?: x_input, éventuellement reordered si besoin
            ?: y_input, éventuellement reordered si besoin
        '''
        # Récupération de certaines information en premier
        x_input_shape = x_input.shape[-1] if len(x_input.shape) > 1 else 1
        if y_input is not None:
            y_input_shape = y_input.shape[-1] if len(y_input.shape) > 1 else 1
        else:
            y_input_shape = 0 # non utilisé

        # Gestion fit_function = True
        if fit_function == True:
            if y_input is None:
                raise AttributeError("L'argument y_input est obligatoire avec fit_function == True")
            if self.x_col is None:
                self.logger.warning("Attention, l'attribut x_col n'a pas été set à la création du modèle")
                self.logger.warning("On le set maintenant avec les données en entrées de la fonction fit")
                if hasattr(x_input, 'columns'):
                    self.x_col = list(x_input.columns)
                else:
                    self.x_col = [_ for _ in range(x_input_shape)]
            # On fait pareil pour y_col
            if self.y_col is None:
                self.logger.warning("Attention, l'attribut y_col n'a pas été set à la création du modèle")
                self.logger.warning("On le set maintenant avec les données en entrées de la fonction fit")
                if hasattr(y_input, 'columns'):
                    self.y_col = list(y_input.columns)
                else:
                    self.y_col = [_ for _ in range(y_input_shape)]
                # Si 1 seul élément -> on enlève la liste
                if y_input_shape == 1:
                    self.y_col = self.y_col[0]
            # On set pipeline & columns_in & mandatory_columns si à None
            if self.preprocess_pipeline is None: # i.e. pas de pipeline précisée à l'init. de la classe
                preprocess_str = "no_preprocess"
                preprocess_pipeline = preprocess.get_pipeline(preprocess_str) # Attention, besoin d'être fit
                preprocess_pipeline.fit(x_input) # On fit pour set les colonnes nécessaires à la pipeline
                self.preprocess_pipeline = preprocess_pipeline
                self.columns_in, self.mandatory_columns = utils_models.get_columns_pipeline(self.preprocess_pipeline)

        # Vérifications x_input
        if self.x_col is None:
            self.logger.warning("Impossible de vérifier le format d'entrée (x) car x_col n'est pas set...")
        else:
            # On check le format de x_input
            x_col_len = len(self.x_col)
            if x_input_shape != x_col_len:
                raise ValueError(f"Les données en entrées (x) n'ont pas le bon format ({x_input_shape} != {x_col_len})")
            # On check la présence des colonnes
            if hasattr(x_input, 'columns'):
                can_reorder = True
                for col in self.x_col:
                    if col not in x_input.columns:
                        can_reorder = False
                        self.logger.warning(f"La colonne {col} est manquante dans les données en entrées (x)")
                # Si on ne peut pas reorder, message warning, sinon on check si besoin
                if not can_reorder:
                    self.logger.warning("On est pas ISO sur le nom des colonnes, mais on continue car bon nombre de colonnes")
                else:
                    if list(x_input.columns) != self.x_col:
                        self.logger.warning("Les colonnes des données en entrées (x) ne sont pas dans le bon ordre -> reorder automatique !")
                        x_input = x_input[self.x_col]
            else:
                self.logger.warning(f"Les données en entrées (x) n'expose pas l'attribut 'columns' -> impossible de vérifier l'ordre des colonnes")

        # Vérifications y_input
        if y_input is not None:
            if self.y_col is None:
                self.logger.warning("Impossible de vérifier le format d'entrée (y) car y_col n'est pas set...")
            else:
                # On check le format de y_input
                y_col_len = len(self.y_col) if type(self.y_col) == list else 1
                if y_input_shape != y_col_len:
                    raise ValueError(f"Les données en entrées (y) n'ont pas le bon format ({y_input_shape} != {y_col_len})")
                # On check la présence des colonnes
                if hasattr(y_input, 'columns'):
                    can_reorder = True
                    for col in self.y_col:
                        if col not in y_input.columns:
                            can_reorder = False
                            self.logger.warning(f"La colonne {col} est manquante dans les données en entrées (y)")
                    # Si on ne peut pas reorder, message warning, sinon on check si besoin
                    if not can_reorder:
                        self.logger.warning("On est pas ISO sur le nom des colonnes, mais on continue car bon nombre de colonnes")
                    else:
                        if list(y_input.columns) != self.y_col:
                            self.logger.warning("Les colonnes des données en entrées (y) ne sont pas dans le bon ordre -> reorder automatique !")
                            y_input = y_input[self.y_col]
                else:
                    self.logger.warning(f"Les données en entrées (y) n'expose pas l'attribut 'columns' -> impossible de vérifier l'ordre des colonnes")

        # Return
        return x_input, y_input

    def display_if_gpu_activated(self):
        '''Fonction pour afficher si on utilise un GPU'''
        if self._is_gpu_activated():
            ascii_art = '''
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
*         (=========)                                                                                                            (=========)         *
*         |=========|                                                                                                            |=========|         *
*         |====_====|                                                                                                            |====_====|         *
*         |== / \ ==|                                                                                                            |== / \ ==|         *
*         |= / _ \ =|                                                                                                            |= / _ \ =|         *
*      _  |=| ( ) |=|                                                                                                         _  |=| ( ) |=|         *
*     /=\ |=|     |=| /=\                                                                                                    /=\ |=|     |=| /=\     *
*     |=| |=| GPU |=| |=|        _____ _____  _    _            _____ _______ _______      __  _______ ______ _____          |=| |=| GPU |=| |=|     *
*     |=| |=|  _  |=| |=|       / ____|  __ \| |  | |     /\   / ____|__   __|_   _\ \    / /\|__   __|  ____|  __ \         |=| |=|  _  |=| |=|     *
*     |=| |=| | | |=| |=|      | |  __| |__) | |  | |    /  \ | |       | |    | |  \ \  / /  \  | |  | |__  | |  | |        |=| |=| | | |=| |=|     *
*     |=| |=| | | |=| |=|      | | |_ |  ___/| |  | |   / /\ \| |       | |    | |   \ \/ / /\ \ | |  |  __| | |  | |        |=| |=| | | |=| |=|     *
*     |=| |=| | | |=| |=|      | |__| | |    | |__| |  / ____ \ |____   | |   _| |_   \  / ____ \| |  | |____| |__| |        |=| |=| | | |=| |=|     *
*     |=| |/  | |  \| |=|       \_____|_|     \____/  /_/    \_\_____|  |_|  |_____|   \/_/    \_\_|  |______|_____/         |=| |/  | |  \| |=|     *
*     |=|/    | |    \|=|                                                                                                    |=|/    | |    \|=|     *
*     |=/ ADS |_| ADS \=|                                                                                                    |=/ ADS |_| ADS \=|     *
*     |(_______________)|                                                                                                    |(_______________)|     *
*     |=| |_|__|__|_| |=|                                                                                                    |=| |_|__|__|_| |=|     *
*     |=|   ( ) ( )   |=|                                                                                                    |=|   ( ) ( )   |=|     *
*    /===\           /===\                                                                                                  /===\           /===\    *
*   |||||||         |||||||                                                                                                |||||||         |||||||   *
*   -------         -------                                                                                                -------         -------   *
*    (~~~)           (~~~)                                                                                                  (~~~)           (~~~)    *
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
            '''
        else:
            ascii_art = ''
        print(ascii_art)

    def _is_gpu_activated(self):
        '''Fonction pour vérifier si on utilise un GPU

        Returns:
            bool: whether GPU is available or not
        '''
        # Par défaut, pas de GPU
        return False

if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.error("Ce script ne doit pas être exécuté, il s'agit d'un package.")