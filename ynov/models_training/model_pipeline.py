#!/usr/bin/env python3

## Modèle générique pour pipeline sklearn
# Auteurs : Agence dataservices
# Date : 28/10/2019
#
# Classes :
# - ModelPipeline -> Modèle générique pour pipeline sklearn


import logging
import os
import dill as pickle
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from ynov import utils
from ynov.models_training import utils_models
from ynov.models_training.model_class import ModelClass


class ModelPipeline(ModelClass):
    '''Modèle générique pour pipeline sklearn'''

    _default_name = 'model_pipeline'

    # Not implemented :
    # -> reload

    def __init__(self, pipeline=None, **kwargs):
        '''Initialisation de la classe (voir ModelClass pour arguments supplémentaires)

        Kwargs:
            pipeline (Pipeline): pipeline à utiliser.
        '''
        # Init.
        super().__init__(**kwargs)

        # Get logger (must be done after super init)
        self.logger = logging.getLogger(__name__)

        # Gestion modèle - à implémenter par les classes enfants
        self.pipeline = pipeline

    def fit(self, x_train, y_train, **kwargs):
        '''Entrainement du modèle
           **kwargs permet la comptabilité avec les modèles keras
        Args:
            x_train (?): array-like or sparse matrix of shape = [n_samples, n_features]
            y_train (?): array-like, shape = [n_samples, n_features]
        Raises:
            RuntimeError: si on essaie d'entrainer un modèle déjà fit
            ValueError: si le type de modèle n'est pas classifier ou regressor
        '''
        if self.trained:
            self.logger.error("Il n'est pas prévu de pouvoir réentrainer un modèle de type pipeline sklearn")
            self.logger.error("Veuillez entrainer un nouveau modèle")
            raise RuntimeError("Impossible de réentrainer un modèle de type pipeline sklearn")

        # On check le format des entrants
        x_train, y_train = self._check_input_format(x_train, y_train, fit_function=True)

        if self.model_type == 'classifier':
            self._fit_classifier(x_train, y_train, **kwargs)
        elif self.model_type == 'regressor':
            self._fit_regressor(x_train, y_train, **kwargs)
        else:
            raise ValueError(f"Le type de modèle ({self.model_type}) doit être 'classifier' ou 'regressor'")

    def _fit_classifier(self, x_train, y_train, **kwargs):
        '''Entrainement d'un modèle de type classifier
           **kwargs permet la comptabilité avec les modèles keras
        Args:
            x_train (?): array-like or sparse matrix of shape = [n_samples, n_features]
            y_train (?): array-like, shape = [n_samples, n_features]
        '''
        # On check "juste" si pas multiclass multilabel (pas gérable par la plupart des pipelines SKLEARN)
        if self.multi_label:
            df_tmp = pd.DataFrame(y_train)
            for col in df_tmp:
                uniques = df_tmp[col].unique()
                if len(uniques) > 2:
                    self.logger.warning(' - /!\\/!\\/!\\/!\\/!\\/!\\/!\\/!\\/!\\/!\\ - ')
                    self.logger.warning("La plupart des pipeline sklearn ne supportent pas le multiclass-multilabel")
                    self.logger.warning(' - /!\\/!\\/!\\/!\\/!\\/!\\/!\\/!\\/!\\/!\\ - ')
                    # On "laisse" le programme planter tout seul
                    break

        # Fit pipeline
        self.pipeline.fit(x_train, y_train)

        # Set list classes
        if not self.multi_label:
            self.list_classes = list(self.pipeline.classes_)
        else:
            if hasattr(y_train, 'columns'):
                self.list_classes = list(y_train.columns)
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

    def _fit_regressor(self, x_train, y_train, **kwargs):
        '''Entrainement d'un modèle de type regressor
           **kwargs permet la comptabilité avec les modèles keras
        Args:
            x_train (?): array-like or sparse matrix of shape = [n_samples, n_features]
            y_train (?): array-like, shape = [n_samples, n_features]
        '''
        # Fit pipeline
        self.pipeline.fit(x_train, y_train)

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
        Raises:
            ValueError: si modèle pas classifier & return_proba = True
        Returns:
            (?): array of shape = [n_samples]
        '''
        # Manage errros
        if return_proba == True and self.model_type != 'classifier':
            raise ValueError(f"Les modèles de type {self.model_type} n'implémente ne gère pas les probabilités")

        # On check le format des entrants
        x_test, _ = self._check_input_format(x_test)

        #
        if not return_proba:
            return np.array(self.pipeline.predict(x_test))
        else:
            return self.predict_proba(x_test)

    @utils.trained_needed
    def predict_proba(self, x_test: pd.DataFrame, **kwargs):
        '''Prédictions probabilité sur test - Classifier only

        Args:
            x_test (pd.DataFrame): DataFrame sur laquelle faire les prédictions
        Raises:
            ValueError: si modèle pas classifier
        Returns:
            (?): array of shape = [n_samples]
        '''
        if self.model_type != 'classifier':
            raise ValueError(f"Les modèles de type {self.model_type} n'implémente pas la fonction predict_proba")

        # On check le format des entrants
        x_test, _ = self._check_input_format(x_test)

        #
        probas = np.array(self.pipeline.predict_proba(x_test))
        # Very specific fix: in some cases, with OvR, strategy, all estimators return 0, which generates a division per 0 when normalizing
        # Hence, we replace NaNs with 1 / nb_classes
        if not np.isnan(probas).any():
            probas = np.nan_to_num(probas, nan=1/len(self.list_classes))

        # Si utilisation de MultiOutputClassifier -> retourne proba de 0 et proba de 1 pour tous les éléments et toutes les classes
        # Pareil pour certains modèles de base
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

        json_data['librairie'] = 'scikit-learn'

        # Add each pipeline steps' conf
        if self.pipeline is not None:
            for step in self.pipeline.steps:
                name = step[0]
                confs = step[1].get_params()
                # Get rid of some non serializable conf
                for special_conf in ['dtype', 'base_estimator', 'estimator', 'estimator__base_estimator',
                                     'estimator__estimator', 'estimator__estimator__base_estimator']:
                    if special_conf in confs.keys():
                        confs[special_conf] = str(confs[special_conf])
                json_data[f'{name}_confs'] = confs

        # Save
        super().save(json_data=json_data)

        # Save model standalone if wanted & pipeline is not None & level_save > 'LOW'
        if self.pipeline is not None and self.level_save in ['MEDIUM', 'HIGH']:
            pkl_path = os.path.join(self.model_dir, f"{self.model_name}_standalone.pkl")
            # Sauvegarde model
            with open(pkl_path, 'wb') as f:
                pickle.dump(self.pipeline, f)

    def reload_from_standalone(self, **kwargs):
        '''Fonction pour recharger un modèle à partir de sa configuration et de sa pipeline
        - /!\\ Needs to be overrided /!\\ -
        '''
        raise NotImplementedError("'reload' needs to be overrided")


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.error("Ce script ne doit pas être exécuté, il s'agit d'un package.")