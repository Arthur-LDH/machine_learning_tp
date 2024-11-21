#!/usr/bin/env python3

## Définition d'une classe parent pour les modèles
# Auteurs : Agence dataservices
# Date : 07/04/2021
#
# Classes :
# - ModelRegressorMixin -> Classe parent regressor


import os
import re
import json
import logging
import numpy as np
import pandas as pd
import seaborn as sns
from typing import List
import matplotlib.pyplot as plt
from sklearn.metrics import (explained_variance_score, mean_absolute_error, mean_squared_error, r2_score)
from sklearn.linear_model import LinearRegression
from yellowbrick.regressor import ResidualsPlot, PredictionError
from ynov import utils
from ynov.models_training import utils_models
from ynov.monitoring.model_logger import ModelLogger

sns.set(style="darkgrid")


class ModelRegressorMixin:
    '''Classe parent (Mixin) pour les modèles de type regressor'''

    def __init__(self, level_save: str = 'HIGH', **kwargs):
        '''Initialisation de la classe parent - Regressor

        Kwargs:
            level_save (str): Niveau de sauvegarde
                LOW: statistiques + configurations + logger keras - /!\\ modèle non réutilisable /!\\ -
                MEDIUM: LOW + hdf5 + pkl + plots
                HIGH: MEDIUM + predictions
        Raises:
            ValueError : si l'objet level_save n'est pas une option valable (['LOW', 'MEDIUM', 'HIGH'])
        '''
        super().__init__(level_save=level_save, **kwargs)  # forwards level_save & all unused arguments

        if level_save not in ['LOW', 'MEDIUM', 'HIGH']:
            raise ValueError(f"L'objet level_save ({level_save}) n'est pas une option valide (['LOW', 'MEDIUM', 'HIGH'])")

        # Get logger
        self.logger = logging.getLogger(__name__)

        # Type de model
        self.model_type = 'regressor'

        # TODO: ajouter multi output !

        # Other options
        self.level_save = level_save

    def inverse_transform(self, y):
        '''Fonction identité - Permet de gérer compatibilité avec classifiers

        Args:
            y (?): array-like, shape = [n_samples, 1]
        Returns:
            (?): list of shape = [n_samples, 1]
        '''
        return list(y) if type(y) == np.ndarray else y

    def get_and_save_metrics(self, y_true, y_pred, df_x: pd.DataFrame = None, series_to_add: List[pd.Series] = None, type_data: str = '', model_logger=None):
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
        if series_to_add is not None:
            if sum([1 if type(_) == pd.Series else 0 for _ in series_to_add]) != len(series_to_add):
                raise TypeError("L'objet series_to_add doit être composé de pd.Series uniquement")

        # Cast to np.array
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Sauvegarde d'un fichier de prédiction si souhaité
        if self.level_save == 'HIGH':
            # Inverse transform
            y_true_df = list(self.inverse_transform(y_true))
            y_pred_df = list(self.inverse_transform(y_pred))

            # Concat dans une dataframe
            if df_x is not None:
                df = df_x.copy()
                df['y_true'] = y_true_df
                df['y_pred'] = y_pred_df
            else:
                df = pd.DataFrame({'y_true': y_true_df, 'y_pred': y_pred_df})
            # Ajout colonne abs_err
            df.loc[:, 'abs_err'] = df[['y_true', 'y_pred']].apply(lambda x: x.y_true - x.y_pred, axis=1)
            # Ajout colonne rel_err
            df.loc[:, 'rel_err'] = df[['y_true', 'y_pred']].apply(lambda x: (x.y_true - x.y_pred) / abs(x.y_true), axis=1)
            # Ajout colonnes supplémentaires
            if series_to_add is not None:
                for ser in series_to_add:
                    df[ser.name] = ser.reset_index(drop=True).reindex(index=df.index)  # Reindex comme il faut

            # Sauvegarde des prédiction
            file_path = os.path.join(self.model_dir, f"predictions{'_' + type_data if len(type_data) > 0 else ''}.csv")
            df.sort_values('abs_err', ascending=True).to_csv(file_path, sep=',', index=None, encoding='utf-8')

        # Récupération métriques globales ;
        metric_mae = mean_absolute_error(y_true, y_pred)
        metric_mse = mean_squared_error(y_true, y_pred)
        metric_rmse = mean_squared_error(y_true, y_pred, squared=False)
        metric_explained_variance_score = explained_variance_score(y_true, y_pred)
        metric_r2 = r2_score(y_true, y_pred)

        # Statistiques globale
        self.logger.info('-- * * * * * * * * * * * * * * --')
        self.logger.info(f"Statistiques{' ' + type_data if len(type_data) > 0 else ''}")
        self.logger.info('--------------------------------')
        self.logger.info(f"MAE : {round(metric_mae, 5)}")
        self.logger.info(f"MSE : {round(metric_mse, 5)}")
        self.logger.info(f"RMSE : {round(metric_rmse, 5)}")
        self.logger.info(f"Explained variance : {round(metric_explained_variance_score, 5)}")
        self.logger.info(f"R² (coefficient of determination) : {round(metric_r2, 5)}")
        self.logger.info('--------------------------------')

        # Fichier metrics
        df_stats = pd.DataFrame(columns=['Label', 'MAE', 'MSE',
                                         'RMSE', 'Explained variance',
                                         'Coefficient of determination'])

        # TODO : plus tard, ajouter multi output et donc stats par output

        # Ajout statistiques globales
        global_stats = {
            'Label': 'All',
            'MAE': metric_mae,
            'MSE': metric_mse,
            'RMSE': metric_rmse,
            'Explained variance': metric_explained_variance_score,
            'Coefficient of determination': metric_r2,
        }
        df_stats = df_stats.append(global_stats, ignore_index=True)

        # Sauvegarde du csv
        file_path = os.path.join(self.model_dir, f"mae{'_' + type_data if len(type_data) > 0 else ''}@{metric_mae}.csv")
        df_stats.to_csv(file_path, sep=',', index=False, encoding='utf-8')

        # Sauvegarde de quelques métriques
        mae_path = os.path.join(self.model_dir, f"mae{'_' + type_data if len(type_data) > 0 else ''}@{round(metric_mae, 5)}")
        with open(mae_path, 'w') as f:
            pass
        mse_path = os.path.join(self.model_dir, f"mse{'_' + type_data if len(type_data) > 0 else ''}@{round(metric_mse, 5)}")
        with open(mse_path, 'w') as f:
            pass
        rmse_path = os.path.join(self.model_dir, f"rmse{'_' + type_data if len(type_data) > 0 else ''}@{round(metric_rmse, 5)}")
        with open(rmse_path, 'w') as f:
            pass
        explained_variance_path = os.path.join(self.model_dir, f"explained_variance{'_' + type_data if len(type_data) > 0 else ''}@{round(metric_explained_variance_score, 5)}")
        with open(explained_variance_path, 'w') as f:
            pass
        r2_path = os.path.join(self.model_dir, f"r2{'_' + type_data if len(type_data) > 0 else ''}@{round(metric_r2, 5)}")
        with open(r2_path, 'w') as f:
            pass

        # Upload des metriques sur mlflow (ou autre plateforme)
        if model_logger is not None:
            # TODO : à mettre dans une fonction
            # Prepare params.
            label_col = 'Label'
            metrics_columns = [col for col in df_stats.columns if col != label_col]

            # Log labels
            labels = df_stats[label_col].values
            for i, label in enumerate(labels):
                model_logger.log_param(f'Label {i}', label)
            # Log metrics
            ml_flow_metrics = {}
            for i, row in df_stats.iterrows():
                for c in metrics_columns:
                    metric_key = f"{row[label_col]} --- {c}"
                    # On check que ML FLOW accepte la key, sinon on remplace
                    if not model_logger.valid_name(metric_key):
                        metric_key = f"Label {i} --- {c}"
                    ml_flow_metrics[metric_key] = row[c]
            # Log metrics
            model_logger.log_metrics(ml_flow_metrics)

        # Plots
        if self.level_save in ['MEDIUM', 'HIGH']:
            # TODO: mettre une condition sur nombre max de points ?
            is_train = True if type_data == 'train' else False
            if is_train:
                self.plot_prediction_errors(y_true_train=y_true, y_pred_train=y_pred,
                                            y_true_test=None, y_pred_test=None,
                                            type_data=type_data)
                self.plot_residuals(y_true_train=y_true, y_pred_train=y_pred,
                                    y_true_test=None, y_pred_test=None,
                                    type_data=type_data)
            else:
                self.plot_prediction_errors(y_true_train=None, y_pred_train=None,
                                            y_true_test=y_true, y_pred_test=y_pred,
                                            type_data=type_data)
                self.plot_residuals(y_true_train=None, y_pred_train=None,
                                    y_true_test=y_true, y_pred_test=y_pred,
                                    type_data=type_data)

        # Return metrics
        return df_stats

    def get_metrics_simple(self, y_true, y_pred):
        '''Fonction pour obtenir des métriques sur des prédictions (pour l'instant mono-output)
        Permet de faire comme la fonction get_and_save_metrics mais sans tout ce qu'il y a autour (sauvegarde, etc.)

        Args:
            y_true (?): array-like, shape = [n_samples, n_features]
            y_pred (?): array-like, shape = [n_samples, n_features]
        Raises:
            ValueError: si pas en mode monolabel
        Returns:
            pd.DataFrame: la df qui contient les statistiques
        '''
        # Cast to np.array
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Récupération métriques globales ;
        metric_mae = mean_absolute_error(y_true, y_pred)
        metric_mse = mean_squared_error(y_true, y_pred)
        metric_rmse = mean_squared_error(y_true, y_pred, squared=False)
        metric_explained_variance_score = explained_variance_score(y_true, y_pred)
        metric_r2 = r2_score(y_true, y_pred)

        # Fichier metrics
        df_stats = pd.DataFrame(columns=['Label', 'MAE', 'MSE',
                                         'RMSE', 'Explained variance',
                                         'Coefficient of determination'])

        # TODO : plus tard, ajouter multi output et donc stats par output

        # Ajout statistiques globales
        global_stats = {
            'Label': 'All',
            'MAE': metric_mae,
            'MSE': metric_mse,
            'RMSE': metric_rmse,
            'Explained variance': metric_explained_variance_score,
            'Coefficient of determination': metric_r2,
        }
        df_stats = df_stats.append(global_stats, ignore_index=True)

        # Return dataframe
        return df_stats

    def plot_prediction_errors(self, y_true_train: np.ndarray = None, y_pred_train: np.ndarray = None,
                               y_true_test: np.ndarray = None, y_pred_test: np.ndarray = None,
                               type_data: str = ''):
        '''Fonction pour plot les erreurs de prédictions

        On utilise yellowbrick pour les plots + un trick pour être model agnostic

        Kwargs:
            y_true_train (?): array-like, shape = [n_samples, n_features]
            y_pred_train (?): array-like, shape = [n_samples, n_features]
            y_true_test (?): array-like, shape = [n_samples, n_features]
            y_pred_test (?): array-like, shape = [n_samples, n_features]
            type_data (str): type du dataset (validation, test, ...)
        Raises:
            ValueError: si un "true" est renseigné, mais pas son "pred" (ou l'inverse)
        '''
        if (y_true_train is not None and y_pred_train is None) or (y_true_train is None and y_pred_train is not None):
            raise ValueError('"true" et "pred" doivent être renseignés ensemble, ou pas du tout - train')
        if (y_true_test is not None and y_pred_test is None) or (y_true_test is None and y_pred_test is not None):
            raise ValueError('"true" et "pred" doivent être renseignés ensemble, ou pas du tout - test')

        # Get figure & ax
        fig, ax = plt.subplots(figsize=(12, 10))

        # Set visualizer
        visualizer = PredictionError(LinearRegression(), ax=ax, bestfit=False, is_fitted=True) # Trick model non utilisé
        visualizer.name = self.model_name

         # PredictionError ne supporte pas train et test en même temps :'(

        # Train
        if y_true_train is not None:
            visualizer.score_ = r2_score(y_true_train, y_pred_train)
            visualizer.draw(y_true_train, y_pred_train)

        # Test
        if y_true_test is not None:
            visualizer.score_ = r2_score(y_true_test, y_pred_test)
            visualizer.draw(y_true_test, y_pred_test)

        # Save
        plots_path = os.path.join(self.model_dir, 'plots')
        if not os.path.exists(plots_path):
            os.makedirs(plots_path)
        file_name = f"{type_data + '_' if len(type_data) > 0 else ''}errors.png"
        visualizer.show(outpath=os.path.join(plots_path, file_name))

        # Close figures
        plt.close('all')

    def plot_residuals(self, y_true_train: np.ndarray = None, y_pred_train: np.ndarray = None,
                       y_true_test: np.ndarray = None, y_pred_test: np.ndarray = None,
                       type_data: str = ''):
        '''Fonction pour plot les "residuals" à partir des résultats de prédictions

        On utilise yellowbrick pour les plots + un trick pour être model agnostic

        Kwargs:
            y_true_train (?): array-like, shape = [n_samples, n_features]
            y_pred_train (?): array-like, shape = [n_samples, n_features]
            y_true_test (?): array-like, shape = [n_samples, n_features]
            y_pred_test (?): array-like, shape = [n_samples, n_features]
            type_data (str): type du dataset (validation, test, ...)
        Raises:
            ValueError: si un "true" est renseigné, mais pas son "pred" (ou l'inverse)
        '''
        if (y_true_train is not None and y_pred_train is None) or (y_true_train is None and y_pred_train is not None):
            raise ValueError('"true" et "pred" doivent être renseignés ensemble, ou pas du tout - train')
        if (y_true_test is not None and y_pred_test is None) or (y_true_test is None and y_pred_test is not None):
            raise ValueError('"true" et "pred" doivent être renseignés ensemble, ou pas du tout - test')

        # Get figure & ax
        fig, ax = plt.subplots(figsize=(12, 10))

        # Set visualizer
        visualizer = ResidualsPlot(LinearRegression(), ax=ax, is_fitted=True) # Trick model non utilisé
        visualizer.name = self.model_name

        # Train
        if y_true_train is not None:
            visualizer.train_score_ = r2_score(y_true_train, y_pred_train)
            residuals = y_pred_train - y_true_train
            visualizer.draw(y_pred_train, residuals, train=True)

        # Test
        if y_true_test is not None:
            visualizer.test_score_ = r2_score(y_true_test, y_pred_test)
            residuals = y_pred_test - y_true_test
            visualizer.draw(y_pred_test, residuals, train=False)

        # Save
        plots_path = os.path.join(self.model_dir, 'plots')
        if not os.path.exists(plots_path):
            os.makedirs(plots_path)
        file_name = f"{type_data + '_' if len(type_data) > 0 else ''}residuals.png"
        visualizer.show(outpath=os.path.join(plots_path, file_name))

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

        # Pour l'instant rien à rajouter
        # TODO: multi output plus tard

        # Save
        super().save(json_data=json_data)


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.error("Ce script ne doit pas être exécuté, il s'agit d'un package.")