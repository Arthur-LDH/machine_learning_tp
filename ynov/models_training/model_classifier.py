#!/usr/bin/env python3

## Définition d'une classe parent pour les modèles
# Auteurs : Agence dataservices
# Date : 07/04/2021
#
# Classes :
# - ModelClassifierMixin -> Classe parent classifier


import os
import re
import json
import logging
import numpy as np
import pandas as pd
import seaborn as sns
from typing import List
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, auc, confusion_matrix, f1_score,
                             multilabel_confusion_matrix, precision_score,
                             recall_score, roc_curve)
from ynov import utils
from ynov.models_training import utils_models
from ynov.monitoring.model_logger import ModelLogger

sns.set(style="darkgrid")


class ModelClassifierMixin:
    '''Classe parent (Mixin) pour les modèles de type classifier'''

    def __init__(self, level_save: str = 'HIGH', multi_label: bool = False, **kwargs):
        '''Initialisation de la classe parent - Classifier

        Kwargs:
            level_save (str): Niveau de sauvegarde
                LOW: statistiques + configurations + logger keras - /!\\ modèle non réutilisable /!\\ -
                MEDIUM: LOW + hdf5 + pkl + plots
                HIGH: MEDIUM + predictions
            multi_label (bool): si la classification doit être multi label
        Raises:
            ValueError : si l'objet level_save n'est pas une option valable (['LOW', 'MEDIUM', 'HIGH'])
        '''
        super().__init__(level_save=level_save, **kwargs)  # forwards level_save & all unused arguments

        if level_save not in ['LOW', 'MEDIUM', 'HIGH']:
            raise ValueError(f"L'objet level_save ({level_save}) n'est pas une option valide (['LOW', 'MEDIUM', 'HIGH'])")

        # Get logger
        self.logger = logging.getLogger(__name__)

        # Type de model
        self.model_type = 'classifier'

        # Multi label ?
        self.multi_label = multi_label

        # Liste classes à traiter (set on fit)
        self.list_classes = None
        self.dict_classes = None

        # Other options
        self.level_save = level_save

    @utils.trained_needed
    def predict_with_proba(self, x_test: pd.DataFrame):
        '''Prédictions sur test avec probabilités

        Args:
            x_test (pd.DataFrame): DataFrame sur laquelle faire les prédictions
        Returns:
            (?): array of shape = [n_samples, n_classes]
            (?): array of shape = [n_samples, n_classes]
        '''
        # Process
        predicted_proba = self.predict(x_test, return_proba=True)
        predicted_class = self.get_classes_from_proba(predicted_proba)
        return predicted_class, predicted_proba

    @utils.trained_needed
    def get_predict_position(self, x_test: pd.DataFrame, y_true):
        '''Fonction pour obtenir l'odre de prédictions de y_true
        Les positions commencent à 1 (pas de 0)

        Args:
            x_test (pd.DataFrame): DataFrame sur laquelle faire les prédictions
            y_true (?): array-like, shape = [n_samples, n_features]
        Raises:
            ValueError: non disponible en mode multi label
        Returns:
            (?): array of shape = [n_samples]
        '''
        if self.multi_label:
            raise ValueError("La fonction 'get_predict_position' n'est pas disponible en mode multi-label")
        # Process
        # Cast en pd.Series
        y_true = pd.Series(y_true)
        # Get predicted proba
        predicted_proba = self.predict(x_test, return_proba=True)
        # Get position
        order = predicted_proba.argsort()
        ranks = len(self.dict_classes.values()) - order.argsort()
        df_probas = pd.DataFrame(ranks, columns=self.dict_classes.values())
        predict_positions = np.array([df_probas.loc[i, cl] if cl in df_probas.columns else -1 for i, cl in enumerate(y_true)])
        return predict_positions

    def get_classes_from_proba(self, predicted_proba):
        '''Function pour récupérer les classes à partir de probabilités

        Args:
            predicted_proba (?): array-like or sparse matrix of shape = [n_samples] -> probabilités
        Returns:
            (?): array of shape = [n_samples, n_classes] -> classes
        '''
        if not self.multi_label:
            predicted_class = np.vectorize(lambda x: self.dict_classes[x])(predicted_proba.argmax(axis=-1))
        else:
            # Si multi label, retourne list de 0 et de 1
            predicted_class = np.vectorize(lambda x: 1 if x >= 0.5 else 0)(predicted_proba)
        return predicted_class

    def get_top_n_from_proba(self, predicted_proba, n: int = 5):
        '''Function pour récupérer les TOP N prédictions depuis des probas

        Args:
            predicted_proba (?): array-like or sparse matrix of shape = [n_samples] -> probabilités
        kwargs:
            n (int): nombre de classes à retourner
        Raises:
            ValueError: si le nomre de classes à retourner est plus grand que le nombre de classes du modèle
        Returns:
            (?): array of shape = [n_samples, n] -> top n predicted class
        '''
        # TODO: faire en sorte que cette fonction soit dispo pour du multi-label
        if self.multi_label:
            raise ValueError("La fonction 'get_top_n_from_proba' n'est pas disponible en mode multi-label")
        if n > len(self.list_classes):
            raise ValueError("Plus de classes demandées que de classes dans le modèle")
        # Process
        idx = predicted_proba.argsort()[:, -n:][:, ::-1]
        top_n_proba = list(np.take_along_axis(predicted_proba, idx, axis=1))
        top_n = list(np.vectorize(lambda x: self.dict_classes[x])(idx))
        return top_n, top_n_proba

    def inverse_transform(self, y):
        '''Fonction pour obtenir une liste de classes à partir de prédicitons

        Args:
            y (?): array-like, shape = [n_samples, n_features], arrays of 0s and 1s
                   OR (new 12/11/2020) 1D array (only one pred)
        Raises:
            ValueError: Si la taille de y ne correspond pas au nombre de classes du model
        Returns:
            (?): array of shape = [n_samples, ?]
        '''
        # If multi-label, get classes in tuple
        if self.multi_label:
            if y.shape[-1] != len(self.list_classes): # On prend "-1" pour gérer les cas où y serait 1D (i.e. juste une seule pred)
                raise ValueError(f"La taille de y ({y.shape[-1]}) ne correspond pas" +
                                 f" au nombre de classes ({len(self.list_classes)}) du model")
            # Manage 1D array (only one pred)
            if len(y.shape) == 1:
                return tuple(np.array(self.list_classes).compress(y))
            # Several preds
            else:
                return [tuple(np.array(self.list_classes).compress(indicators)) for indicators in y]
        # If mono-label, just cast in list if y is np array
        else:
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
            # Ajout colonne matched
            df.loc[:, 'matched'] = df[['y_true', 'y_pred']].apply(lambda x: 1 if x.y_true == x.y_pred else 0, axis=1)
            # Ajout colonnes supplémentaires
            if series_to_add is not None:
                for ser in series_to_add:
                    df[ser.name] = ser.reset_index(drop=True).reindex(index=df.index)  # Reindex comme il faut

            # Sauvegarde des prédiction
            file_path = os.path.join(self.model_dir, f"predictions{'_' + type_data if len(type_data) > 0 else ''}.csv")
            df.sort_values('matched', ascending=True).to_csv(file_path, sep=',', index=None, encoding='utf-8')

        # Récupération f1 score / acc_tot / trues / falses / precision / recall / support globaux
        if self.multi_label:
            f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            trues = np.sum(np.all(np.equal(y_true, y_pred), axis=1))
            falses = len(y_true) - trues
            acc_tot = trues / len(y_true)
            precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            support = list(pd.DataFrame(y_true).sum().values)
            support = [_ / sum(support) for _ in support] + [1.0]
        else:
            # On fait quand même du 'weighted' si mono-label, car possibilité multiclasses !
            f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            trues = np.sum(y_true == y_pred)
            falses = np.sum(y_true != y_pred)
            acc_tot = accuracy_score(y_true, y_pred)
            precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            labels_tmp, counts_tmp = np.unique(y_true, return_counts=True)
            support = [0] * len(self.list_classes) + [1.0]
            for i, cl in enumerate(self.list_classes):
                if cl in labels_tmp:
                    idx_tmp = list(labels_tmp).index(cl)
                    support[i] = counts_tmp[idx_tmp] / y_pred.shape[0]

        # Statistiques globale
        self.logger.info('-- * * * * * * * * * * * * * * --')
        self.logger.info(f"Statistiques f1-score{' ' + type_data if len(type_data) > 0 else ''}")
        self.logger.info('--------------------------------')
        self.logger.info(f"Accuracy totale : {round(acc_tot * 100, 2)}% \t Trues: {trues} \t Falses: {falses}")
        self.logger.info(f"F1-score (weighted) : {round(f1_weighted, 5)}")
        self.logger.info(f"Precision (weighted) : {round(precision_weighted, 5)}")
        self.logger.info(f"Recall (weighted) : {round(recall_weighted, 5)}")
        self.logger.info('--------------------------------')

        # Fichier metrics
        df_stats = pd.DataFrame(columns=['Label', 'F1-Score', 'Accuracy',
                                         'Precision', 'Recall', 'Trues', 'Falses',
                                         'True positive', 'True negative',
                                         'False positive', 'False negative',
                                         'Condition positive', 'Condition negative',
                                         'Predicted positive', 'Predicted negative'])

        # Ajout metrics en fonction multi/mono label & gestion conf. matrices
        labels = self.list_classes
        log_stats = len(labels) < 50
        if self.multi_label:
            # Détails par catégories
            mcm = multilabel_confusion_matrix(y_true, y_pred)
            for i, label in enumerate(labels):
                c_mat = mcm[i]
                df_stats = df_stats.append(self._update_info_from_c_mat(c_mat, label, log_info=log_stats), ignore_index=True)
                # Plot individual confusion matrix if level_save > LOW
                if self.level_save in ['MEDIUM', 'HIGH']:
                    none_class = 'not_' + label
                    tmp_label = re.sub(r',|:|\s', '_', label)
                    self._plot_confusion_matrix(c_mat, [none_class, label], type_data=f"{tmp_label}_{type_data}",
                                                normalized=False, subdir=type_data)
                    self._plot_confusion_matrix(c_mat, [none_class, label], type_data=f"{tmp_label}_{type_data}",
                                                normalized=True, subdir=type_data)
        else:
            # Plot confusion matrices if level_save > LOW
            if self.level_save in ['MEDIUM', 'HIGH']:
                if len(labels) > 50:
                    self.logger.warning(
                        f"Attention, il y a {len(labels)} catégories à plot sur la matrice de confusion.\n"
                        + f"Fortes chances de ralentissements/bugs d'affichages/crashs ...\n"
                        + f"On SKIP les plots"
                    )
                else:
                    # Global stats
                    c_mat = confusion_matrix(y_true, y_pred, labels=labels)
                    self._plot_confusion_matrix(c_mat, labels, type_data=type_data, normalized=False)
                    self._plot_confusion_matrix(c_mat, labels, type_data=type_data, normalized=True)

            # Get stats per class
            for label in labels:
                none_class = 'None' if label != 'None' else 'others'  # On s'assure que la class n'est pas déjà "None"
                y_true_tmp = [label if _ == label else none_class for _ in y_true]
                y_pred_tmp = [label if _ == label else none_class for _ in y_pred]
                c_mat_tmp = confusion_matrix(y_true_tmp, y_pred_tmp, labels=[none_class, label])
                df_stats = df_stats.append(self._update_info_from_c_mat(c_mat_tmp, label, log_info=log_stats), ignore_index=True)

        # Ajout statistiques globales
        global_stats = {
            'Label': 'All',
            'F1-Score': f1_weighted,
            'Accuracy': acc_tot,
            'Precision': precision_weighted,
            'Recall': recall_weighted,
            'Trues': trues,
            'Falses': falses,
            'True positive': None,
            'True negative': None,
            'False positive': None,
            'False negative': None,
            'Condition positive': None,
            'Condition negative': None,
            'Predicted positive': None,
            'Predicted negative': None,
        }
        df_stats = df_stats.append(global_stats, ignore_index=True)

        # Ajout support
        df_stats['Support'] = support

        # Sauvegarde du csv
        file_path = os.path.join(self.model_dir, f"f1{'_' + type_data if len(type_data) > 0 else ''}@{f1_weighted}.csv")
        df_stats.to_csv(file_path, sep=',', index=False, encoding='utf-8')

        # Sauvegarde de l'accuracy
        acc_path = os.path.join(self.model_dir, f"acc{'_' + type_data if len(type_data) > 0 else ''}@{round(acc_tot, 5)}")
        with open(acc_path, 'w') as f:
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

        return df_stats

    def get_metrics_simple_monolabel(self, y_true, y_pred):
        '''Fonction pour obtenir des métriques sur des prédictions monolabel
        Permet de faire comme la fonction get_and_save_metrics mais sans tout ce qu'il y a autour (sauvegarde, etc.)

        Args:
            y_true (?): array-like, shape = [n_samples, n_features]
            y_pred (?): array-like, shape = [n_samples, n_features]
        Raises:
            ValueError: si pas en mode monolabel
        Returns:
            pd.DataFrame: la df qui contient les statistiques
        '''
        if self.multi_label:
            raise ValueError("La fonction get_metrics_simple_monolabel ne fonctionne que pour les cas monolabels")

        # Cast to np.array
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Récupération f1 score / acc_tot / trues / falses / precision / recall / support globaux
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        trues = np.sum(y_true == y_pred)
        falses = np.sum(y_true != y_pred)
        acc_tot = accuracy_score(y_true, y_pred)
        precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        labels_tmp, counts_tmp = np.unique(y_true, return_counts=True)
        support = [0] * len(self.list_classes) + [1.0]
        for i, cl in enumerate(self.list_classes):
            if cl in labels_tmp:
                idx_tmp = list(labels_tmp).index(cl)
                support[i] = counts_tmp[idx_tmp] / y_pred.shape[0]

        # DataFrame metrics
        df_stats = pd.DataFrame(columns=['Label', 'F1-Score', 'Accuracy',
                                         'Precision', 'Recall', 'Trues', 'Falses',
                                         'True positive', 'True negative',
                                         'False positive', 'False negative',
                                         'Condition positive', 'Condition negative',
                                         'Predicted positive', 'Predicted negative'])

        # Get stats per class
        labels = self.list_classes
        for label in labels:
            none_class = 'None' if label != 'None' else 'others'  # On s'assure que la class n'est pas déjà "None"
            y_true_tmp = [label if _ == label else none_class for _ in y_true]
            y_pred_tmp = [label if _ == label else none_class for _ in y_pred]
            c_mat_tmp = confusion_matrix(y_true_tmp, y_pred_tmp, labels=[none_class, label])
            df_stats = df_stats.append(self._update_info_from_c_mat(c_mat_tmp, label, log_info=False), ignore_index=True)

        # Ajout statistiques globales
        global_stats = {
            'Label': 'All',
            'F1-Score': f1_weighted,
            'Accuracy': acc_tot,
            'Precision': precision_weighted,
            'Recall': recall_weighted,
            'Trues': trues,
            'Falses': falses,
            'True positive': None,
            'True negative': None,
            'False positive': None,
            'False negative': None,
            'Condition positive': None,
            'Condition negative': None,
            'Predicted positive': None,
            'Predicted negative': None,
        }
        df_stats = df_stats.append(global_stats, ignore_index=True)

        # Ajout support
        df_stats['Support'] = support

        # Return dataframe
        return df_stats

    def get_metrics_simple_multilabel(self, y_true, y_pred):
        '''Fonction pour obtenir des métriques sur des prédictions multilabel
        Permet de faire comme la fonction get_and_save_metrics mais sans tout ce qu'il y a autour (sauvegarde, etc.)

        Args:
            y_true (?): array-like, shape = [n_samples, n_features]
            y_pred (?): array-like, shape = [n_samples, n_features]
        Raises:
            ValueError: si pas en mode multilabel
        Returns:
            pd.DataFrame: la df qui contient les statistiques
        '''
        if not self.multi_label:
            raise ValueError("La fonction get_metrics_simple_multilabel ne fonctionne que pour les cas multilabels")

        # Cast to np.array
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Récupération f1 score / acc_tot / trues / falses / precision / recall / support globaux
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        trues = np.sum(np.all(np.equal(y_true, y_pred), axis=1))
        falses = len(y_true) - trues
        acc_tot = trues / len(y_true)
        precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        support = list(pd.DataFrame(y_true).sum().values)
        support = [_ / sum(support) for _ in support] + [1.0]

        # DataFrame metrics
        df_stats = pd.DataFrame(columns=['Label', 'F1-Score', 'Accuracy',
                                         'Precision', 'Recall', 'Trues', 'Falses',
                                         'True positive', 'True negative',
                                         'False positive', 'False negative',
                                         'Condition positive', 'Condition negative',
                                         'Predicted positive', 'Predicted negative'])

        # Ajout metrics
        labels = self.list_classes
        # Détails par catégories
        mcm = multilabel_confusion_matrix(y_true, y_pred)
        for i, label in enumerate(labels):
            c_mat = mcm[i]
            df_stats = df_stats.append(self._update_info_from_c_mat(c_mat, label, log_info=False), ignore_index=True)

        # Ajout statistiques globales
        global_stats = {
            'Label': 'All',
            'F1-Score': f1_weighted,
            'Accuracy': acc_tot,
            'Precision': precision_weighted,
            'Recall': recall_weighted,
            'Trues': trues,
            'Falses': falses,
            'True positive': None,
            'True negative': None,
            'False positive': None,
            'False negative': None,
            'Condition positive': None,
            'Condition negative': None,
            'Predicted positive': None,
            'Predicted negative': None,
        }
        df_stats = df_stats.append(global_stats, ignore_index=True)

        # Ajout support
        df_stats['Support'] = support

        # Return dataframe
        return df_stats

    def _update_info_from_c_mat(self, c_mat: np.ndarray, label: str, log_info: bool = True):
        '''Function to update a dataframe for the funcion get_and_save_metrics, given a confusion matrix

        Args:
            c_mat (np.array): matrice de confusion
            label (str): label à utiliser
        Kwargs:
            log_info (bool): si les stats doivent être loggées
        Returns:
            dict: dictionnaire avec les infos pour màj de la dataframe
        '''

        # Extract all needed info from c_mat
        true_negative = c_mat[0][0]
        true_positive = c_mat[1][1]
        false_negative = c_mat[1][0]
        false_positive = c_mat[0][1]
        condition_positive = false_negative + true_positive
        condition_negative = false_positive + true_negative
        predicted_positive = false_positive + true_positive
        predicted_negative = false_negative + true_negative
        trues_cat = true_negative + true_positive
        falses_cat = false_negative + false_positive
        accuracy = (true_negative + true_positive) / (true_negative + true_positive + false_negative + false_positive)
        precision = 0 if predicted_positive == 0 else true_positive / predicted_positive
        recall = 0 if condition_positive == 0 else true_positive / condition_positive
        f1 = 0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

        # Display some info
        if log_info:
            self.logger.info(
                f"F1-score: {round(f1, 5)}  \t Precision: {round(100 * precision, 2)}% \t"
                f"Recall: {round(100 * recall, 2)}% \t Trues: {trues_cat} \t Falses: {falses_cat} \t\t --- {label} "
            )

        # Return result
        return {
            'Label': f'{label}',
            'F1-Score': f1,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'Trues': trues_cat,
            'Falses': falses_cat,
            'True positive': true_positive,
            'True negative': true_negative,
            'False positive': false_positive,
            'False negative': false_negative,
            'Condition positive': condition_positive,
            'Condition negative': condition_negative,
            'Predicted positive': predicted_positive,
            'Predicted negative': predicted_negative,
        }

    def _plot_confusion_matrix(self, c_mat: np.ndarray, labels: list, type_data: str = '',
                               normalized: bool = False, subdir: str = None):
        '''Function to plot a confusion matrix

        Args:
            c_mat (np.ndarray): matrice de confusion
            labels (list): labels à plot
        Kwargs:
            type_data (str): type du dataset (validation, test, ...)
            normalized (bool): si la matrice de confusion doit être normalisée
            subdir (str): sub directory for plot
        '''

        # Get title
        if normalized:
            title = f"Normalized confusion matrix{' - ' + type_data if len(type_data) > 0 else ''}"
        else:
            title = f"Confusion matrix, without normalization{' - ' + type_data if len(type_data) > 0 else ''}"

        # Init. plot
        width = round(10 + 0.5 * len(c_mat))
        height = round(4 / 5 * width)
        fig, ax = plt.subplots(figsize=(width, height))

        # Plot
        if normalized:
            c_mat = c_mat.astype('float') / c_mat.sum(axis=1)[:, np.newaxis]
            sns.heatmap(c_mat, annot=True, fmt=".2f", cmap=plt.cm.Blues, ax=ax)
        else:
            sns.heatmap(c_mat, annot=True, fmt="d", cmap=plt.cm.Blues, ax=ax)

        # labels, title and ticks
        ax.set_xlabel('Predicted classes', fontsize=height * 2)
        ax.set_ylabel('Real classes', fontsize=height * 2)
        ax.set_title(title, fontsize=width * 2)
        ax.xaxis.set_ticklabels(labels)
        ax.yaxis.set_ticklabels(labels)
        plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
        plt.setp(ax.get_yticklabels(), rotation=30, horizontalalignment='right')
        plt.tight_layout()

        # Save
        plots_path = os.path.join(self.model_dir, 'plots')
        if subdir is not None:  # Ajout subdir
            plots_path = os.path.join(plots_path, subdir)
        file_name = f"{type_data + '_' if len(type_data) > 0 else ''}confusion_matrix{'_normalized' if normalized else ''}.png"
        if not os.path.exists(plots_path):
            os.makedirs(plots_path)
        plt.savefig(os.path.join(plots_path, file_name))

        # Close figures
        plt.close('all')

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

        json_data['list_classes'] = self.list_classes
        json_data['dict_classes'] = self.dict_classes
        json_data['multi_label'] = self.multi_label

        # Save
        super().save(json_data=json_data)


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.error("Ce script ne doit pas être exécuté, il s'agit d'un package.")