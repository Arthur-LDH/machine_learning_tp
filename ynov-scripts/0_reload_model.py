#!/usr/bin/env python3

## Rechargement d'un modèle train "ailleurs"
# Auteurs : Agence dataservices
# Date : 07/07/2020
#
# Ex: poetry run python 0_reload_model.py -m best_model -c configurations.json -s model_preprocess_pipeline_svm_standalone.pkl

import os
import json
import ntpath
import logging
import argparse
import pandas as pd

from ynov import utils
from ynov.models_training import utils_deep_keras
from ynov.preprocessing import preprocess
from ynov.models_training import utils_models
from ynov.models_training.classifiers import (model_rf_classifier, model_dense_classifier,
                                                          model_ridge_classifier, model_logistic_regression_classifier,
                                                          model_sgd_classifier, model_svm_classifier, model_knn_classifier,
                                                          model_gbt_classifier, model_lgbm_classifier, model_xgboost_classifier)
from ynov.models_training.regressors import (model_rf_regressor, model_dense_regressor,
                                                         model_elasticnet_regressor, model_bayesian_ridge_regressor,
                                                         model_kernel_ridge_regressor, model_svr_regressor,
                                                         model_sgd_regressor, model_knn_regressor, model_pls_regressor,
                                                         model_gbt_regressor, model_xgboost_regressor, model_lgbm_regressor)

# Get logger
logger = logging.getLogger('ynov.0_reload_model')


def main(model_dir: str, config_file: str, standalone_file: str,
         weights_file: str, preprocess_pipeline_file: str):
    '''Fonction principale pour recharger un modèle train depuis un autre pc / une autre librairie

    Args:
        model_dir (str): Nom du model à charger
        config_file (str): Nom du fichier de configuration
    Kwargs:
        standalone_file (str): Model standalone sklearn (si approprié) ou xgboost
        weights_file (str): Poids du réseau à charger (si approprié)
        preprocess_pipeline_file (str): Nom du fichier avec la pipeline de preprocessing
    Raises:
        FileNotFoundError: si model_dir est introuvable
        FileNotFoundError: si config_file est introuvable
        ValueError: si le type de modèle n'est pas reconnu
    '''
    logger.info(f"Rechargement d'un modèle")

    ##############################################
    # Chargement configuration
    ##############################################

    # Récupération path model
    models_dir = utils.get_models_path()
    model_path = None
    for path, subdirs, files in os.walk(models_dir):
        for name in subdirs:
            if name == model_dir:
                model_path = os.path.join(path, name)
    if model_path is None:
        raise FileNotFoundError(f"Impossible de trouver le modèle {model_dir}")

    # Load conf
    conf_path = os.path.join(model_path, config_file)
    if not os.path.exists(conf_path):
        raise FileNotFoundError(f"Impossible de trouver le fichier {conf_path}")
    with open(conf_path, 'r', encoding='utf-8') as f:
        configs = json.load(f)

    ##############################################
    # Récupération type de modèle
    ##############################################

    # Get model type
    model_type_dicts = {
        'model_ridge_classifier': model_ridge_classifier.ModelRidgeClassifier
        'model_logistic_regression_classifier': model_logistic_regression_classifier.ModelLogisticRegressionClassifier,
        'model_svm_classifier': model_svm_classifier.ModelSVMClassifier,
        'model_sgd_classifier': model_sgd_classifier.ModelSGDClassifier,
        'model_knn_classifier': model_knn_classifier.ModelKNNClassifier,
        'model_rf_classifier': model_rf_classifier.ModelRFClassifier,
        'model_gbt_classifier': model_gbt_classifier.ModelGBTClassifier,
        'model_xgboost_classifier': model_xgboost_classifier.ModelXgboostClassifier,
        'model_lgbm_classifier': model_lgbm_classifier.ModelLGBMClassifier,
        'model_dense_classifier': model_dense_classifier.ModelDenseClassifier,
        'model_dense_regressor': model_dense_regressor.ModelDenseRegressor,
        'model_elasticnet_regressor': model_elasticnet_regressor.ModelElasticNetRegressor,
        'model_bayesian_ridge_regressor': model_bayesian_ridge_regressor.ModelBayesianRidgeRegressor,
        'model_kernel_ridge_regressor': model_kernel_ridge_regressor.ModelKernelRidgeRegressor,
        'model_svr_regressor': model_svr_regressor.ModelSVRRegressor,
        'model_sgd_regressor': model_sgd_regressor.ModelSGDRegressor,
        'model_knn_regressor': model_knn_regressor.ModelKNNRegressor,
        'model_pls_regressor': model_pls_regressor.ModelPLSRegression,
        'model_rf_regressor': model_rf_regressor.ModelRFRegressor,
        'model_gbt_regressor': model_gbt_regressor.ModelGBTRegressor,
        'model_xgboost_regressor': model_xgboost_regressor.ModelXgboostRegressor,
        'model_lgbm_regressor': model_lgbm_regressor.ModelLGBMRegressor,
    }
    model_type = configs['model_name']
    if model_type not in model_type_dicts:
        raise ValueError(f"Le type {model_type} n'est pas reconnu, vous devez recharger votre modèle 'à la main'")
    else:
        model_class = model_type_dicts[model_type]

    ##############################################
    # Rechargement du modèle
    ##############################################

    # Reload model
    model = model_class()
    files_dict = {
        'configuration_path': os.path.join(model_path, config_file) if config_file is not None else None,
        'model_pipeline_path': os.path.join(model_path, standalone_file) if standalone_file is not None else None,
        'xgboost_path': os.path.join(model_path, standalone_file) if standalone_file is not None else None,
        'hdf5_path': os.path.join(model_path, weights_file) if weights_file is not None else None,
        'preprocess_pipeline_path': os.path.join(model_path, preprocess_pipeline_file) if preprocess_pipeline_file is not None else None,
    }
    model.reload_from_standalone(**files_dict)

    ##############################################
    # Gestion paramètres et sauvegarde
    ##############################################

    # Sauvegarde du modèle
    json_data = {
        'filename': configs['filename'] if 'filename' in configs.keys() else None,
        'min_rows': configs['min_rows'] if 'min_rows' in configs.keys() else None,
        'preprocess_str': configs['preprocess_str'] if 'preprocess_str' in configs.keys() else None,
        'fit_time': configs['fit_time'] if 'fit_time' in configs.keys() else None,
        'date': configs['date'] if 'date' in configs.keys() else None,
        '_get_model': configs['_get_model'] if '_get_model' in configs.keys() else None,  # Fonctionne car priorité à json_data quand sauvegarde
        '_get_learning_rate_scheduler': configs['_get_learning_rate_scheduler'] if '_get_learning_rate_scheduler' in configs.keys() else None,  # Same
        'custom_objects': configs['custom_objects'] if 'custom_objects' in configs.keys() else None,  # Same
    }

    # On rajoute l'information de la version utilisée pour le training si dispo
    if 'package_version' in configs:
        # On garde quand même par défaut 'trained_version' si présent dans conf
        trained_version = configs['trained_version'] if 'trained_version' in configs.keys() else configs['package_version']
        if trained_version != utils.get_package_version():
            json_data['trained_version'] = trained_version

    # Save
    json_data = {k: v for k, v in json_data.items() if v is not None}  # On ne garde que les items non nulls
    model.save(json_data)

    logger.info(f"Le modèle {model_dir} a été rechargé avec succès")
    logger.info(f"Répertoire du nouveau modèle : {model.model_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # model_X should be the model's directory name: e.g. model_preprocess_pipeline_svm_2019_12_05-12_57_18
    parser.add_argument('-m', '--model_dir', default=None, help='Nom du model à charger')
    parser.add_argument('-c', '--config_file', default='configurations.json', help='Nom du fichier de configuration')
    parser.add_argument('-s', '--standalone_file', default=None, help='Model standalone sklearn (si approprié)')
    parser.add_argument('-w', '--weights_file', default=None, help='Poids du réseau à charger (si approprié)')
    parser.add_argument('-p', '--preprocess_pipeline_file', default=None, help='Pipeline de preproccessing du modèle')
    args = parser.parse_args()
    main(model_dir=args.model_dir, config_file=args.config_file, weights_file=args.weights_file,
         standalone_file=args.standalone_file, preprocess_pipeline_file=args.preprocess_pipeline_file)