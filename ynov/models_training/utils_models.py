#!/usr/bin/env python3

## Utils - fonctions-outils pour l'apprentissage
# Auteurs : Agence dataservices
# Date : 02/12/2019
#
# Fonctions :
# - normal_split -> Séparation du dataframe en train et en test
# - stratified_split -> Séparation du dataframe en train et en test de manière stratifiée
# - remove_small_classes -> Fonction pour supprimer les classes pas assez représentées
# - display_train_test_shape -> Fonction pour afficher la taille d'une répartition train/test
# - preprocess_model_multilabel -> Fonction pour préparer une dataframe à un modèle multi-label
# - load_pipeline -> Chargement d'une pipeline depuis le dossier des pipelines
# - load_model -> Fonction pour load un model à partir d'un chemin
# - get_columns_pipeline -> Function to retrieve a pipeline wanted columns, and mandatory ones
# - apply_pipeline -> Fonction pour appliquer une pipeline fitted à une dataframe
# - predict -> Fonction pour obtenir les prédictions d'un modèle sur un contenu
# - predict_with_proba -> Fonction pour obtenir les prédictions d'un modèle sur un contenu, avec probabilités
# - search_hp_cv -> Fonction pour effectuer une recherche d'hyperparamètres


import os
import json
import math
import dill
import dill as pickle
import pprint
import logging
import gc
import shutil
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils.validation import check_is_fitted
from ynov import utils
from ynov.preprocessing import preprocess


# Get logger
logger = logging.getLogger(__name__)


def normal_split(df: pd.DataFrame, test_size: float = 0.25, seed: int = 42):
    '''Séparation du dataframe en train et en test

    Args:
        df (DataFrame): dataframe contenant les offres
    Kwargs:
        test_size (float): proportion représentant la taille du test attendue
        seed (int): seed pour le random
    Raises:
        ValueError: si l'objet test_size n'est compris entre 0 et 1
    Returns:
        DataFrame: dataframe train
        DataFrame: dataframe test
    '''
    logger.debug('Appel à la fonction utils_models.normal_split')
    if test_size < 0 or test_size > 1:
        raise ValueError('L\'objet test_size doit être compris entre 0 et 1')

    # Normal split
    logger.info("Normal split")
    df_train, df_test = train_test_split(df, test_size=test_size, random_state=seed)

    # Display
    display_train_test_shape(df_train, df_test, df_shape=df.shape[0])

    # Return
    return df_train, df_test


def stratified_split(df: pd.DataFrame, col, test_size: float = 0.25, seed: int = 42):
    '''Séparation du dataframe en train et en test de manière stratifiée

    Args:
        df (DataFrame): dataframe contenant les offres
        col (str ou int): colonne sur laquelle faire le split stratifié
    Kwargs:
        test_size (float): proportion représentant la taille du test attendue
        seed (int): seed pour le random
    Raises:
        TypeError: si l'objet col n'est pas du type str ou int
        ValueError: si l'objet test_size n'est compris entre 0 et 1
    Returns:
        DataFrame: dataframe train
        DataFrame: dataframe test
    '''
    logger.debug('Appel à la fonction utils_models.stratified_split')
    if type(col) not in [str, int]:
        raise TypeError('L\'objet col doit être du type str ou int')
    if test_size < 0 or test_size > 1:
        raise ValueError('L\'objet test_size doit être compris entre 0 et 1')

    # Stratified split
    logger.info("Stratified split")
    df = remove_small_classes(df, col, min_rows=math.ceil(1 / test_size))  # minimum de lignes par cat pour split
    df_train, df_test = train_test_split(df, stratify=df[col], test_size=test_size, random_state=seed)

    # Display
    display_train_test_shape(df_train, df_test, df_shape=df.shape[0])

    # Return
    return df_train, df_test


def remove_small_classes(df: pd.DataFrame, col, min_rows: int = 2):
    '''Fonction pour supprimer les classes pas assez représentées

    Args:
        df (pd.DataFrame): jeu de données
        col (str ou int): class column
    Kwargs:
        min_rows (int): nombre de lignes minimum dans le jeu d'apprentissage (default: 2)
    Raises:
        TypeError: si l'objet col n'est pas du type str ou int
        ValueError: si l'objet min_rows n'est pas strictement positif
    Returns:
        pd.DataFrame: nouveau jeu de données
    '''
    logger.debug('Appel à la fonction utils_models.remove_small_classes')
    if type(col) not in [str, int]:
        raise TypeError('L\'objet col doit être du type str ou int')
    if min_rows < 1:
        raise ValueError("L'objet min_rows doit être strictement positif")

    # Recherche classes avec moins de min_rows lignes
    v_count = df[col].value_counts()
    classes_to_remove = v_count[v_count < min_rows].index.values
    for cl in classes_to_remove:
        logger.warning(
            f"/!\\ /!\\ /!\\ /!\\ La classe {cl} a moins de {min_rows} lignes dans le jeu d'entrainement. Cette classe est automatiquement supprimée du dataset."
        )
    return df[~df[col].isin(classes_to_remove)]


def display_train_test_shape(df_train: pd.DataFrame, df_test: pd.DataFrame, df_shape: int = None):
    '''Fonction pour afficher la taille d'une répartition train/test

    Args:
        df_train (pd.DataFrame): jeu de train
        df_test (pd.DataFrame): jeu de test
    Kwargs:
        df_shape (int): taille du jeu de données initial
    Raises:
        ValueError: si l'objet df_shape n'est pas strictement positif
    '''
    logger.debug('Appel à la fonction utils_models.display_train_test_shape')
    if df_shape is not None and df_shape < 1:
        raise ValueError("L'objet df_shape doit être strictement positif")

    # Process
    if df_shape is None:
        df_shape = df_train.shape[0] + df_test.shape[0]
    logger.info(f"Il y a {df_train.shape[0]} lignes dans la table train et {df_test.shape[0]} dans la table test.")
    logger.info(f"{round(100 * df_train.shape[0] / df_shape, 2)}% des données sont dans le train")
    logger.info(f"{round(100 * df_test.shape[0] / df_shape, 2)}% des données sont dans le test")


def preprocess_model_multilabel(df: pd.DataFrame, y_col, classes: list = None):
    '''Fonction pour préparer une dataframe à un modèle de classification multi-label

    Args:
        df (pd.DataFrame): Nom du jeu de données pour apprentissage
            Ce jeu de données doit être preprocessé.
            Exemple:
                # Group by & apply tuple to y_col
                x_cols = [col for col in list(df.columns) if col != y_col]
                df = pd.DataFrame(df.groupby(x_cols)[y_col].apply(tuple))
        y_col (str ou int): nom de la colonne à utiliser pour l'apprentissage - y
    Kwargs:
        classes (list): liste de classes à considérer
    Raises:
        TypeError: si l'objet y_col n'est pas du type str ou int
        TypeError: si l'objet ohe n'est pas du type bool
    Returns:
        DataFrame: dataframe pour l'apprentissage
        list: liste des colonnes 'y'
    '''
    # TODO: ajouter possibilité sortie en sparse
    logger.info("Preprocess dataframe pour model multi-label")
    if type(y_col) not in (str, int):
        raise TypeError('L\'objet y_col doit être du type str ou int.')
    # Process
    logger.info("Preparing dataset for multi-label format. Might take several minutes.")
    # /!\ The reset_index is compulsory in order to have the same indexes between df, and MLB transformed values
    df = df.reset_index(drop=True)
    # Apply MLB
    mlb = MultiLabelBinarizer(classes=classes)
    df = df.assign(**pd.DataFrame(mlb.fit_transform(df[y_col]), columns=mlb.classes_))
    # Return dataframe & y_cols (i.e. classes)
    return df, list(mlb.classes_)


def load_pipeline(pipeline_dir: str, is_path: bool = False):
    '''Chargement d'une pipeline depuis le dossier des pipelines

    Args:
        pipeline_dir (str): nom du dossier contenant la pipeline à récupérer
    Kwargs:
        is_path (bool): Si chemin du dossier au lieu du nom (permet de charger des modèles d'ailleurs)
    Raises:
        FileNotFoundError : si le dossier pipeline_dir n'existe pas
    Returns:
        Pipeline: pipeline reloaded
        str: nom du preprocessing utilisé
    '''
    # Si pipeline_dir == None, on backup sur "no_preprocess"
    if pipeline_dir is None:
        logger.warning(f"Le répertoire de la pipeline est à None. On backup sur 'no_preprocess'")
        preprocess_str = "no_preprocess"
        preprocess_pipeline = preprocess.get_pipeline(preprocess_str) # Attention, besoin d'être fit
        return preprocess_pipeline, preprocess_str

    # Sinon, cas nominal
    # Find pipeline path
    if not is_path:
        pipelines_dir = utils.get_pipelines_path()
        pipeline_path = None
        for path, subdirs, files in os.walk(pipelines_dir):
            for name in subdirs:
                if name == pipeline_dir:
                    pipeline_path = os.path.join(path, name)
        if pipeline_path is None:
            raise FileNotFoundError(f"Impossible de trouver la pipeline {pipeline_dir}")
    else:
        pipeline_path = pipeline_dir
        if not os.path.exists(pipeline_path):
            raise FileNotFoundError(f"Impossible de trouver la pipeline {pipeline_path} (considée comme un chemin)")

    # Get pipeline
    pipeline_path = os.path.join(pipeline_path, 'pipeline.pkl')
    with open(pipeline_path, 'rb') as f:
        pipeline_dict = pickle.load(f)

    # Return
    return pipeline_dict['preprocess_pipeline'], pipeline_dict['preprocess_str']


def load_model(model_dir: str, is_path: bool = False):
    '''Fonction pour load un model à partir d'un chemin

    Args:
        model_dir (str): Nom du dossier contenant le modèle (e.g. model_autres_2019_11_07-13_43_19)
    Kwargs:
        is_path (bool): Si chemin du dossier au lieu du nom (permet de charger des modèles d'ailleurs)
    Raises:
        FileNotFoundError : si le dossier model_dir n'existe pas
    Returns:
        ?: modèle
        dict: configurations du modèle
    '''
    logger.debug('Appel à la fonction utils_models.load_model')

    # Find model path
    if not is_path:
        models_dir = utils.get_models_path()
        model_path = None
        for path, subdirs, files in os.walk(models_dir):
            for name in subdirs:
                if name == model_dir:
                    model_path = os.path.join(path, name)
        if model_path is None:
            raise FileNotFoundError(f"Impossible de trouver le modèle {model_dir}")
    else:
        model_path = model_dir
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Impossible de trouver le modèle {model_path} (considéré comme un chemin)")


    # Get configs
    configuration_path = os.path.join(model_path, 'configurations.json')
    with open(configuration_path, 'r', encoding='utf-8') as f:
        configs = json.load(f)
    # Can't set int as keys in json, so need to cast it after realoading
    # dict_classes keys are always ints
    if 'dict_classes' in configs.keys() and configs['dict_classes'] is not None:
        configs['dict_classes'] = {int(k): v for k, v in configs['dict_classes'].items()}

    # Load model
    pkl_path = os.path.join(model_path, f"{configs['model_name']}.pkl")
    with open(pkl_path, 'rb') as f:
        model = pickle.load(f)

    # Change model_dir if diff
    if model_path != model.model_dir:
        model.model_dir = model_path
        configs['model_dir'] = model_path

    # Load specifics
    hdf5_path = os.path.join(model_path, 'best.hdf5')

    # Check for keras model
    if os.path.exists(hdf5_path):
        model.model = model.reload_model(hdf5_path)

    # Display if GPU is being used
    model.display_if_gpu_activated()

    # Return model & configs
    return model, configs


def get_columns_pipeline(preprocess_pipeline: ColumnTransformer):
    '''Function to retrieve a pipeline wanted columns, and mandatory ones

    Args:
        preprocess_pipeline (ColumnTransformer): preprocessing pipeline
    Returns:
        list: list of columns in
        list: list of mandatory ones
    '''
    # On commence par vérifier que la pipeline est bien fitted
    check_is_fitted(preprocess_pipeline)
    # On récupère les noms de colonnes en entrées
    columns_in = preprocess_pipeline._feature_names_in.tolist()
    # On récupère les noms de colonnes "obligatoires"
    if preprocess_pipeline._remainder[1] == 'drop':
        # Si drop, on récupère depuis _columns
        mandatory_columns = list(utils.flatten(preprocess_pipeline._columns))
    else:
        # Sinon, il faut toutes les colonnes
        mandatory_columns = columns_in
    # Returns
    return columns_in, mandatory_columns


def apply_pipeline(df: pd.DataFrame, preprocess_pipeline: ColumnTransformer):
    '''Fonction pour appliquer une pipeline fitted à une dataframe

    Problème :
        La pipeline attend en entrée les mêmes colonnes, et dans le même ordre
        Même si certaines colonnes sont par la suite drop (donc inutile)
            -> https://github.com/scikit-learn/scikit-learn/issues/14251
    Solution (expérimental 14/04/2021):
        On ajoute les colonnes "inutiles" par des NaNs

    Args:
        df (pd.DataFrame): dataframe à preprocessed
        preprocess_pipeline (ColumnTransformer): pipeline à utiliser
    Raises:
        ValueError: s'il manque des colonnes obligatoires
    Returns:
        pd.DataFrame: DataFrame preprocessed
    '''
    columns_in, mandatory_columns = get_columns_pipeline(preprocess_pipeline)

    # On enlève les colonnes "en trop"
    df = df[[col for col in df.columns if col in columns_in]]
    optionals_columns = [col for col in columns_in if col not in mandatory_columns]

    # On vérifie qu'on a bien les colonnes obligatoires
    missing_mandatory_columns = [col for col in mandatory_columns if col not in df.columns]
    if len(missing_mandatory_columns) > 0:
        for missing_col in missing_mandatory_columns:
            logger.error(f"Colonne manquante dans votre jeu de données : {missing_col}")
        raise ValueError("Il manque des colonnes obligatoires pour faire le preprocessing")

    # On rajoute les non obligatoires si pas déjà dans df
    # Note : concerne seulement le cas remainder = "drop" (cas nominal)
    missing_optionals_columns = [col for col in optionals_columns if col not in df.columns]
    for col in missing_optionals_columns:
        logger.warning(f'La colonne {col} est manquante pour le preprocessing.')
        logger.warning(f'Expérimental : normalement inutile -> on crée une colonne vide')
        df[col] = np.nan

    # Apply transform on reordered columns
    preprocessed_x = preprocess_pipeline.transform(df[columns_in])
    # Reconstruct dataframe & return
    preprocessed_df = pd.DataFrame(preprocessed_x)
    preprocessed_df = preprocess.retrieve_columns_from_pipeline(preprocessed_df, preprocess_pipeline)
    return preprocessed_df


def predict(content: pd.DataFrame, model):
    '''Fonction pour obtenir les prédictions d'un modèle sur un contenu

    Args:
        content (pd.DataFrame): Nouveau contenu sur lequel effectué une prédiction
        model (?): modèle à utiliser pour obtenir les prédictions
    Returns:
        REGRESSION :
            float: prediction
        CLASSIFICATION MONOLABEL:
            str: prediction
        CLASSIFICATION MULTILABEL:
            tuple: predictions

        Si plusieurs éléments -> list
    '''
    logger.debug('Appel à la fonction utils_models.predict')

    # Apply preprocessing
    if model.preprocess_pipeline is not None:
        df_prep = apply_pipeline(content, model.preprocess_pipeline)
    else:
        df_prep = content.copy()
        logger.warning("On ne trouve pas de pipeline de preprocessing - on considère no preprocessing, mais ce n'est pas normal !")

    # Get prediction
    predictions = model.predict(df_prep)

    # Inverse transform (neede for classification)
    predictions = model.inverse_transform(predictions)

    # Return only first element if dataframe has one row
    if content.shape[0] == 1:
        predictions = predictions[0]

    # Return
    return predictions


def predict_with_proba(content: pd.DataFrame, model):
    '''Fonction pour obtenir les prédictions d'un modèle sur un contenu, avec probabilités

    Args:
        content (pd.DataFrame): Nouveau contenu sur lequel effectué une prédiction
        model (?): modèle à utiliser pour obtenir les prédictions
    Raises:
        ValueError: si le modèle n'est pas du type classifier
    Returns:
        CLASSIFICATION MONOLABEL:
            str: prediction
        CLASSIFICATION MULTILABEL:
            tuple: predictions

        Si plusieurs éléments -> list
    '''
    logger.debug('Appel à la fonction utils_models.predict_with_proba')

    # Regressions
    if not model.model_type == 'classifier':
        raise ValueError(f"Le type de modèle ({model.model_type}) n'est pas supporté par la fonction predict_with_proba")

    # Apply preprocessing
    if model.preprocess_pipeline is not None:
        df_prep = apply_pipeline(content, model.preprocess_pipeline)
    else:
        df_prep = content.copy()
        logger.warning("On ne trouve pas de pipeline de preprocessing - on considère no preprocessing, mais ce n'est pas normal !")

    # Get predictions
    predictions, probas = model.predict_with_proba(df_prep)

    # Rework format
    if not model.multi_label:
        prediction = model.inverse_transform(predictions)
        proba = list(probas.max(axis=1))
    else:
        prediction = [tuple(np.array(model.list_classes).compress(indicators)) for indicators in predictions]
        proba = [tuple(np.array(probas[i]).compress(indicators)) for i, indicators in enumerate(predictions)]

    # Return only first element if dataframe has one row
    if content.shape[0] == 1:
        prediction = prediction[0]
        proba = proba[0]

    # Return prediction & proba
    return prediction, proba


def search_hp_cv_classifier(model_cls, model_params: dict, hp_params: dict, scoring_fn, kwargs_fit: dict, n_splits: int = 5):
    '''Fonction pour effectuer une recherche d'hyperparamètres

    Args:
        model_cls (?): classe de modèle sur laquelle effectuer une recherche d'hyperparamètres
        model_params (dict): ensemble de paramètres "fixes" du modèle (e.g. x_col, y_col).
            Doit contenir 'multi_label'.
        hp_params (dict): ensemble de paramètres "variables" avec lesquels effectuer une recherche d'hyperparamètres
        scoring_fn (str ou func): fonction de scoring à maximiser
            Cette fonction doit prendre en entrée un dictionnaire qui contient des métriques
            e.g. {'F1-Score': 0.85, 'Accuracy': 0.57, 'Precision': 0.64, 'Recall': 0.90}
        kwargs_fit (dict): ensemble de kwargs à passer à la fonction fit
            Doit contenir 'x_train' et 'y_train'
    Kwargs:
        n_splits (int): nombre de folds à utiliser
    Raises:
        TypeError: Si scoring_fn n'est pas du type str ou une fonction
        ValueError: Si scoring_fn n'est pas une string reconnue
        ValueError: Si multi_label n'est pas une entrée de model_params
        ValueError: Si x_train n'est pas une entrée de kwargs_fit
        ValueError: Si y_train n'est pas une entrée de kwargs_fit
        ValueError: Si une entrée de model_params est aussi dans hp_params
        ValueError: Si les entrées de hp_params sont des listes
        ValueError: Si les entrées de hp_params ne font pas la même longueur
        ValueError: Si le nombre de split de crossvalidation est inférieur ou égal à 1
    Returns:
        ?: best model à "fitter" sur l'ensemble des données
    '''
    logger.debug('Appel à la fonction utils_models.search_hp_cv')
    list_known_scoring = ['accuracy', 'f1', 'precision', 'recall']

    #################
    # Gestion erreurs
    #################

    if type(scoring_fn) is not str and not callable(scoring_fn):
        raise TypeError("L'argument scoring_fn doit être du type str ou une fonction")

    if type(scoring_fn) is str and scoring_fn not in list_known_scoring:
        raise ValueError(f"L'entrée {scoring_fn} n'est pas une valeur possible pour scoring_fn")

    if 'multi_label' not in model_params.keys():
        raise ValueError("L'entrée 'multi_label' doit être présente dans le dictionnaire model_params")

    if 'x_train' not in kwargs_fit.keys():
        raise ValueError("L'entrée 'x_train' doit être présente dans le dictionnaire kwargs_fit")

    if 'y_train' not in kwargs_fit.keys():
        raise ValueError("L'entrée 'y_train' doit être présente dans le dictionnaire kwargs_fit")

    if any([k in hp_params.keys() for k in model_params.keys()]):
        # On ne peut pas avoir en même temps une clé "fixe" et "variable"
        raise ValueError("Une clé du dictionnaire model_params est aussi présente dans le dictionnaire hp_params")

    if any([type(_) != list for _ in hp_params.values()]):
        raise ValueError("Les entrées de hp_params doivent être des listes")

    if len(set([len(_) for _ in hp_params.values()])) != 1:
        raise ValueError("Les entrées de hp_params doivent faire la même longueur")

    if n_splits <= 1:
        raise ValueError(f"Le nombre de split de crossvalidation ({n_splits}) doit être supérieur à 1")

    #################
    # Gestion scoring
    #################

    # Récupération fonction de scoring
    if scoring_fn == 'accuracy':
        scoring_fn = lambda x: x['Accuracy']
    elif scoring_fn == 'f1':
        scoring_fn = lambda x: x['F1-Score']
    elif scoring_fn == 'precision':
        scoring_fn = lambda x: x['Precision']
    elif scoring_fn == 'recall':
        scoring_fn = lambda x: x['Recall']

    #################
    # Gestion format x_train & y_train
    #################

    if type(kwargs_fit['x_train']) not in [pd.Series, pd.DataFrame]:
        kwargs_fit['x_train'] = pd.Series(kwargs_fit['x_train'].copy())

    if type(kwargs_fit['y_train']) not in [pd.Series, pd.DataFrame]:
        kwargs_fit['y_train'] = pd.Series(kwargs_fit['y_train'].copy())

    #################
    # Process
    #################

    # On boucle sur les hyperparamètres
    nb_search = len(list(hp_params.values())[0])
    logger.info("Début de la recherche d'hyperparamètres")
    logger.info(f"Nous allons fit {nb_search} (nb recherches) x {n_splits} (nb splits CV) = {nb_search * n_splits} modèles")

    # DataFrame de stockage des métriques :
    metrics_df = pd.DataFrame(columns=['index_params', 'index_fold', 'Score', 'Accuracy', 'F1-Score', 'Precision', 'Recall'])
    for i in range(nb_search):

        # Display informations
        logger.info(f"Recherche n°{i + 1}")
        tmp_hp_params = {k: v[i] for k, v in hp_params.items()}
        logger.info("Hyperparamètres testés : ")
        logger.info(pprint.pformat(tmp_hp_params))

        # Get folds (shuffle conseillé car les classes peuvent être ordonnées)
        if model_params['multi_label'] == True:
            k_fold = KFold(n_splits=n_splits, shuffle=True)  # On ne peut pas stratified sur du multi label
        else:
            k_fold = StratifiedKFold(n_splits=n_splits, shuffle=True)

        # Process each fold
        for j, (train_index, valid_index) in enumerate(k_fold.split(kwargs_fit['x_train'], kwargs_fit['y_train'])):
            logger.info(f"Recherche n°{i + 1}/{nb_search} - fit n°{j + 1}/{n_splits}")
            # get tmp x, y
            x_train, x_valid = kwargs_fit['x_train'].iloc[train_index], kwargs_fit['x_train'].iloc[valid_index]
            y_train, y_valid = kwargs_fit['y_train'].iloc[train_index], kwargs_fit['y_train'].iloc[valid_index]
            # Get tmp model
            # On gère le model_dir
            tmp_model_dir = os.path.join(utils.get_models_path(), datetime.now().strftime("tmp_%Y_%m_%d-%H_%M_%S"))
            # La formulation suivante priorise le dernier dictionnaire
            # On force un répertoire temporaire et un niveau de sauvegarde minimal (on souhaite juste avoir les métriques)
            model_tmp = model_cls(**{**model_params, **tmp_hp_params, **{'model_dir': tmp_model_dir, 'level_save': 'LOW'}})
            # On set le log level à ERROR
            model_tmp.logger.setLevel(logging.ERROR)
            # Let's fit ! (priorité au dernier dictionnaire)
            model_tmp.fit(**{**kwargs_fit, **{'x_train': x_train, 'y_train': y_train, 'x_valid': x_valid, 'y_valid': y_valid}})
            # Let's predict !
            y_pred = model_tmp.predict(x_valid)
            # Get metrics !
            metrics_func = model_tmp.get_metrics_simple_multilabel if model_tmp.multi_label else model_tmp.get_metrics_simple_monolabel
            metrics_tmp = metrics_func(y_valid, y_pred)
            metrics_tmp = metrics_tmp[metrics_tmp.Label == "All"].copy()  # Ajout copy pour éviter settingwithcopy de padnas
            metrics_tmp["Score"] = scoring_fn(metrics_tmp.iloc[0].to_dict())
            metrics_tmp["index_params"] = i
            metrics_tmp["index_fold"] = j
            metrics_tmp = metrics_tmp[metrics_df.columns]  # On garde seulement les colonnes nécessaires
            metrics_df = pd.concat([metrics_df, metrics_tmp], ignore_index=True)
            # Supression du modèle temporaire : le modèle final devra être ré-entraîné sur la totalité des données
            del model_tmp
            gc.collect()
            shutil.rmtree(tmp_model_dir)
        # Display score
        logger.info(f"Score pour la recherche n°{i + 1}: {metrics_df[metrics_df['index_params'] == i]['Score'].mean()}")

    # Agrégation des métriques de toutes les folds
    metrics_df = metrics_df.join(metrics_df[['index_params', 'Score']].groupby('index_params').mean().rename({'Score':'mean_score'}, axis=1), on='index_params', how='left')

    # On sélectionne le set de paramètres ayant obtenu le meilleur score moyen (entre les différentes folds)
    best_index = metrics_df[metrics_df.mean_score == metrics_df.mean_score.max()]["index_params"].values[0]
    best_params = {k: v[best_index] for k, v in hp_params.items()}
    logger.info(f"Meilleur résultat pour le set de paramètres n°{best_index + 1}: {pprint.pformat(best_params)}")

    # On instancie un nouveau modèle avec ces paramètres
    best_model = model_cls(**{**model_params, **best_params})

    # Sauvegarde du rapport de métriques de la recherche d'hyper paramètres, et des paramètres testés
    csv_path = os.path.join(best_model.model_dir, f"hyper_params_results.csv")
    metrics_df.to_csv(csv_path, sep=',', index=False, encoding='utf-8')
    json_data = {
        'model_params': model_params,
        'scoring_fn': dill.source.getsourcelines(scoring_fn)[0],
        'n_splits': n_splits,
        'hp_params_set': {i: {k: v[i] for k, v in hp_params.items()} for i in range(nb_search)},
    }
    json_path = os.path.join(best_model.model_dir, f"hyper_params_tested.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=4, cls=utils.NpEncoder)

    # TODO: On est obligé de reset le niveau de logging, qui (TO FIX) est associé à la classe
    best_model.logger.setLevel(logging.getLogger('ynov').getEffectiveLevel())

    # Return model to be fitted
    return best_model


if __name__ == '__main__':
    logger.error("Ce script ne doit pas être exécuté, il s'agit d'un package.")