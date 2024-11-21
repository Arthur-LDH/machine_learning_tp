#!/usr/bin/env python3

## Preprocessing des données
# Auteurs : Agence dataservices
# Date : 02/12/2019
#
# Ex: poetry run python 1_preprocess_data.py -f dataset_train.csv --target_col Survived

import os
import gc
import time
import dill as pickle
import logging
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime

from ynov import utils
from ynov.preprocessing import preprocess


# Get logger
logger = logging.getLogger('ynov.1_preprocess_data')


def main(filenames: list, preprocessing: str, target_col, sep: str = ',', encoding: str = 'utf-8'):
    '''Fonction principale pour preprocess des jeux de données

    Idée :
        - Pour chaque fichier :
            - On récupère une NOUVELLE Pipeline
            - On fit_transform sur le jeu de données
            - On sauvegarde la pipeline
            - On sauvegarde le fichier preprocessed

    Il est donc important de noter qu'il ne faut PAS preprocess les jeux de validation/test ici !
    En effet, on crée une pipeline par jeu de données (donc moyenne, écart type, etc., peuvent être différents)
    Pour réappliquer une pipeline à un autre fichier, il faut utiliser 2_apply_existing_pipeline.py

    Args:
        filenames (list): Nom des jeux de données à traiter
        preprocessing (str): Preprocessing à appliquer
        target_col (list): Colonne(s) cible(s) du dataframe
            Attention, si plusieurs cibles, votre preprocessing doit être compatible
    Kwargs:
        sep (str): Séparateur du fichier de données
        encoding (str): Encodage du fichier de données
    Raises:
        TypeError: si l'objet target_col n'est pas du type str ou int
        FileNotFoundError : si un des fichiers n'est pas un fichier existant
    '''
    logger.info("Preprocessing des données")

    # Get preprocess dictionnary
    pipelines_dict = preprocess.get_pipelines_dict()

    # Get preprocessing(s) to apply
    if preprocessing is not None:
        # Check presence in pipelines_dict
        if preprocessing not in pipelines_dict.keys():
            raise ValueError(f"Le preprocessing {preprocessing} n'est pas reconnu")
        preprocessing_list = [preprocessing]
    # Par défaut on applique tous les preprocessings
    else:
        preprocessing_list = list(pipelines_dict.keys())

    # Apply each preprocess one by one
    for preprocess_str in preprocessing_list:
        # On ne fait pas 'no_preprocess'
        if preprocess_str == 'no_preprocess':
            continue
        gc.collect()
        # Apply this pipeline to each file
        for filename in filenames:
            logger.info(f'Preprocessing {filename} avec {preprocess_str}')
            # Get pipeline
            # On récupère une nouvelle pipeline à chaque fois
            preprocess_pipeline = preprocess.get_pipeline(preprocess_str)
            # Get paths
            data_path = utils.get_data_path()
            dataset_path = os.path.join(data_path, filename)
            if not os.path.exists(dataset_path):
                raise FileNotFoundError(f"Le fichier {dataset_path} n'existe pas.")
            # Get dataset
            df = pd.read_csv(dataset_path, sep=sep, encoding=encoding)
            # Split X, y
            y = df[target_col]
            X = df.drop(target_col, axis=1)
            # Apply pipeline
            new_X = preprocess_pipeline.fit_transform(X, y)
            # Try to retrieve new columns name (experimental)
            new_df = pd.DataFrame(new_X)
            new_df = preprocess.retrieve_columns_from_pipeline(new_df, preprocess_pipeline)
            # Reinject y
            for col in target_col:
                if col in new_df.columns:
                    new_df.rename(columns={col: f'new_{col}'}, inplace=True)
            new_df[target_col] = y
            # On sauvegarde la pipeline de preprocessing
            # Idée: sauvegarde des pipelines dans un dossier pour être rechargé à la création d'un modèle
            # Elle sera de nouveau sauvegarder dans le modèle pour ne plus dépendre de la sauvegarde dans le dossier pipelines
            pipeline_dir, pipeline_name = get_pipeline_dir(preprocess_str)
            pipeline_path = os.path.join(pipeline_dir, 'pipeline.pkl')
            # On sauvegarde un dictionnaire avec la pipeline et le nom du preprocessing utilisé
            pipeline_dict = {
                                'preprocess_pipeline': preprocess_pipeline,
                                'preprocess_str': preprocess_str,
                            }
            with open(pipeline_path, 'wb') as f:
                pickle.dump(pipeline_dict, f)
            # On sauvegarde aussi un fichier json 'lisible'
            # On sauvegarde aussi un fichier info 'lisible'
            info_path = os.path.join(pipeline_dir, 'pipeline.info')
            with open(info_path, 'w', encoding='utf-8') as f:
                f.write(f"'preprocess_str': {preprocess_str}")
                f.write('\n')
                f.write(f"'preprocess_pipeline': {str(preprocess_pipeline)}")
            # Save dataframe (utf-8, ',')
            basename = Path(filename).stem
            dataset_preprocessed_path = os.path.join(data_path, f'{basename}_{preprocess_str}.csv')
            utils.to_csv(new_df, dataset_preprocessed_path, first_line=f'#{pipeline_name}', sep=',', encoding='utf-8')


def get_pipeline_dir(preprocess_str: str):
    '''Fonction pour récupérer un nouveau répertoire pour sauvegarder une pipeline

    Args:
        preprocess_str (str): nom du preprocessing utilisé
    Returns:
        str: path vers répertoire
        str: nom de la pipeline (i.e. nom du répertoire)
    '''
    pipelines_path = utils.get_pipelines_path()
    pipeline_name = datetime.now().strftime(f"{preprocess_str}_%Y_%m_%d-%H_%M_%S")
    pipeline_dir = os.path.join(pipelines_path, pipeline_name)
    if os.path.isdir(pipeline_dir):
        # Trick : si le répertoire existe déjà, on attends une seconde pour changer de nom ...
        time.sleep(1)
        return get_pipeline_dir(preprocess_str)
    else:
        os.makedirs(pipeline_dir)
    return pipeline_dir, pipeline_name



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filenames', nargs='+', required=True, help='Nom du ou des jeux de données à traiter.')
    parser.add_argument('-p', '--preprocessing', default=None, help='Preprocessing à appliquer. Tous si égal à None.')
    parser.add_argument('--target_col', nargs='+', required=True, help='Colonne(s) cible(s) du dataframe')
    parser.add_argument('--sep', default=',', help='Séparateur utilisé dans le jeu de données.')
    parser.add_argument('--encoding', default="utf-8", help='Encoding du csv')
    args = parser.parse_args()
    main(filenames=args.filenames, preprocessing=args.preprocessing, target_col=args.target_col, sep=args.sep, encoding=args.encoding)