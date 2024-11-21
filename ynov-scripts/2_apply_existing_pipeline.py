#!/usr/bin/env python3

## Preprocessing des données - Application d'une pipeline existante à un fichier
# Auteurs : Agence dataservices
# Date : 13/04/2021
#
# Ex: poetry run python 2_apply_existing_pipeline.py -f dataset_valid.csv -p preprocess_P1_2021_04_09-14_34_48 --target_col Survived

import os
import logging
import argparse
import pandas as pd
from pathlib import Path

from ynov import utils
from ynov.models_training import utils_models
from ynov.preprocessing import preprocess


# Get logger
logger = logging.getLogger('ynov.2_apply_existing_pipeline')


def main(filenames: list, pipeline: str, target_col, sep: str = ',', encoding: str = 'utf-8'):
    '''Main fonction to preprocess the main dataset

    Args:
        filenames (list): Nom des jeux de données à traiter
        pipeline (str): Pipeline (existante) à appliquer
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

    # Get pipeline
    preprocess_pipeline, preprocess_str = utils_models.load_pipeline(pipeline)

    # Apply this pipeline to each file
    for filename in filenames:
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
        new_X = utils_models.apply_pipeline(X, preprocess_pipeline)
        # Try to retrieve new columns name (experimental)
        new_df = pd.DataFrame(new_X)
        new_df = preprocess.retrieve_columns_from_pipeline(new_df, preprocess_pipeline)
        # Reinject y
        for col in target_col:
            if col in new_df.columns:
                new_df.rename(columns={col: f'new_{col}'}, inplace=True)
        new_df[target_col] = y
        # Save dataframe (utf-8, ',')
        basename = Path(filename).stem
        dataset_preprocessed_path = os.path.join(data_path, f'{basename}_{preprocess_str}.csv')
        utils.to_csv(new_df, dataset_preprocessed_path, first_line=f'#{pipeline}', sep=',', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filenames', nargs='+', required=True, help='Nom du ou des jeux de données à traiter.')
    parser.add_argument('-p', '--pipeline', default=None, help='Pipeline (existante) à appliquer.')
    parser.add_argument('--target_col', nargs='+', required=True, help='Colonne(s) cible(s) du dataframe')
    parser.add_argument('--sep', default=',', help='Séparateur utilisé dans le jeu de données.')
    parser.add_argument('--encoding', default="utf-8", help='Encoding du csv')
    args = parser.parse_args()
    main(filenames=args.filenames, pipeline=args.pipeline, target_col=args.target_col, sep=args.sep, encoding=args.encoding)