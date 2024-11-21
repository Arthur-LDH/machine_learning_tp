!/usr/bin/env python3

## Merge de plusieurs fichiers
# Auteurs : Agence dataservices
# Date : 04/12/2019
#
# Ex: poetry run python 0_merge_files.py -f original_ALLMAILS_v3_try_3.csv ALLMAILS_v3_try_3.csv original_datasetSeptembre1552.csv original_newdataset_juinV2_2150mails.csv --sep ; --encoding latin-1 -n dataset.csv -c Texte Categorie

import argparse
import logging
import ntpath
import os

import pandas as pd
from ynov import utils


# Get logger
logger = logging.getLogger('ynov.0_merge_files')


def main(filenames: list, cols: list, new_filename: str = 'dataset.csv', sep: str = ',',
         encoding: str = 'utf-8'):
    '''Fonction principale pour merger plusieurs fichiers et ajouter la target
    - /!\\ format sortie : sep , & encoding utf-8 /!\\ -

    Args:
        datadir (str): path to the data dir
    Kwargs:
        new_filename (str): Nom du fichier à créer
        sep (str): Séparateur des fichiers de données
        encoding (str): Encodage des fichiers de données
    Raises:
        ValueError : si le nouveau fichier à créer existe déjà
    '''
    logger.info("Create evolution for all files")

    # Get path
    data_dir = utils.get_data_path()
    paths = [os.path.join(data_dir, filename) for filename in filenames]
    for path in paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Le fichier {path} n'existe pas.")

    # Manage new file
    new_file_path = os.path.join(data_dir, new_filename)
    if os.path.isfile(new_file_path):
        raise FileNotFoundError(f"Le fichier {new_file_path} existe déjà.")

    # Init dataframe
    df = pd.DataFrame(columns=cols)
    # Concat with all files
    for path in paths:
        # Check if first line starts with '#'
        with open(path, 'r', encoding=encoding) as f:
            first_line = f.readline()
        if first_line.startswith('#'):
            raise ValueError('Ce script ne prend pas en compte les fichiers avec des métadata (#)')
        # Load data & concat
        # TODO: à vérifier -> on load tout en string pour éviter les erreurs + fillna
        df_tmp = pd.read_csv(path, sep=sep, encoding=encoding, dtype=str).fillna('')[cols]
        df = pd.concat([df, df_tmp]).reset_index(drop=True)
        utils.display_shape(df)

    # Taille finale
    utils.display_shape(df)

    # Save
    df.to_csv(new_file_path, sep=',', encoding='utf-8', index=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filenames', nargs='+', required=True, help='Fichiers à merge.')
    parser.add_argument('-c', '--cols', nargs='+', required=True, help='Liste des colonnes à garder.')
    parser.add_argument('-n', '--new_filename', default='dataset.csv', help='Nom fichier à créer')
    parser.add_argument('--sep', default=',', help='Séparateur utilisé dans les jeux de données.')
    parser.add_argument('--encoding', default="utf-8", help='Encoding des csv')
    args = parser.parse_args()
    main(filenames=args.filenames, cols=args.cols, new_filename=args.new_filename, sep=args.sep, encoding=args.encoding)
