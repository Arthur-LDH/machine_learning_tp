#!/usr/bin/env python3

## Extractions de samples à partir d'un fichier
# Auteurs : Agence dataservices
# Date : 02/12/2019
#
# Ex: poetry run python 0_create_samples.py -f original_newdataset_juinV2_2150mails.csv --encoding latin-1 --sep ; -n 100

import os
import ntpath
import logging
import argparse

import pandas as pd
from ynov import utils


# Get logger
logger = logging.getLogger('ynov.0_create_samples')


def main(filenames: list, sep: str = ',', encoding: str = 'utf-8', n_samples: int = 100):
    '''Fonction principale pour extraire un subset de données depuis un fichier

    Args:
        filenames (list): Nom des fichier de données à traiter
    Kwargs:
        sep (str): Séparateur du fichier de données
        encoding (str): Encodage du fichier de données
        n_samples (int): Nombre de données à extraire
    '''
    logger.info(f"Création de samples")

    # Si aucun fichier en entrée, en tente de process tous les fichiers .csv
    if len(filenames) == 0:
        # Get path
        data_path = utils.get_data_path()
        # Get file lists
        files = [os.path.join(data_path, f) for f in os.listdir(data_path)]
        filenames = [f for f in files if os.path.isfile(f) and f.endswith('.csv') and not f.endswith('_sample.csv')]

    for filename in filenames:
        # Dans le cas ou mauvais encoding, on ne gère pas les erreurs (on skip !)
        try:
            process_file(filename, sep=sep, encoding=encoding, n_samples=n_samples)
        except:
            logger.warning(f"Impossible de lire le fichier {filename} avec l'encoding {encoding} et le séparateur {sep} ! SKIP !!!")
            continue


def process_file(filename: str, sep: str = ',', encoding: str = 'utf-8', n_samples: int = 100):
    '''Fonction principale pour extraire un subset de données depuis un fichier

    Args:
        filename (str): Nom du fichier de données pour le test
    Kwargs:
        sep (str): Séparateur du fichier de données
        encoding (str): Encodage du fichier de données
        n_samples (int): Nombre de données à extraire
    Raises:
        ValueError : si l'objet filename ne termine pas par .csv
        FileNotFoundError : si l'objet filename n'est pas un fichier existant
    '''
    logger.info(f"Création sample du fichier {filename}")
    if not filename.endswith('.csv'):
        raise ValueError('L\'objet filename doit terminé par ".csv".')

    # Get path
    data_dir = utils.get_data_path()
    file_path = os.path.join(data_dir, filename)
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Le fichier {filename} n'existe pas.")

    # Process
    base_file_name = '.'.join(ntpath.basename(file_path).split('.')[:-1])
    new_file_name = f"{base_file_name}_sample.csv"
    new_path = os.path.join(data_dir, new_file_name)
    if os.path.exists(new_path):
        logger.info(f"{new_path} already exists. Pass.")
        pass
    else:
        logger.info(f"Processing {base_file_name}.")
        # Récupération dataset & first_line
        df, first_line = utils.read_csv(file_path, sep=sep, encoding=encoding, dtype=str)
        # Get extract
        extract = df.sample(n=min(n_samples, df.shape[0]))
        utils.to_csv(extract, new_path, first_line=first_line, sep=',', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filenames', nargs='+', default=[], help='Nom des jeux de données à traiter -> si vide, tous les csv')
    parser.add_argument('--sep', default=',', help='Séparateur utilisé dans le jeu de données.')
    parser.add_argument('--encoding', default="utf-8", help='Encoding du csv')
    parser.add_argument('-n', '--n_samples', type=int, default=100, help='Nombre de données à extraire')
    args = parser.parse_args()
    main(filenames=args.filenames, sep=args.sep, encoding=args.encoding, n_samples=args.n_samples)