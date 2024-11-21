#!/usr/bin/env python3

## Définition d'une classe permettant d'abstraire le fonctionnement de MlFlow
# Auteurs : Nicolas G, Alex G.
# Date : 06/10/2020
#
# Classes :
# - ModelLogger -> Classe ermettant d'abstraire le fonctionnement de MlFlow

import logging
import re
import math
import uuid
import mlflow
import socket

# On desactive les warnings GIT de mlflow
import os

os.environ["GIT_PYTHON_REFRESH"] = "quiet"


def is_running(host: str, port: int, logger):
    '''Fonction permettant de vérifier si un host est up & running

    Args:
        host (str): URI de l'host
        port (int): port à checker
        logger (?): logger d'une instance de ModelLogger
    Returns:
        bool: si l'host est joignable
    '''
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5)
    reachable = False
    try:
        host = re.sub(r'(?i)http(s)*://', '', host)  # Remove http:// to test connexion
        sock.connect((host, port))
        sock.shutdown(socket.SHUT_RDWR)
        reachable = True
    except:
        logger.error(f'Monitoring - MlFlow  @ {host} not reachable => nothing will be stored')
    finally:
        sock.close()

    # Return state
    return reachable


def is_local(host: str):
    '''Fonction to check is ml flow is running in local

    Args:
        host (str): URI de l'host
    Returns:
        bool: whether ml flow is running in local
    '''
    l1 = len(host)
    host = re.sub(r'(?i)http(s)*://', '', host)
    l2 = len(host)
    if l1 == l2:  # pas de http
        return True
    else:
        return False


def is_mlflow_up(func):
    '''Decorator permettant de check si le serveur mlflow est up & running
    avant d'appeler la fonction décorée

    Args:
        func (?): fonction à décorer
    Returns:
        ?: wrapper
    '''

    def wrapper(self, *args, **kwargs):

        # On run qui si running à True (i.e. connexion initiale ok)
        if self.running:

            # On check si on peut run
            if is_local(self.tracking_uri):
                to_run = True  # OK car local
            elif is_running(self.tracking_uri, 80, self.logger):
                to_run = True  # OK car still running
            else:
                to_run = False  # KO

            # run si possible
            if to_run:
                try:
                    func(self, *args, **kwargs)
                except Exception as e:  # gestion erreurs de MLFLOW (on continue le process)
                    self.logger.error("Impossible de logger sur ML FLOW")
                    self.logger.error(repr(e))
            # Else : do nothing (error already logged)

    wrapper.wrapped_fn = func  # For test purposes

    return wrapper


class ModelLogger:
    '''Classe permettant d'abstraire le fonctionnement de MlFlow'''

    _default_name = f'ynov-approche-{uuid.uuid4()}'
    _default_tracking_uri = ''

    def __init__(self, tracking_uri: str = None, experiment_name: str = None):
        '''Initialisation de la classe

        Kwargs:
            tracking_uri (str): URI du tracking server
            experiment_name (str): nom de l'expérimentaiton à activer
        Raises:
            TypeError : si l'objet tracking_uri n'est pas du type str
            TypeError : si l'objet experiment_name n'est pas du type str
        '''
        if tracking_uri is not None and type(tracking_uri) is not str:
            raise TypeError('tracking_uri doit être du type str')
        if experiment_name is not None and type(experiment_name) is not str:
            raise TypeError('experiment_name doit être de type str')

        # Get logger
        self.logger = logging.getLogger(__name__)
        # Set tracking URI & experiment name
        self.tracking_uri = tracking_uri if tracking_uri is not None else self._default_tracking_uri
        self.experiment_name = experiment_name if experiment_name is not None else self._default_name
        # On initie le tracking
        # On met un try...except pour tester si ml flow est bien joignable
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(f'/{self.experiment_name}')
            self.logger.info(f'Ml Flow running, metrics available @ {self.tracking_uri}')
            self.running = True
        except:
            self.logger.warning(f"Host {self.tracking_uri} is not reachable. ML flow won't run")
            self.logger.warning("Attention, pour un travail en local, ML flow n'accepte que des chemins relatifs ...")
            self.logger.warning("Pensez à utiliser os.path.relpath()")
            self.running = False

    def stop_run(self):
        '''Stop an MLflow run'''
        try:
            mlflow.end_run()
        except:
            self.logger.error("Impossible d'arrêter le run de MLflow")

    @is_mlflow_up
    def log_metric(self, key: str, value, step: int = None):
        '''Log une métrique sur ML Flow

        Args:
            key (str): nom de la métrique
            value (float, ?): valeur de la métrique
        Kwargs:
            step (int): metric step
        Raises:
            TypeError : si l'objet key n'est pas du type str
            TypeError : si l'objet step n'est pas du type int
        '''
        if type(key) is not str:
            raise TypeError('key must be str')
        if step is not None and type(step) != int:
            raise TypeError('step must be int')

        # Check for None
        if value is None:
            value = math.nan

        # Log métrique
        mlflow.log_metric(key, value, step)

    @is_mlflow_up
    def log_metrics(self, metrics: dict, step: int = None):
        '''Log un ensemble de métriques sur ML Flow

        Args:
            metrics (dict): métriques à logger
        Kwargs:
            step (int): metric step
        Raises:
            TypeError : si l'objet metrics n'est pas du type dict
            TypeError : si l'objet step n'est pas du type int
        '''
        if type(metrics) is not dict:
            raise TypeError('metrics must be dict')
        if step is not None and type(step) != int:
            raise TypeError('step must be int')

        # Check for Nones
        for k, v in metrics.items():
            if v is None:
                metrics[k] = math.nan

        # Log métriques
        mlflow.log_metrics(metrics, step)

    @is_mlflow_up
    def log_param(self, key: str, value):
        '''Log d'un paramètre sur ML Flow

        Args:
            key (str): nom du paramètre
            value (str, ?): valeur du paramètre (sera "stringified" si pas str)
        Raises:
            TypeError : si l'objet key n'est pas du type str
        '''
        if type(key) is not str:
            raise TypeError('key must be str')
        if value is None:
            value = 'None'

        # Log parameter
        mlflow.log_param(key, value)

    @is_mlflow_up
    def log_params(self, params: dict):
        '''Log d'un ensemble de paramètres sur ML Flow

        Args:
            params (dict): nom du paramètre
        Raises:
            TypeError : si l'objet params n'est pas du type dict
        '''
        if type(params) is not dict:
            raise TypeError('params must be dict')

        # Check for Nones
        for k, v in metrics.items():
            if v is None:
                metrics[k] = 'None'

        # Log parameters
        mlflow.log_params(params)

    @is_mlflow_up
    def set_tag(self, key: str, value):
        '''Log d'un tag sur ML Flow

        Args:
            key (str): nom du tag
            value (str, ?): valeur du tag (sera "stringified" si pas str)
        Raises:
            TypeError : si l'objet key n'est pas du type str
            ValueError : si l'objet value est égal à None
        '''
        if type(key) is not str:
            raise TypeError('key must be str')
        if value is None:
            raise ValueError('value must not be None')

        # Log tag
        mlflow.set_tag(key, value)

    @is_mlflow_up
    def set_tags(self, tags: dict):
        '''Log d'un ensemble de tags sur ML Flow

        Args:
            tags (dict): nom du tag
        Raises:
            TypeError : si l'objet tags n'est pas du type dict
        '''
        if type(tags) is not dict:
            raise TypeError('tags must be dict')

        # Log tags
        mlflow.set_tags(tags)

    def valid_name(self, key: str):
        '''Fonction to valid key names

        Args:
            key (str): key to check
        Returns:
            bool: whether key is a valid ML FLOW key
        '''
        return mlflow.mlflow.utils.validation._VALID_PARAM_AND_METRIC_NAMES.match(key)