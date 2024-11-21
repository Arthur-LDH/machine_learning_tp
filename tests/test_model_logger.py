#!/usr/bin/env python3

# Libs unittest
import unittest
from unittest.mock import patch
from unittest.mock import Mock

# Utils libs
import os
import json
import shutil
import mlflow
import pandas as pd
import numpy as np
from ynov import utils
from ynov.monitoring.model_logger import ModelLogger, is_running, is_local, is_mlflow_up

# Disable logging
import logging
logging.disable(logging.CRITICAL)


class ModelLoggerTests(unittest.TestCase):
    '''Main class to test model_logger'''


    def setUp(self):
        '''SetUp fonction'''
        # On se place dans le bon répertoire
        # Change directory to script directory
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname)


    def test01_is_running(self):
        '''Test de la fonction ynov.monitoring.model_logger.is_running'''
        save_dir = os.path.join(os.getcwd(), 'ml_flow_test')
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        model = ModelLogger(os.path.relpath(save_dir)) # on veut juste le logger
        bad_host = 'http://toto.titi.tata.test'
        bad_port = 80
        self.assertFalse(is_running(bad_host, bad_port, model.logger))
        # Clear
        model.stop_run()
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        # TODO: concourse n'a pas accès à google
        # Il faudrait trouver une adresse valide quoiqu'il arrive...
        # correct_host = 'http://www.google.com'
        # correct_port = 80
        # self.assertTrue(is_running(correct_host, correct_port, model.logger))

    def test02_is_local(self):
        '''Test de la fonction ynov.monitoring.model_logger.is_local'''
        local = 'ceci/est/un/test'
        distant = 'http://ceci.est.un.faux.site.com'
        self.assertTrue(is_local(local))
        self.assertFalse(is_local(distant))

    def test03_model_logger_init(self):
        '''Test de l'initialisation de ynov.monitoring.model_logger.ModelLogger'''
        # On test avec un host bidon
        host = 'http://toto.titi.tata.test'
        name = 'test'
        model = ModelLogger(tracking_uri=host, experiment_name=name)
        self.assertEqual(model.tracking_uri, host)
        self.assertEqual(model.experiment_name, name)
        self.assertFalse(model.running)
        # Clear
        model.stop_run()

        # On test avec un host existant (mais pas ml flow dessus)
        # TODO: concourse n'a pas accès à google
        # Il faudrait trouver une adresse valide quoiqu'il arrive...
        # host = 'http://www.google.com'
        # name = 'test'
        # model = ModelLogger(tracking_uri=host, experiment_name=name)
        # self.assertEqual(model.tracking_uri, host)
        # self.assertEqual(model.experiment_name, name)
        # self.assertFalse(model.running)

        # On test sans rien (par défaut -> mlflow save dans un dossier 'ml_runs')
        save_dir = os.path.join(os.getcwd(), 'mlruns')
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        self.assertFalse(os.path.exists(save_dir))
        model = ModelLogger()
        self.assertEqual(model.tracking_uri, '')
        self.assertTrue(model.running)
        self.assertTrue(os.path.exists(save_dir))
        # Clear
        model.stop_run()
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)

        # On aimerait bien testé avec un serveur mlflow, mais pas de nom fix

    def test04_model_logger_stop_run(self):
        '''Test de ynov.monitoring.model_logger.ModelLogger.stop_run'''
        # Init. logger
        save_dir = os.path.join(os.getcwd(), 'ml_flow_test')
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        model = ModelLogger(os.path.relpath(save_dir))
        # On active un run via un log
        model.log_param('test', 'toto')
        # Utilisation stop_run
        model.stop_run()
        # Check
        self.assertEqual(mlflow.active_run(), None)
        # Clear
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)

    def test05_model_logger_log_metric(self):
        '''Test de ynov.monitoring.model_logger.ModelLogger.log_metric'''
        # Init. logger
        save_dir = os.path.join(os.getcwd(), 'ml_flow_test')
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        model = ModelLogger(os.path.relpath(save_dir))

        # Fonctionnement nominal
        model.log_metric('test', 5)
        model.log_metric('test', 5, step=2)

        # Check errors
        # wrapped_fn -> avoid wrapper
        with self.assertRaises(TypeError):
            model.log_metric.wrapped_fn(model, 2, 5)
        with self.assertRaises(TypeError):
            model.log_metric.wrapped_fn(model, 'test', 5, step='toto')

        # Clear
        model.stop_run()
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)

    def test06_model_logger_log_metrics(self):
        '''Test de ynov.monitoring.model_logger.ModelLogger.log_metrics'''
        # Init. logger
        save_dir = os.path.join(os.getcwd(), 'ml_flow_test')
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        model = ModelLogger(os.path.relpath(save_dir))

        # Fonctionnement nominal
        model.log_metrics({'test': 5})
        model.log_metrics({'test': 5}, step=2)

        # Check errors
        # wrapped_fn -> avoid wrapper
        with self.assertRaises(TypeError):
            model.log_metrics.wrapped_fn(model, 'toto')
        with self.assertRaises(TypeError):
            model.log_metrics.wrapped_fn(model, {'test': 5}, step='toto')

        # Clear
        model.stop_run()
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)

    def test07_model_logger_log_param(self):
        '''Test de ynov.monitoring.model_logger.ModelLogger.log_param'''
        # Init. logger
        save_dir = os.path.join(os.getcwd(), 'ml_flow_test')
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        model = ModelLogger(os.path.relpath(save_dir))

        # Fonctionnement nominal
        model.log_param('test', 5)

        # Check errors
        # wrapped_fn -> avoid wrapper
        with self.assertRaises(TypeError):
            model.log_param.wrapped_fn(model, 2, 5)

        # Clear
        model.stop_run()
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)

    def test08_model_logger_log_params(self):
        '''Test de ynov.monitoring.model_logger.ModelLogger.log_params'''
        # Init. logger
        save_dir = os.path.join(os.getcwd(), 'ml_flow_test')
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        model = ModelLogger(os.path.relpath(save_dir))

        # Fonctionnement nominal
        model.log_params({'test': 5})

        # Check errors
        # wrapped_fn -> avoid wrapper
        with self.assertRaises(TypeError):
            model.log_params.wrapped_fn(model, 'toto')

        # Clear
        model.stop_run()
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)

    def test09_model_logger_set_tag(self):
        '''Test de ynov.monitoring.model_logger.ModelLogger.set_tag'''
        # Init. logger
        save_dir = os.path.join(os.getcwd(), 'ml_flow_test')
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        model = ModelLogger(os.path.relpath(save_dir))

        # Fonctionnement nominal
        model.set_tag('test', 5)

        # Check errors
        # wrapped_fn -> avoid wrapper
        with self.assertRaises(TypeError):
            model.set_tag.wrapped_fn(model, 2, 5)

        # Clear
        model.stop_run()
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)

    def test10_model_logger_set_tags(self):
        '''Test de ynov.monitoring.model_logger.ModelLogger.set_tags'''
        # Init. logger
        save_dir = os.path.join(os.getcwd(), 'ml_flow_test')
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        model = ModelLogger(os.path.relpath(save_dir))

        # Fonctionnement nominal
        model.set_tags({'test': 5})

        # Check errors
        # wrapped_fn -> avoid wrapper
        with self.assertRaises(TypeError):
            model.set_tags.wrapped_fn(model, 'toto')

        # Clear
        model.stop_run()
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)


# Execution des tests
if __name__ == '__main__':
    # Start tests
    unittest.main()