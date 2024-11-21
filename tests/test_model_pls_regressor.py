#!/usr/bin/env python3

# Libs unittest
import unittest
from unittest.mock import patch
from unittest.mock import Mock

# Utils libs
import os
import json
import shutil
import pandas as pd
import numpy as np
from ynov import utils
from ynov.models_training.regressors.model_pls_regressor import ModelPLSRegressor

# Disable logging
import logging
logging.disable(logging.CRITICAL)


def remove_dir(path):
    if os.path.isdir(path): shutil.rmtree(path)


class ModelPLSRegressorTests(unittest.TestCase):
    '''Main class to test model_pls_regressor'''


    def setUp(self):
        '''SetUp fonction'''
        # On se place dans le bon répertoire
        # Change directory to script directory
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname)


    def test01_model_pls_regressor_init(self):
        '''Test de l'initialisation de ynov.models_training.model_pls_regressor.ModelPLSRegressor'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Init., test all params
        model = ModelPLSRegressor(model_dir=model_dir)
        self.assertEqual(model.model_dir, model_dir)
        self.assertTrue(os.path.isdir(model_dir))
        self.assertTrue(model.pipeline is not None)
        self.assertEqual(model.model_type, 'regressor')
        # On fait juste un appel à display_if_gpu_activated et _is_gpu_activated
        self.assertTrue(type(model._is_gpu_activated()) == bool)
        model.display_if_gpu_activated()
        remove_dir(model_dir)

        # Check PLS params
        model = ModelPLSRegressor(model_dir=model_dir, pls_params={'n_components': 5, 'max_iter': 500})
        self.assertEqual(model.pipeline['pls'].n_components, 5)
        self.assertEqual(model.pipeline['pls'].max_iter, 500)
        remove_dir(model_dir)

    def test02_model_pls_regressor_predict(self):
        '''Test de la fonction predict de ynov.models_training.model_pls_regressor.ModelPLSRegressor'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Set vars
        x_train = pd.DataFrame({'col_1': [-5, -1, 0, -2, 2, -6, 3] * 10, 'col_2': [2, -1, -8, 2, 3, 12, 2] * 10})
        y_train_regressor = pd.Series([-3, -2, -8, 0, 5, 6, 5] * 10)
        x_col = ['col_1', 'col_2']
        y_col_mono = ['toto']

        # Regressor
        model = ModelPLSRegressor(x_col=x_col, y_col=y_col_mono, model_dir=model_dir)
        model.fit(x_train, y_train_regressor)
        preds = model.predict(x_train, return_proba=False)
        self.assertEqual(preds.shape, (len(x_train),))
        with self.assertRaises(ValueError):
            proba = model.predict(x_train, return_proba=True)
        remove_dir(model_dir)

        # Model needs to be fitted
        with self.assertRaises(AttributeError):
            model = ModelPLSRegressor(x_col=x_col, y_col=y_col_mono, model_dir=model_dir)
            model.predict(pd.Series([-2, 3]))
        remove_dir(model_dir)

    def test03_model_pls_regressor_save(self):
        '''Test de la fonction save de ynov.models_training.model_pls_regressor.ModelPLSRegressor'''
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Fonctionnement nominal
        model = ModelPLSRegressor(model_dir=model_dir)
        model.save(json_data={'test': 8})
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, f"{model.model_name}.pkl")))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, f"{model.model_name}_standalone.pkl")))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'preprocess_pipeline.pkl')))
        with open(os.path.join(model.model_dir, 'configurations.json'), 'r', encoding='utf-8') as f:
            configs = json.load(f)
        self.assertEqual(configs['test'], 8)
        self.assertTrue('mainteners' in configs.keys())
        self.assertTrue('date' in configs.keys())
        self.assertTrue('package_version' in configs.keys())
        self.assertEqual(configs['package_version'], utils.get_package_version())
        self.assertTrue('model_name' in configs.keys())
        self.assertTrue('model_dir' in configs.keys())
        self.assertTrue('model_type' in configs.keys())
        self.assertEqual(configs['model_type'], 'regressor')
        self.assertTrue('trained' in configs.keys())
        self.assertTrue('nb_fit' in configs.keys())
        self.assertTrue('x_col' in configs.keys())
        self.assertTrue('y_col' in configs.keys())
        self.assertTrue('columns_in' in configs.keys())
        self.assertTrue('mandatory_columns' in configs.keys())
        self.assertTrue('level_save' in configs.keys())
        self.assertTrue('librairie' in configs.keys())
        self.assertEqual(configs['librairie'], 'scikit-learn')
        # Spécifique model utilisé
        self.assertTrue('pls_confs' in configs.keys())
        remove_dir(model_dir)

    def test04_model_pls_regressor_reload_from_standalone(self):
        '''Test de la fonction ynov.models_training.regressors.model_pls_regressor.ModelPLSRegressor.reload_from_standalone'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Set vars
        x_train = pd.DataFrame({'col_1': [-5, -1, 0, -2, 2, -6, 3] * 10, 'col_2': [2, -1, -8, 2, 3, 12, 2] * 10})
        y_train_regressor = pd.Series([-3, -2, -8, 0, 5, 6, 5] * 10)
        x_col = ['col_1', 'col_2']
        y_col_mono = ['toto']

        ############################################
        # Regression
        ############################################

        # Create model
        model = ModelPLSRegressor(x_col=x_col, y_col=y_col_mono, model_dir=model_dir)
        pls = model.pls
        model.fit(x_train, y_train_regressor)
        model.save()
        # Reload
        pkl_path = os.path.join(model.model_dir, f"{model.model_name}_standalone.pkl")
        conf_path = os.path.join(model.model_dir, "configurations.json")
        preprocess_pipeline_path = os.path.join(model.model_dir, "preprocess_pipeline.pkl")
        new_model = ModelPLSRegressor()
        self.assertTrue(new_model.preprocess_pipeline is None)
        new_model.reload_from_standalone(configuration_path=conf_path, model_pipeline_path=pkl_path,
                                         preprocess_pipeline_path=preprocess_pipeline_path)
        # Tests
        self.assertEqual(model.model_name, new_model.model_name)
        self.assertEqual(model.model_type, new_model.model_type)
        self.assertEqual(model.trained, new_model.trained)
        self.assertEqual(model.nb_fit, new_model.nb_fit)
        self.assertEqual(model.x_col, new_model.x_col)
        self.assertEqual(model.y_col, new_model.y_col)
        self.assertEqual(model.columns_in, new_model.columns_in)
        self.assertEqual(model.mandatory_columns, new_model.mandatory_columns)
        self.assertEqual(model.level_save, new_model.level_save)
        self.assertEqual(model.pls.get_params(), pls.get_params())
        self.assertTrue(new_model.preprocess_pipeline is not None)
        # On ne peut pas vraiment tester la pipeline, du coup on test les predictions
        self.assertEqual([[_] for _ in model.predict(x_train)], [[_] for _ in new_model.predict(x_train)])
        remove_dir(new_model.model_dir)
        # On ne remove pas model_dir pour tester les erreurs

        ############################################
        # Errors
        ############################################

        with self.assertRaises(FileNotFoundError):
            new_model = ModelPLSRegressor()
            new_model.reload_from_standalone(configuration_path='toto.json', model_pipeline_path=pkl_path,
                                             preprocess_pipeline_path=preprocess_pipeline_path)
        with self.assertRaises(FileNotFoundError):
            new_model = ModelPLSRegressor()
            new_model.reload_from_standalone(configuration_path=conf_path, model_pipeline_path='toto.pkl',
                                             preprocess_pipeline_path=preprocess_pipeline_path)
        with self.assertRaises(FileNotFoundError):
            new_model = ModelPLSRegressor()
            new_model.reload_from_standalone(configuration_path=conf_path, model_pipeline_path=pkl_path,
                                             preprocess_pipeline_path='toto.pkl')

        # Clean
        remove_dir(model_dir)


# Execution des tests
if __name__ == '__main__':
    # Start tests
    unittest.main()