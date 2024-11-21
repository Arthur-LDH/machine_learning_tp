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
from ynov.models_training.regressors.model_xgboost_regressor import ModelXgboostRegressor

# Disable logging
import logging
logging.disable(logging.CRITICAL)


def remove_dir(path):
    if os.path.isdir(path): shutil.rmtree(path)


class ModelXgboostRegressorTests(unittest.TestCase):
    '''Main class to test model_xgboost_regressor'''


    def setUp(self):
        '''SetUp fonction'''
        # On se place dans le bon répertoire
        # Change directory to script directory
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname)


    def test01_model_xgboost_regressor_init(self):
        '''Test de l'initialisation de ynov.models_training.model_xgboost_regressor.ModelXgboostRegressor'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Init., test all params
        model = ModelXgboostRegressor(model_dir=model_dir)
        self.assertEqual(model.model_dir, model_dir)
        self.assertTrue(os.path.isdir(model_dir))
        self.assertTrue(model.model is not None)
        self.assertEqual(model.model_type, 'regressor')
        # On fait juste un appel à display_if_gpu_activated et _is_gpu_activated
        self.assertTrue(type(model._is_gpu_activated()) == bool)
        model.display_if_gpu_activated()
        remove_dir(model_dir)

        #
        model = ModelXgboostRegressor(model_dir=model_dir, xgboost_params={'toto': 5})
        # Ajout auto objective quand non précisé
        self.assertEqual(model.xgboost_params, {'objective': 'reg:squarederror', 'toto': 5})
        remove_dir(model_dir)

        #
        model = ModelXgboostRegressor(model_dir=model_dir, early_stopping_rounds=8)
        self.assertEqual(model.early_stopping_rounds, 8)
        remove_dir(model_dir)

        #
        model = ModelXgboostRegressor(model_dir=model_dir, validation_split=0.3)
        self.assertEqual(model.validation_split, 0.3)
        remove_dir(model_dir)

    def test02_model_xgboost_regressor_fit(self):
        '''Test de la fonction fit de ynov.models_training.regressors.model_xgboost_regressor.ModelXgboostRegressor'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Set vars
        x_train = pd.DataFrame({'col_1': [-5, -1, 0, -2, 2, -6, 3] * 10, 'col_2': [2, -1, -8, 2, 3, 12, 2] * 10})
        y_train_regressor = pd.Series([-3, -2, -8, 0, 5, 6, 5] * 10)
        x_col = ['col_1', 'col_2']
        y_col_mono = ['toto']

        # Regression
        model = ModelXgboostRegressor(x_col=x_col, y_col=y_col_mono, model_dir=model_dir, xgboost_params={'n_estimators': 5})
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        self.assertEqual(model.x_col, x_col)
        self.assertEqual(model.y_col, y_col_mono)
        model.fit(x_train, y_train_regressor, x_valid=None, y_valid=None, with_shuffle=True)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(model.x_col, x_col)
        self.assertEqual(model.y_col, y_col_mono)
        remove_dir(model_dir)
        # avec valid
        model = ModelXgboostRegressor(x_col=x_col, y_col=y_col_mono, model_dir=model_dir, xgboost_params={'n_estimators': 5})
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        self.assertEqual(model.x_col, x_col)
        self.assertEqual(model.y_col, y_col_mono)
        model.fit(x_train, y_train_regressor, x_valid=x_train, y_valid=y_train_regressor, with_shuffle=True)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(model.x_col, x_col)
        self.assertEqual(model.y_col, y_col_mono)
        remove_dir(model_dir)
        # Avec shuffle à False
        model = ModelXgboostRegressor(x_col=x_col, y_col=y_col_mono, model_dir=model_dir, xgboost_params={'n_estimators': 5})
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        self.assertEqual(model.x_col, x_col)
        self.assertEqual(model.y_col, y_col_mono)
        model.fit(x_train, y_train_regressor, x_valid=x_train, y_valid=y_train_regressor, with_shuffle=False)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(model.x_col, x_col)
        self.assertEqual(model.y_col, y_col_mono)
        remove_dir(model_dir)

        #
        ############
        # Test continue training
        model = ModelXgboostRegressor(x_col=x_col, y_col=y_col_mono, model_dir=model_dir, xgboost_params={'n_estimators': 5})
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        model.fit(x_train, y_train_regressor)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        # second fit
        with self.assertRaises(RuntimeError):
            model.fit(x_train[:50], y_train_regressor[:50])
        self.assertEqual(model_dir, model.model_dir)
        remove_dir(model_dir)

    def test03_model_xgboost_regressor_predict(self):
        '''Test de la fonction predict de ynov.models_training.model_xgboost_regressor.ModelXgboostRegressor'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Set vars
        x_train = pd.DataFrame({'col_1': [-5, -1, 0, -2, 2, -6, 3] * 10, 'col_2': [2, -1, -8, 2, 3, 12, 2] * 10})
        y_train_regressor = pd.Series([-3, -2, -8, 0, 5, 6, 5] * 10)
        x_col = ['col_1', 'col_2']
        y_col_mono = ['toto']

        # Regressor
        model = ModelXgboostRegressor(x_col=x_col, y_col=y_col_mono, model_dir=model_dir)
        model.fit(x_train, y_train_regressor)
        preds = model.predict(x_train)
        self.assertEqual(preds.shape, (len(x_train),))
        remove_dir(model_dir)

        # Model needs to be fitted
        with self.assertRaises(AttributeError):
            model = ModelXgboostRegressor(x_col=x_col, y_col=y_col_mono, model_dir=model_dir)
            model.predict(pd.Series([-2, 3]))
        remove_dir(model_dir)

    def test04_model_xgboost_regressor_save(self):
        '''Test de la fonction save de ynov.models_training.model_xgboost_regressor.ModelXgboostRegressor'''
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Set vars
        x_train = pd.DataFrame({'col_1': [-5, -1, 0, -2, 2, -6, 3] * 10, 'col_2': [2, -1, -8, 2, 3, 12, 2] * 10})
        y_train_regressor = pd.Series([-3, -2, -8, 0, 5, 6, 5] * 10)
        x_col = ['col_1', 'col_2']
        y_col_mono = ['toto']


        # Fonctionnement nominal - sans fit
        model = ModelXgboostRegressor(x_col=x_col, y_col=y_col_mono, model_dir=model_dir, xgboost_params={'n_estimators': 5})
        model.save(json_data={'test': 8})
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, f'{model.model_name}.model')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, f"{model.model_name}.pkl")))
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
        self.assertEqual(configs['librairie'], 'xgboost')
        self.assertTrue('xgboost_params' in configs.keys())
        self.assertTrue('early_stopping_rounds' in configs.keys())
        self.assertTrue('validation_split' in configs.keys())
        remove_dir(model_dir)

        # Fonctionnement nominal - avec fit
        model = ModelXgboostRegressor(x_col=x_col, y_col=y_col_mono, model_dir=model_dir, xgboost_params={'n_estimators': 5})
        model.fit(x_train, y_train_regressor)
        model.save(json_data={'test': 8})
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, f'{model.model_name}.model')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, f"{model.model_name}.pkl")))
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
        self.assertEqual(configs['librairie'], 'xgboost')
        self.assertTrue('xgboost_params' in configs.keys())
        self.assertTrue('early_stopping_rounds' in configs.keys())
        self.assertTrue('validation_split' in configs.keys())
        remove_dir(model_dir)

        # AVEC level_save = 'LOW' & fit
        model = ModelXgboostRegressor(x_col=x_col, y_col=y_col_mono, model_dir=model_dir, xgboost_params={'n_estimators': 5}, level_save='LOW')
        model.fit(x_train, y_train_regressor)
        # Save
        model.save(json_data={'test': 8})
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, f'{model.model_name}.model')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, f"{model.model_name}.pkl")))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'preprocess_pipeline.pkl')))
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
        self.assertEqual(configs['librairie'], 'xgboost')
        self.assertTrue('xgboost_params' in configs.keys())
        self.assertTrue('early_stopping_rounds' in configs.keys())
        self.assertTrue('validation_split' in configs.keys())
        remove_dir(model_dir)

    def test05_model_xgboost_regressor_reload_from_standalone(self):
        '''Test de la fonction ynov.models_training.regressors.model_xgboost_regressor.ModelXgboostRegressor.reload_from_standalone'''

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
        model = ModelXgboostRegressor(x_col=x_col, y_col=y_col_mono, model_dir=model_dir, xgboost_params={'n_estimators': 5})
        model.fit(x_train, y_train_regressor)
        xgb_model = model.model
        model.save()
        # Reload
        xgboost_path = os.path.join(model.model_dir, f"{model.model_name}.model")
        conf_path = os.path.join(model.model_dir, "configurations.json")
        preprocess_pipeline_path = os.path.join(model.model_dir, "preprocess_pipeline.pkl")
        new_model = ModelXgboostRegressor()
        self.assertTrue(new_model.preprocess_pipeline is None)
        new_model.reload_from_standalone(configuration_path=conf_path, xgboost_path=xgboost_path,
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
        self.assertEqual(model.xgboost_params, new_model.xgboost_params)
        self.assertEqual(model.early_stopping_rounds, new_model.early_stopping_rounds)
        self.assertEqual(model.validation_split, new_model.validation_split)
        self.assertEqual(model.model.get_params(), xgb_model.get_params())
        self.assertTrue(new_model.preprocess_pipeline is not None)
        # On ne peut pas vraiment tester la pipeline, du coup on test les predictions
        self.assertEqual([[_] for _ in model.predict(x_train)], [[_] for _ in new_model.predict(x_train)])
        remove_dir(new_model.model_dir)
        # On ne remove pas model_dir pour tester les erreurs

        ############################################
        # Errors
        ############################################

        with self.assertRaises(FileNotFoundError):
            new_model = ModelXgboostRegressor()
            new_model.reload_from_standalone(configuration_path='toto.json', xgboost_path=xgboost_path,
                                             preprocess_pipeline_path=preprocess_pipeline_path)
        with self.assertRaises(FileNotFoundError):
            new_model = ModelXgboostRegressor()
            new_model.reload_from_standalone(configuration_path=conf_path, xgboost_path='toto.pkl',
                                             preprocess_pipeline_path=preprocess_pipeline_path)
        with self.assertRaises(FileNotFoundError):
            new_model = ModelXgboostRegressor()
            new_model.reload_from_standalone(configuration_path=conf_path, xgboost_path=xgboost_path,
                                             preprocess_pipeline_path='toto.pkl')

        # Clean
        remove_dir(model_dir)


# Execution des tests
if __name__ == '__main__':
    # Start tests
    unittest.main()