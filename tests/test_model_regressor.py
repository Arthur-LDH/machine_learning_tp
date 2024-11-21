#!/usr/bin/env python3

# Libs unittest
import unittest
from unittest.mock import patch
from unittest.mock import Mock

# Utils libs
import os
import json
import math
import shutil
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from ynov import utils
from ynov.preprocessing import preprocess
from ynov.monitoring.model_logger import ModelLogger
from ynov.models_training.model_class import ModelClass
from ynov.models_training.model_regressor import ModelRegressorMixin

# Disable logging
import logging
logging.disable(logging.CRITICAL)


def remove_dir(path):
    if os.path.isdir(path): shutil.rmtree(path)


class ModelMockRegressor(ModelRegressorMixin, ModelClass):
    '''We need a mock implementation of the Mixin class'''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pipeline = Pipeline([('rf', RandomForestRegressor())])
    def fit(self, x_train, y_train, **kwargs):
        '''Version simplifiée fit'''
        x_train, y_train = self._check_input_format(x_train, y_train, fit_function=True)
        self.pipeline.fit(x_train, y_train)
        self.trained = True
        self.nb_fit += 1
    def predict(self, x_test: pd.DataFrame, return_proba: bool = False, **kwargs):
        '''Version simplifiée predict'''
        x_test, _ = self._check_input_format(x_test)
        if not return_proba:
            return np.array(self.pipeline.predict(x_test))
        else:
            return self.predict_proba(x_test)
    def predict_proba(self, x_test: pd.DataFrame, **kwargs):
        '''Version simplifiée predict_proba'''
        if self.model_type != 'classifier':
            raise ValueError()
        pass


class ModelRegressorMixinTests(unittest.TestCase):
    '''Main class to test model_regressor'''


    def setUp(self):
        '''SetUp fonction'''
        # On se place dans le bon répertoire
        # Change directory to script directory
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname)


    def test01_model_regressor_init(self):
        '''Test de l'initialisation de ynov.models_training.model_regressor.ModelRegressorMixin'''

        # Création d'un modèle
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        model_name = 'test'
        x_col = ['test_x1', 'test_x2']
        y_col = 'test_y'

        # Fonctionnement nominal
        model = ModelMockRegressor(model_dir=model_dir, model_name=model_name)
        self.assertEqual(model.model_dir, model_dir)
        self.assertTrue(os.path.isdir(model_dir))
        remove_dir(model_dir)

        # Test level_save
        model = ModelMockRegressor(model_dir=model_dir, model_name=model_name, level_save='HIGH')
        self.assertEqual(model.level_save, 'HIGH')
        remove_dir(model_dir)
        #
        model = ModelMockRegressor(model_dir=model_dir, model_name=model_name, level_save='MEDIUM')
        self.assertEqual(model.level_save, 'MEDIUM')
        remove_dir(model_dir)
        #
        model = ModelMockRegressor(model_dir=model_dir, model_name=model_name, level_save='LOW')
        self.assertEqual(model.level_save, 'LOW')
        remove_dir(model_dir)

        # Gestion errors
        with self.assertRaises(ValueError):
            ModelMockRegressor(model_dir=model_dir, model_name=model_name, level_save='toto')
        remove_dir(model_dir)


    def test02_model_regressor_inverse_transform(self):
        '''Test de la fonction ynov.models_training.model_regressor.ModelRegressorMixin.inverse_transform'''

        # Création d'un modèle
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        model_name = 'test'

        # inverse_transform - mono-label
        model = ModelMockRegressor(model_dir=model_dir, model_name=model_name)
        y1 = np.array(['toto', 'titi', 'tata', 'toto'])
        expected_result1 = ['toto', 'titi', 'tata', 'toto']
        y2 = 'toto'
        expected_result2 = 'toto'
        self.assertEqual(model.inverse_transform(y1), expected_result1)
        self.assertEqual(model.inverse_transform(y2), expected_result2)
        remove_dir(model_dir)


    def test03_model_regressor_get_and_save_metrics(self):
        '''Test de la fonction ynov.models_training.model_regressor.ModelRegressorMixin.get_and_save_metrics'''

        # Création d'un modèle
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        model_name = 'test'

        # get_and_save_metrics - mono-label
        model = ModelMockRegressor(model_dir=model_dir, model_name=model_name)
        y_true = np.array([0.12, 1.5, -1.6, 12.1])
        y_pred = np.array([0.32, 1.4, -1.3, 11.9])
        df_metrics = model.get_and_save_metrics(y_true, y_pred)
        self.assertEqual(df_metrics.shape[0], 1) # All
        self.assertEqual(df_metrics.loc[0, :]['Label'], 'All')
        self.assertAlmostEqual(df_metrics.loc[0, :]['MAE'], 0.2)
        mse = (0.1*0.1 + 0.3*0.3 + 0.2*0.2 + 0.2*0.2)/4
        self.assertAlmostEqual(df_metrics.loc[0, :]['MSE'], mse)
        self.assertAlmostEqual(df_metrics.loc[0, :]['RMSE'], math.sqrt(mse))
        self.assertTrue(df_metrics.loc[0, :]['Explained variance'] > 0.99)
        self.assertTrue(df_metrics.loc[0, :]['Coefficient of determination'] > 0.99)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'predictions.csv')))
        plots_path = os.path.join(model.model_dir, 'plots')
        self.assertTrue(os.path.exists(os.path.join(plots_path, 'errors.png')))
        self.assertTrue(os.path.exists(os.path.join(plots_path, 'residuals.png')))
        remove_dir(model_dir)
        #
        # get_and_save_metrics, avec les autres paramètres
        model = ModelMockRegressor(model_dir=model_dir, model_name=model_name)
        y_true = np.array([0.12, 1.5, -1.6, 12.1])
        y_pred = np.array([0.32, 1.4, -1.3, 11.9])
        df_x = pd.DataFrame({'col_1': [-5, -1, 0, 2], 'col_2': [2, -1, -8, 3]})
        series_to_add = [pd.Series(['a', 'b', 'c', 'd'], name='test')]
        type_data = 'toto'
        model_logger = ModelLogger(
            tracking_uri="http://toto.titi.tata.test",
            experiment_name="test"
        )
        df_metrics = model.get_and_save_metrics(y_true, y_pred, df_x=df_x, series_to_add=series_to_add, type_data=type_data, model_logger=model_logger)
        self.assertEqual(df_metrics.shape[0], 1) # All
        self.assertEqual(df_metrics.loc[0, :]['Label'], 'All')
        self.assertAlmostEqual(df_metrics.loc[0, :]['MAE'], 0.2)
        mse = (0.1*0.1 + 0.3*0.3 + 0.2*0.2 + 0.2*0.2)/4
        self.assertAlmostEqual(df_metrics.loc[0, :]['MSE'], mse)
        self.assertAlmostEqual(df_metrics.loc[0, :]['RMSE'], math.sqrt(mse))
        self.assertTrue(df_metrics.loc[0, :]['Explained variance'] > 0.99)
        self.assertTrue(df_metrics.loc[0, :]['Coefficient of determination'] > 0.99)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, f'predictions_{type_data}.csv')))
        plots_path = os.path.join(model.model_dir, 'plots')
        self.assertTrue(os.path.exists(os.path.join(plots_path, f'{type_data}_errors.png')))
        self.assertTrue(os.path.exists(os.path.join(plots_path, f'{type_data}_residuals.png')))
        df_preds = pd.read_csv(os.path.join(model.model_dir, f'predictions_{type_data}.csv'), sep=',', encoding='utf-8')
        self.assertTrue('col_1' in df_preds.columns)
        self.assertTrue('col_2' in df_preds.columns)
        self.assertTrue('y_true' in df_preds.columns)
        self.assertTrue('y_pred' in df_preds.columns)
        self.assertTrue('abs_err' in df_preds.columns)
        self.assertTrue('rel_err' in df_preds.columns)
        self.assertTrue('test' in df_preds.columns)
        remove_dir(model_dir)


    def test04_model_regressor_get_metrics_simple(self):
        '''Test de la fonction ynov.models_training.model_regressor.ModelRegressorMixin.get_metrics_simple_monolabel'''

        # Création d'un modèle
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        model_name = 'test'

        # get_and_save_metrics - mono-label
        model = ModelMockRegressor(model_dir=model_dir, model_name=model_name)
        y_true = np.array([0.12, 1.5, -1.6, 12.1])
        y_pred = np.array([0.32, 1.4, -1.3, 11.9])
        df_metrics = model.get_metrics_simple(y_true, y_pred)
        self.assertEqual(df_metrics.shape[0], 1) # All
        self.assertEqual(df_metrics.loc[0, :]['Label'], 'All')
        self.assertAlmostEqual(df_metrics.loc[0, :]['MAE'], 0.2)
        mse = (0.1*0.1 + 0.3*0.3 + 0.2*0.2 + 0.2*0.2)/4
        self.assertAlmostEqual(df_metrics.loc[0, :]['MSE'], mse)
        self.assertAlmostEqual(df_metrics.loc[0, :]['RMSE'], math.sqrt(mse))
        self.assertTrue(df_metrics.loc[0, :]['Explained variance'] > 0.99)
        self.assertTrue(df_metrics.loc[0, :]['Coefficient of determination'] > 0.99)
        remove_dir(model_dir)


    def test05_model_regressor_save(self):
        '''Test de la fonction ynov.models_training.model_regressor.ModelRegressorMixin.save'''

        # Création d'un modèle
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        model_name = 'test'
        preprocess_pipeline = preprocess.get_pipeline("no_preprocess") # Attention, besoin d'être fit
        preprocess_pipeline.fit(pd.DataFrame({'test_x1': [1, 2, 3], 'test_x2': [4, 5, 6]}))

        # test save
        model = ModelMockRegressor(model_dir=model_dir, model_name=model_name, preprocess_pipeline=preprocess_pipeline)
        pkl_path = os.path.join(model.model_dir, f"{model.model_name}.pkl")
        configuration_path = os.path.join(model.model_dir, 'configurations.json')
        preprocess_pipeline_path = os.path.join(model.model_dir, 'preprocess_pipeline.pkl')
        model.save(json_data={'test': 8})
        self.assertTrue(os.path.exists(pkl_path))
        self.assertTrue(os.path.exists(configuration_path))
        self.assertTrue(os.path.exists(preprocess_pipeline_path))
        with open(configuration_path, 'r', encoding='utf-8') as f:
            configs = json.load(f)
        self.assertEqual(configs['test'], 8)
        self.assertTrue('mainteners' in configs.keys())
        self.assertTrue('date' in configs.keys())
        self.assertTrue('package_version' in configs.keys())
        self.assertEqual(configs['package_version'], utils.get_package_version())
        self.assertTrue('model_name' in configs.keys())
        self.assertTrue('model_dir' in configs.keys())
        self.assertTrue('model_type' in configs.keys())
        self.assertTrue('trained' in configs.keys())
        self.assertTrue('nb_fit' in configs.keys())
        self.assertTrue('x_col' in configs.keys())
        self.assertTrue('y_col' in configs.keys())
        self.assertTrue('columns_in' in configs.keys())
        self.assertTrue('mandatory_columns' in configs.keys())
        self.assertTrue('level_save' in configs.keys())
        self.assertTrue('librairie' in configs.keys())
        self.assertEqual(configs['librairie'], None)
        remove_dir(model_dir)

        # test save, level_save = 'LOW'
        model = ModelMockRegressor(model_dir=model_dir, model_name=model_name, preprocess_pipeline=preprocess_pipeline, level_save='LOW')
        pkl_path = os.path.join(model.model_dir, f"{model.model_name}.pkl")
        configuration_path = os.path.join(model.model_dir, 'configurations.json')
        preprocess_pipeline_path = os.path.join(model.model_dir, 'preprocess_pipeline.pkl')
        model.save(json_data={'test': 8})
        self.assertFalse(os.path.exists(pkl_path))
        self.assertTrue(os.path.exists(configuration_path))
        self.assertFalse(os.path.exists(preprocess_pipeline_path))
        with open(configuration_path, 'r', encoding='utf-8') as f:
            configs = json.load(f)
        self.assertEqual(configs['test'], 8)
        remove_dir(model_dir)


# Execution des tests
if __name__ == '__main__':
    # Start tests
    unittest.main()