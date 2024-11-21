#!/usr/bin/env python3

# Libs unittest
import unittest
from unittest.mock import patch
from unittest.mock import Mock

# Utils libs
import os
import json
import shutil
import tensorflow
import numpy as np
import pandas as pd
from ynov import utils
from ynov.models_training import utils_deep_keras
from ynov.models_training.regressors.model_dense_regressor import ModelDenseRegressor

# Disable logging
import logging
logging.disable(logging.CRITICAL)


def remove_dir(path):
    if os.path.isdir(path): shutil.rmtree(path)


class ModelDenseRegressorTests(unittest.TestCase):
    '''Main class to test model_dense_regressor'''


    def setUp(self):
        '''SetUp fonction'''
        # On se place dans le bon répertoire
        # Change directory to script directory
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname)


    def test01_model_dense_regressor_init(self):
        '''Test de l'initialisation de ynov.models_training.model_dense_regressor.ModelDenseRegressor'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Init., test all params
        model = ModelDenseRegressor(model_dir=model_dir)
        self.assertEqual(model.model_dir, model_dir)
        self.assertTrue(os.path.isdir(model_dir))
        self.assertEqual(model.model_type, 'regressor')
        self.assertEqual(model.model, None)
        # On fait juste un appel à display_if_gpu_activated et _is_gpu_activated
        self.assertTrue(type(model._is_gpu_activated()) == bool)
        model.display_if_gpu_activated()
        remove_dir(model_dir)

        #
        model = ModelDenseRegressor(model_dir=model_dir, batch_size=8)
        self.assertEqual(model.batch_size, 8)
        remove_dir(model_dir)

        #
        model = ModelDenseRegressor(model_dir=model_dir, epochs=42)
        self.assertEqual(model.epochs, 42)
        remove_dir(model_dir)

        #
        model = ModelDenseRegressor(model_dir=model_dir, validation_split=0.3)
        self.assertEqual(model.validation_split, 0.3)
        remove_dir(model_dir)

        #
        model = ModelDenseRegressor(model_dir=model_dir, patience=65)
        self.assertEqual(model.patience, 65)
        remove_dir(model_dir)

        #
        model = ModelDenseRegressor(model_dir=model_dir, nb_iter_keras=2)
        self.assertEqual(model.nb_iter_keras, 2)
        remove_dir(model_dir)

        # keras_params doit marcher avec n'importe quoi !
        model = ModelDenseRegressor(model_dir=model_dir, keras_params={'toto': 5})
        self.assertEqual(model.keras_params, {'toto': 5})
        remove_dir(model_dir)

    def test02_model_dense_regressor_predict(self):
        '''Test de la fonction predict de ynov.models_training.model_dense_regressor.ModelDenseRegressor'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Set vars
        x_train = pd.DataFrame({'col_1': [-5, -1, 0, -2, 2, -6, 3] * 10, 'col_2': [2, -1, -8, 2, 3, 12, 2] * 10})
        y_train_regressor = pd.Series([-3, -2, -8, 0, 5, 6, 5] * 10)
        x_col = ['col_1', 'col_2']
        y_col_mono = ['toto']

        # Regressor
        model = ModelDenseRegressor(x_col=x_col, y_col=y_col_mono, model_dir=model_dir, batch_size=8, epochs=2)
        model.fit(x_train, y_train_regressor)
        preds = model.predict(x_train, return_proba=False)
        self.assertEqual(preds.shape, (len(x_train),))
        with self.assertRaises(ValueError):
            proba = model.predict(x_train, return_proba=True)
        preds = model.predict(x_train, return_proba=False, experimental_version=True)
        self.assertEqual(preds.shape, (len(x_train),))
        with self.assertRaises(ValueError):
            proba = model.predict(x_train, return_proba=True, experimental_version=True)
        remove_dir(model_dir)
        # nb iter > 0
        model = ModelDenseRegressor(x_col=x_col, y_col=y_col_mono, model_dir=model_dir, batch_size=8, epochs=2, nb_iter_keras=3)
        model.fit(x_train, y_train_regressor)
        preds = model.predict(x_train, return_proba=False)
        self.assertEqual(preds.shape, (len(x_train),))
        with self.assertRaises(ValueError):
            proba = model.predict(x_train, return_proba=True)
        preds = model.predict(x_train, return_proba=False, experimental_version=True)
        self.assertEqual(preds.shape, (len(x_train),))
        with self.assertRaises(ValueError):
            proba = model.predict(x_train, return_proba=True, experimental_version=True)
        remove_dir(model_dir)

        # Model needs to be fitted
        with self.assertRaises(AttributeError):
            model = ModelDenseRegressor(x_col=x_col, y_col=y_col_mono, model_dir=model_dir, batch_size=8, epochs=2)
            model.predict(x_train)
        remove_dir(model_dir)

    def test03_model_dense_regressor_get_model(self):
        '''Test de la fonction ynov.models_training.regressors.model_dense_regressor.ModelDenseRegressor._get_model'''

        # Set vars
        x_col = ['col_1', 'col_2']
        y_col_mono = ['toto']

        # Create model
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        model = ModelDenseRegressor(x_col=x_col, y_col=y_col_mono, model_dir=model_dir, epochs=2)

        # Fonctionnement nominal
        model_res = model._get_model()
        self.assertTrue(type(model_res) in [tensorflow.python.keras.engine.sequential.Sequential, tensorflow.python.keras.engine.functional.Functional])

        # Clean
        remove_dir(model_dir)

    def test04_model_dense_regressor_save(self):
        '''Test de la fonction save de ynov.models_training.model_dense_regressor.ModelDenseRegressor'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Set vars
        x_col = ['col_1', 'col_2']
        y_col_mono = ['toto']

        # Fonctionnement nominal
        model = ModelDenseRegressor(x_col=x_col, y_col=y_col_mono, model_dir=model_dir, epochs=2)
        model.save(json_data={'test': 8})
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        # self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5'))) -> pas de modèle trained
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
        self.assertEqual(configs['librairie'], 'keras')
        self.assertTrue('batch_size' in configs.keys())
        self.assertTrue('epochs' in configs.keys())
        self.assertTrue('validation_split' in configs.keys())
        self.assertTrue('patience' in configs.keys())
        self.assertTrue('nb_iter_keras' in configs.keys())
        self.assertTrue('keras_params' in configs.keys())
        self.assertTrue('_get_model' in configs.keys())
        self.assertTrue('_get_learning_rate_scheduler' in configs.keys())
        self.assertTrue('custom_objects' in configs.keys())
        remove_dir(model_dir)

        # Fonctionnement avec custom_objects qui contient une fonction "partial"
        model = ModelDenseRegressor(x_col=x_col, y_col=y_col_mono, model_dir=model_dir, epochs=2)
        custom_objects = utils_deep_keras.custom_objects
        custom_objects['fb_loss'] = utils_deep_keras.get_fb_loss(0.5)
        model.custom_objects = custom_objects
        model.save(json_data={'test': 8})
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        # self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5'))) -> pas de modèle trained
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
        self.assertEqual(configs['librairie'], 'keras')
        self.assertTrue('batch_size' in configs.keys())
        self.assertTrue('epochs' in configs.keys())
        self.assertTrue('validation_split' in configs.keys())
        self.assertTrue('patience' in configs.keys())
        self.assertTrue('nb_iter_keras' in configs.keys())
        self.assertTrue('keras_params' in configs.keys())
        self.assertTrue('_get_model' in configs.keys())
        self.assertTrue('_get_learning_rate_scheduler' in configs.keys())
        self.assertTrue('custom_objects' in configs.keys())
        remove_dir(model_dir)

    def test05_model_dense_regressor_reload_model(self):
        '''Test de la fonction reload_model de ynov.models_training.regressors.model_dense_regressor.ModelDenseRegressor'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Set vars
        x_train = pd.DataFrame({'col_1': [-5, -1, 0, -2, 2, -6, 3] * 10, 'col_2': [2, -1, -8, 2, 3, 12, 2] * 10})
        y_train_regressor = pd.Series([-3, -2, -8, 0, 5, 6, 5] * 10)
        x_col = ['col_1', 'col_2']
        y_col_mono = ['toto']

        # Regression
        model = ModelDenseRegressor(model_dir=model_dir, batch_size=8, epochs=2)
        model.fit(x_train, y_train_regressor)
        model.save()
        # Reload keras
        hdf5_path = os.path.join(model.model_dir, 'best.hdf5')
        reloaded_model = model.reload_model(hdf5_path)
        np.testing.assert_almost_equal([list(_) for _ in reloaded_model.predict(x_train)], [list(_) for _ in model.model.predict(x_train)], 5)
        # Test sans custom_objects
        model.custom_objects = None
        reloaded_model = model.reload_model(hdf5_path)
        np.testing.assert_almost_equal([list(_) for _ in reloaded_model.predict(x_train)], [list(_) for _ in model.model.predict(x_train)], 5)
        # Clean
        remove_dir(model_dir)

    def test06_model_dense_regressor_reload_from_standalone(self):
        '''Test de la fonction ynov.models_training.regressors.model_dense_regressor.ModelDenseRegressor.reload_from_standalone'''

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
        model = ModelDenseRegressor(x_col=x_col, y_col=y_col_mono, model_dir=model_dir, batch_size=8, epochs=2)
        model.fit(x_train, y_train_regressor)
        model.save()
        # Reload
        conf_path = os.path.join(model.model_dir, "configurations.json")
        hdf5_path = os.path.join(model.model_dir, "best.hdf5")
        preprocess_pipeline_path = os.path.join(model.model_dir, "preprocess_pipeline.pkl")
        new_model = ModelDenseRegressor()
        self.assertTrue(new_model.preprocess_pipeline is None)
        new_model.reload_from_standalone(configuration_path=conf_path, hdf5_path=hdf5_path,
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
        self.assertEqual(model.batch_size, new_model.batch_size)
        self.assertEqual(model.epochs, new_model.epochs)
        self.assertEqual(model.validation_split, new_model.validation_split)
        self.assertEqual(model.patience, new_model.patience)
        self.assertEqual(model.nb_iter_keras, new_model.nb_iter_keras)
        self.assertEqual(model.keras_params, new_model.keras_params)
        self.assertEqual(model.custom_objects, new_model.custom_objects)
        self.assertTrue(new_model.preprocess_pipeline is not None)
        # On ne peut pas vraiment tester la pipeline, du coup on test les predictions
        self.assertEqual([[_] for _ in model.predict(x_train)], [[_] for _ in new_model.predict(x_train)])
        remove_dir(new_model.model_dir)
        # On ne remove pas model_dir pour tester les erreurs

        ############################################
        # Errors
        ############################################

        with self.assertRaises(FileNotFoundError):
            new_model = ModelDenseRegressor()
            new_model.reload_from_standalone(configuration_path='toto.json', hdf5_path=hdf5_path,
                                             preprocess_pipeline_path=preprocess_pipeline_path)
        with self.assertRaises(FileNotFoundError):
            new_model = ModelDenseRegressor()
            new_model.reload_from_standalone(configuration_path=conf_path, hdf5_path='toto.pkl',
                                             preprocess_pipeline_path=preprocess_pipeline_path)
        with self.assertRaises(FileNotFoundError):
            new_model = ModelDenseRegressor()
            new_model.reload_from_standalone(configuration_path=conf_path, hdf5_path=hdf5_path,
                                             preprocess_pipeline_path='toto.pkl')

        # Clean
        remove_dir(model_dir)


# Execution des tests
if __name__ == '__main__':
    # Start tests
    unittest.main()