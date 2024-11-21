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
from ynov.models_training.classifiers.model_dense_classifier import ModelDenseClassifier

# Disable logging
import logging
logging.disable(logging.CRITICAL)


def remove_dir(path):
    if os.path.isdir(path): shutil.rmtree(path)


class ModelDenseClassifierTests(unittest.TestCase):
    '''Main class to test model_dense_classifier'''


    def setUp(self):
        '''SetUp fonction'''
        # On se place dans le bon répertoire
        # Change directory to script directory
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname)


    def test01_model_dense_classifier_init(self):
        '''Test de ynov.models_training.classifiers.model_dense_classifier.ModelDenseClassifier.__init__'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)


        # Init., test all params
        model = ModelDenseClassifier(model_dir=model_dir)
        self.assertEqual(model.model_dir, model_dir)
        self.assertEqual(model.model_type, 'classifier')
        self.assertTrue(os.path.isdir(model_dir))
        self.assertEqual(model.model, None)
        # On fait juste un appel à display_if_gpu_activated et _is_gpu_activated
        self.assertTrue(type(model._is_gpu_activated()) == bool)
        model.display_if_gpu_activated()
        remove_dir(model_dir)

        #
        model = ModelDenseClassifier(model_dir=model_dir, batch_size=8)
        self.assertEqual(model.batch_size, 8)
        remove_dir(model_dir)

        #
        model = ModelDenseClassifier(model_dir=model_dir, epochs=42)
        self.assertEqual(model.epochs, 42)
        remove_dir(model_dir)

        #
        model = ModelDenseClassifier(model_dir=model_dir, validation_split=0.3)
        self.assertEqual(model.validation_split, 0.3)
        remove_dir(model_dir)

        #
        model = ModelDenseClassifier(model_dir=model_dir, patience=65)
        self.assertEqual(model.patience, 65)
        remove_dir(model_dir)

        #
        model = ModelDenseClassifier(model_dir=model_dir, nb_iter_keras=2)
        self.assertEqual(model.nb_iter_keras, 2)
        remove_dir(model_dir)

        # keras_params doit marcher avec n'importe quoi !
        model = ModelDenseClassifier(model_dir=model_dir, keras_params={'toto': 5})
        self.assertEqual(model.keras_params, {'toto': 5})
        remove_dir(model_dir)

    # On ne test pas le fit : déjà bien fait dans test_model_keras.py

    def test02_model_dense_classifier_predict(self):
        '''Test de la fonction predict de ynov.models_training.classifiers.model_dense_classifier.ModelDenseClassifier'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Set vars
        x_train = pd.DataFrame({'col_1': [-5, -1, 0, -2, 2, -6, 3] * 10, 'col_2': [2, -1, -8, 2, 3, 12, 2] * 10})
        y_train_mono_2 = pd.Series([0, 0, 0, 0, 1, 1, 1] * 10)
        y_train_mono_3 = pd.Series([0, 0, 0, 2, 1, 1, 1] * 10)
        y_train_multi = pd.DataFrame({'y1': [0, 0, 0, 0, 1, 1, 1] * 10, 'y2': [1, 0, 0, 1, 1, 1, 1] * 10, 'y3': [0, 0, 1, 0, 1, 0, 1] * 10})
        x_col = ['col_1', 'col_2']
        y_col_mono = ['toto']
        y_col_multi = ['y1', 'y2', 'y3']

        # Classification - Mono label - Mono Class
        model = ModelDenseClassifier(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False)
        model.fit(x_train, y_train_mono_2)
        preds = model.predict(x_train, return_proba=False)
        self.assertEqual(preds.shape, (len(x_train),))
        proba = model.predict(x_train, return_proba=True)
        self.assertEqual(proba.shape, (len(x_train), 2)) # 2 classes
        preds = model.predict(x_train, return_proba=False, experimental_version=True)
        self.assertEqual(preds.shape, (len(x_train),))
        proba = model.predict(x_train, return_proba=True, experimental_version=True)
        self.assertEqual(proba.shape, (len(x_train), 2)) # 2 classes
        remove_dir(model_dir)
        # nb iter > 0
        model = ModelDenseClassifier(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False, nb_iter_keras=3)
        model.fit(x_train, y_train_mono_2)
        preds = model.predict(x_train, return_proba=False)
        self.assertEqual(preds.shape, (len(x_train),))
        proba = model.predict(x_train, return_proba=True)
        self.assertEqual(proba.shape, (len(x_train), 2)) # 2 classes
        preds = model.predict(x_train, return_proba=False, experimental_version=True)
        self.assertEqual(preds.shape, (len(x_train),))
        proba = model.predict(x_train, return_proba=True, experimental_version=True)
        self.assertEqual(proba.shape, (len(x_train), 2)) # 2 classes
        remove_dir(model_dir)

        # Classification - Mono label - Multi Class
        model = ModelDenseClassifier(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False)
        model.fit(x_train, y_train_mono_3)
        preds = model.predict(x_train, return_proba=False)
        self.assertEqual(preds.shape, (len(x_train),))
        proba = model.predict(x_train, return_proba=True)
        self.assertEqual(proba.shape, (len(x_train), 3)) # 3 classes
        preds = model.predict(x_train, return_proba=False, experimental_version=True)
        self.assertEqual(preds.shape, (len(x_train),))
        proba = model.predict(x_train, return_proba=True, experimental_version=True)
        self.assertEqual(proba.shape, (len(x_train), 3)) # 3 classes
        remove_dir(model_dir)
        # nb iter > 0
        model = ModelDenseClassifier(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False, nb_iter_keras=3)
        model.fit(x_train, y_train_mono_3)
        preds = model.predict(x_train, return_proba=False)
        self.assertEqual(preds.shape, (len(x_train),))
        proba = model.predict(x_train, return_proba=True)
        self.assertEqual(proba.shape, (len(x_train), 3)) # 3 classes
        preds = model.predict(x_train, return_proba=False, experimental_version=True)
        self.assertEqual(preds.shape, (len(x_train),))
        proba = model.predict(x_train, return_proba=True, experimental_version=True)
        self.assertEqual(proba.shape, (len(x_train), 3)) # 3 classes
        remove_dir(model_dir)

        # Classification - Multi label
        model = ModelDenseClassifier(model_dir=model_dir, batch_size=8, epochs=2, multi_label=True)
        model.fit(x_train, y_train_multi)
        preds = model.predict(x_train)
        self.assertEqual(preds.shape, (len(x_train), len(y_col_multi)))
        proba = model.predict(x_train, return_proba=True)
        self.assertEqual(proba.shape, (len(x_train), len(y_col_multi)))
        preds = model.predict(x_train, return_proba=False, experimental_version=True)
        self.assertEqual(preds.shape, (len(x_train), len(y_col_multi)))
        proba = model.predict(x_train, return_proba=True, experimental_version=True)
        self.assertEqual(proba.shape, (len(x_train), len(y_col_multi)))
        remove_dir(model_dir)
        # nb iter > 0
        model = ModelDenseClassifier(model_dir=model_dir, batch_size=8, epochs=2, multi_label=True, nb_iter_keras=3)
        model.fit(x_train, y_train_multi)
        preds = model.predict(x_train)
        self.assertEqual(preds.shape, (len(x_train), len(y_col_multi)))
        proba = model.predict(x_train, return_proba=True)
        self.assertEqual(proba.shape, (len(x_train), len(y_col_multi)))
        preds = model.predict(x_train, return_proba=False, experimental_version=True)
        self.assertEqual(preds.shape, (len(x_train), len(y_col_multi)))
        proba = model.predict(x_train, return_proba=True, experimental_version=True)
        self.assertEqual(proba.shape, (len(x_train), len(y_col_multi)))
        remove_dir(model_dir)

        # Model needs to be fitted
        with self.assertRaises(AttributeError):
            model = ModelDenseClassifier(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False)
            model.predict(x_train)
        remove_dir(model_dir)


    def test03_model_dense_classifier_predict_proba(self):
        '''Test de la fonction predict_proba de ynov.models_training.classifiers.model_dense_classifier.ModelDenseClassifier'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Set vars
        x_train = pd.DataFrame({'col_1': [-5, -1, 0, -2, 2, -6, 3] * 10, 'col_2': [2, -1, -8, 2, 3, 12, 2] * 10})
        y_train_mono_2 = pd.Series([0, 0, 0, 0, 1, 1, 1] * 10)
        y_train_mono_3 = pd.Series([0, 0, 0, 2, 1, 1, 1] * 10)
        y_train_multi = pd.DataFrame({'y1': [0, 0, 0, 0, 1, 1, 1] * 10, 'y2': [1, 0, 0, 1, 1, 1, 1] * 10, 'y3': [0, 0, 1, 0, 1, 0, 1] * 10})
        x_col = ['col_1', 'col_2']
        y_col_mono = ['toto']
        y_col_multi = ['y1', 'y2', 'y3']

        # Classification - Mono label - Mono Class
        model = ModelDenseClassifier(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False)
        model.fit(x_train, y_train_mono_2)
        preds = model.predict_proba(x_train)
        self.assertEqual(preds.shape, (len(x_train), 2)) # 2 classes
        self.assertTrue(isinstance(preds[0][0], (np.floating, float)))
        remove_dir(model_dir)

        # Classification - Mono label - Multi Class
        model = ModelDenseClassifier(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False)
        model.fit(x_train, y_train_mono_3)
        preds = model.predict_proba(x_train)
        self.assertEqual(preds.shape, (len(x_train), 3)) # 3 classes
        self.assertTrue(isinstance(preds[0][0], (np.floating, float)))
        remove_dir(model_dir)

        # Classification - Multi label
        model = ModelDenseClassifier(model_dir=model_dir, batch_size=8, epochs=2, multi_label=True)
        model.fit(x_train, y_train_multi)
        preds = model.predict_proba(x_train)
        self.assertEqual(preds.shape, (len(x_train), len(y_col_multi))) # 3 labels
        self.assertTrue(isinstance(preds[0][0], (np.floating, float)))
        remove_dir(model_dir)

        # Model needs to be fitted
        with self.assertRaises(AttributeError):
            model = ModelDenseClassifier(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False)
            model.predict_proba('test')
        remove_dir(model_dir)


    def test04_model_dense_classifier_get_predict_position(self):
        '''Test de la fonction ynov.models_training.classifiers.model_dense_classifier.ModelDenseClassifier.get_predict_position'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Set vars
        x_train = pd.DataFrame({'col_1': [-5, -1, 0, -2, 2, -6, 3] * 10, 'col_2': [2, -1, -8, 2, 3, 12, 2] * 10})
        y_train_mono_2 = pd.Series([0, 0, 0, 0, 1, 1, 1] * 10)
        y_train_mono_3 = pd.Series([0, 0, 0, 2, 1, 1, 1] * 10)
        y_train_multi = pd.DataFrame({'y1': [0, 0, 0, 0, 1, 1, 1] * 10, 'y2': [1, 0, 0, 1, 1, 1, 1] * 10, 'y3': [0, 0, 1, 0, 1, 0, 1] * 10})
        x_col = ['col_1', 'col_2']
        y_col_mono = ['toto']
        y_col_multi = ['y1', 'y2', 'y3']

        # Classification - Mono label - Mono Class
        model = ModelDenseClassifier(x_col=x_col, y_col=y_col_mono, model_dir=model_dir, epochs=2)
        model.fit(x_train, y_train_mono_2)
        predict_positions = model.get_predict_position(x_train, y_train_mono_2)
        self.assertEqual(predict_positions.shape, (len(x_train),))
        remove_dir(model_dir)

        # Classification - Mono label - Multi Class
        model = ModelDenseClassifier(x_col=x_col, y_col=y_col_mono, model_dir=model_dir, epochs=2)
        model.fit(x_train, y_train_mono_3)
        predict_positions = model.get_predict_position(x_train, y_train_mono_2)
        self.assertEqual(predict_positions.shape, (len(x_train),))
        remove_dir(model_dir)

        # Classification - Multi label
        model = ModelDenseClassifier(x_col=x_col, y_col=y_col_multi, model_dir=model_dir, multi_label=True)
        model.fit(x_train, y_train_multi)        # Pas dispo en multi-label
        with self.assertRaises(ValueError):
            model.get_predict_position(x_train, y_train_multi)
        remove_dir(model_dir)


    def test05_model_dense_classifier_get_model(self):
        '''Test de la fonction ynov.models_training.classifiers.model_dense_classifier.ModelDenseClassifier._get_model'''

        # Set vars
        x_col = ['col_1', 'col_2']
        y_col_mono = ['toto']
        y_col_multi = ['y1', 'y2', 'y3']

        # Create model
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        model = ModelDenseClassifier(x_col=x_col, y_col=y_col_mono, model_dir=model_dir, epochs=2)

        # Fonctionnement nominal
        model.list_classes = ['a', 'b']  # On force la création d'une liste de classes
        model_res = model._get_model()
        self.assertTrue(type(model_res) in [tensorflow.python.keras.engine.sequential.Sequential, tensorflow.python.keras.engine.functional.Functional])

        # Clean
        remove_dir(model_dir)

        #######
        # Pareil, en multi label

        # Create model
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        model = ModelDenseClassifier(x_col=x_col, y_col=y_col_multi, model_dir=model_dir, multi_label=True)

        # Fonctionnement nominal
        model.list_classes = ['a', 'b', 'c']  # On force la création d'une liste de classes
        model_res = model._get_model()
        self.assertTrue(type(model_res) in [tensorflow.python.keras.engine.sequential.Sequential, tensorflow.python.keras.engine.functional.Functional])

        # Clean
        remove_dir(model_dir)


    def test06_model_dense_classifier_save(self):
        '''Test de la fonction save de ynov.models_training.classifiers.model_dense_classifier.ModelDenseClassifier'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Set vars
        x_col = ['col_1', 'col_2']
        y_col_mono = ['toto']

        # Fonctionnement nominal
        model = ModelDenseClassifier(x_col=x_col, y_col=y_col_mono, model_dir=model_dir, epochs=2)
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
        self.assertEqual(configs['model_type'], 'classifier')
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
        self.assertTrue('list_classes' in configs.keys())
        self.assertTrue('dict_classes' in configs.keys())
        self.assertTrue('multi_label' in configs.keys())
        remove_dir(model_dir)

        # Fonctionnement avec custom_objects qui contient une fonction "partial"
        model = ModelDenseClassifier(x_col=x_col, y_col=y_col_mono, model_dir=model_dir, epochs=2)
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
        self.assertEqual(configs['model_type'], 'classifier')
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
        self.assertTrue('list_classes' in configs.keys())
        self.assertTrue('dict_classes' in configs.keys())
        self.assertTrue('multi_label' in configs.keys())
        remove_dir(model_dir)


    def test07_model_dense_classifier_reload_model(self):
        '''Test de la fonction reload_model de ynov.models_training.classifiers.model_dense_classifier.ModelDenseClassifier'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Set vars
        x_train = pd.DataFrame({'col_1': [-5, -1, 0, -2, 2, -6, 3] * 10, 'col_2': [2, -1, -8, 2, 3, 12, 2] * 10})
        y_train_mono_2 = pd.Series([0, 0, 0, 0, 1, 1, 1] * 10)
        y_train_mono_3 = pd.Series([0, 0, 0, 2, 1, 1, 1] * 10)
        y_train_multi = pd.DataFrame({'y1': [0, 0, 0, 0, 1, 1, 1] * 10, 'y2': [1, 0, 0, 1, 1, 1, 1] * 10, 'y3': [0, 0, 1, 0, 1, 0, 1] * 10})
        x_col = ['col_1', 'col_2']
        y_col_mono = ['toto']
        y_col_multi = ['y1', 'y2', 'y3']

        # Classification - Mono label - Mono Class
        model = ModelDenseClassifier(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False)
        model.fit(x_train, y_train_mono_2)
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

        # Classification - Mono label - Multi Class
        model = ModelDenseClassifier(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False)
        model.fit(x_train, y_train_mono_3)
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

        # Classification - Multi label
        model = ModelDenseClassifier(model_dir=model_dir, batch_size=8, epochs=2, multi_label=True)
        model.fit(x_train, y_train_multi)
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


    def test08_model_dense_classifier_reload_from_standalone(self):
        '''Test de la fonction ynov.models_training.classifiers.model_dense_classifier.ModelDenseClassifier.reload_from_standalone'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Set vars
        x_train = pd.DataFrame({'col_1': [-5, -1, 0, -2, 2, -6, 3] * 10, 'col_2': [2, -1, -8, 2, 3, 12, 2] * 10})
        y_train_mono_2 = pd.Series([0, 0, 0, 0, 1, 1, 1] * 10)
        y_train_mono_3 = pd.Series([0, 0, 0, 2, 1, 1, 1] * 10)
        y_train_multi = pd.DataFrame({'y1': [0, 0, 0, 0, 1, 1, 1] * 10, 'y2': [1, 0, 0, 1, 1, 1, 1] * 10, 'y3': [0, 0, 1, 0, 1, 0, 1] * 10})
        x_col = ['col_1', 'col_2']
        y_col_mono = ['toto']
        y_col_multi = ['y1', 'y2', 'y3']

        ############################################
        # Classification - Mono label
        ############################################

        # Create model
        model = ModelDenseClassifier(x_col=x_col, y_col=y_col_mono, model_dir=model_dir, epochs=2)
        model.fit(x_train, y_train_mono_2)
        model.save()
        # Reload
        conf_path = os.path.join(model.model_dir, "configurations.json")
        hdf5_path = os.path.join(model.model_dir, "best.hdf5")
        preprocess_pipeline_path = os.path.join(model.model_dir, "preprocess_pipeline.pkl")
        new_model = ModelDenseClassifier()
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
        self.assertEqual(model.list_classes, new_model.list_classes)
        self.assertEqual(model.dict_classes, new_model.dict_classes)
        self.assertEqual(model.batch_size, new_model.batch_size)
        self.assertEqual(model.epochs, new_model.epochs)
        self.assertEqual(model.validation_split, new_model.validation_split)
        self.assertEqual(model.patience, new_model.patience)
        self.assertEqual(model.nb_iter_keras, new_model.nb_iter_keras)
        self.assertEqual(model.keras_params, new_model.keras_params)
        self.assertEqual(model.custom_objects, new_model.custom_objects)
        self.assertTrue(new_model.preprocess_pipeline is not None)
        # On ne peut pas vraiment tester la pipeline, du coup on test les predictions
        self.assertEqual([list(_) for _ in model.predict_proba(x_train)], [list(_) for _ in new_model.predict_proba(x_train)])
        remove_dir(model_dir)
        remove_dir(new_model.model_dir)

        ############################################
        # Classification - Multi label
        ############################################

        # Create model
        model = ModelDenseClassifier(x_col=x_col, y_col=y_col_multi, model_dir=model_dir, multi_label=True, epochs=2)
        model.fit(x_train, y_train_multi)
        model.save()
        # Reload
        conf_path = os.path.join(model.model_dir, "configurations.json")
        hdf5_path = os.path.join(model.model_dir, "best.hdf5")
        preprocess_pipeline_path = os.path.join(model.model_dir, "preprocess_pipeline.pkl")
        new_model = ModelDenseClassifier()
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
        self.assertEqual(model.list_classes, new_model.list_classes)
        self.assertEqual(model.dict_classes, new_model.dict_classes)
        self.assertEqual(model.batch_size, new_model.batch_size)
        self.assertEqual(model.epochs, new_model.epochs)
        self.assertEqual(model.validation_split, new_model.validation_split)
        self.assertEqual(model.patience, new_model.patience)
        self.assertEqual(model.nb_iter_keras, new_model.nb_iter_keras)
        self.assertEqual(model.keras_params, new_model.keras_params)
        self.assertEqual(model.custom_objects, new_model.custom_objects)
        self.assertTrue(new_model.preprocess_pipeline is not None)
        # On ne peut pas vraiment tester la pipeline, du coup on test les predictions
        self.assertEqual([list(_) for _ in model.predict_proba(x_train)], [list(_) for _ in new_model.predict_proba(x_train)])
        remove_dir(new_model.model_dir)
        # On ne remove pas model_dir pour tester les erreurs


        ############################################
        # Errors
        ############################################

        with self.assertRaises(FileNotFoundError):
            new_model = ModelDenseClassifier()
            new_model.reload_from_standalone(configuration_path='toto.json', hdf5_path=hdf5_path,
                                             preprocess_pipeline_path=preprocess_pipeline_path)
        with self.assertRaises(FileNotFoundError):
            new_model = ModelDenseClassifier()
            new_model.reload_from_standalone(configuration_path=conf_path, hdf5_path='toto.pkl',
                                             preprocess_pipeline_path=preprocess_pipeline_path)
        with self.assertRaises(FileNotFoundError):
            new_model = ModelDenseClassifier()
            new_model.reload_from_standalone(configuration_path=conf_path, hdf5_path=hdf5_path,
                                             preprocess_pipeline_path='toto.pkl')

        # Clean
        remove_dir(model_dir)


# Execution des tests
if __name__ == '__main__':
    # Start tests
    unittest.main()