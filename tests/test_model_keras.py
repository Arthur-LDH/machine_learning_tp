#!/usr/bin/env python3

# Libs unittest
import unittest
from unittest.mock import patch
from unittest.mock import Mock

# Utils libs
import os
import json
import shutil
import dill as pickle
import pandas as pd
import numpy as np
import tensorflow
from ynov import utils
from ynov.models_training import utils_deep_keras
from ynov.models_training.model_keras import ModelKeras
from ynov.models_training.classifiers.model_dense_classifier import ModelDenseClassifier
from ynov.models_training.regressors.model_dense_regressor import ModelDenseRegressor

# Disable logging
import logging
logging.disable(logging.CRITICAL)


def remove_dir(path):
    if os.path.isdir(path): shutil.rmtree(path)


class ModelKerasTests(unittest.TestCase):
    '''Main class to test model_keras'''


    def setUp(self):
        '''Setup fonction -> on créer un fake embedding'''
        # On se place dans le bon répertoire
        # Change directory to script directory
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname)


    def test01_model_keras_init(self):
        '''Test de l'initialisation de ynov.models_training.model_keras.ModelKeras'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Init., test all params
        model = ModelKeras(model_dir=model_dir)
        self.assertEqual(model.model_dir, model_dir)
        self.assertEqual(model.model_type, None)
        self.assertTrue(os.path.isdir(model_dir))
        self.assertEqual(model.model, None)
        # On fait juste un appel à display_if_gpu_activated et _is_gpu_activated
        self.assertTrue(type(model._is_gpu_activated()) == bool)
        model.display_if_gpu_activated()
        remove_dir(model_dir)

        #
        model = ModelKeras(model_dir=model_dir, batch_size=8)
        self.assertEqual(model.batch_size, 8)
        remove_dir(model_dir)

        #
        model = ModelKeras(model_dir=model_dir, epochs=42)
        self.assertEqual(model.epochs, 42)
        remove_dir(model_dir)

        #
        model = ModelKeras(model_dir=model_dir, validation_split=0.3)
        self.assertEqual(model.validation_split, 0.3)
        remove_dir(model_dir)

        #
        model = ModelKeras(model_dir=model_dir, patience=65)
        self.assertEqual(model.patience, 65)
        remove_dir(model_dir)

        #
        model = ModelKeras(model_dir=model_dir, nb_iter_keras=2)
        self.assertEqual(model.nb_iter_keras, 2)
        remove_dir(model_dir)

        # keras_params doit marcher avec n'importe quoi !
        model = ModelKeras(model_dir=model_dir, keras_params={'toto': 5})
        self.assertEqual(model.keras_params, {'toto': 5})
        remove_dir(model_dir)


    def test02_model_keras_fit(self):
        '''Test de la fonction fit de ynov.models_training.model_keras.ModelKeras'''
        # /!\ On test avec model_embedding_lstm /!\

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Set vars
        x_train = pd.DataFrame({'col_1': [-5, -1, 0, -2, 2, -6, 3] * 10, 'col_2': [2, -1, -8, 2, 3, 12, 2] * 10})
        y_train_mono_2 = pd.Series([0, 0, 0, 0, 1, 1, 1] * 10)
        y_train_mono_3 = pd.Series([0, 0, 0, 2, 1, 1, 1] * 10)
        y_train_regressor = pd.Series([-3, -2, -8, 0, 5, 6, 5] * 10)
        y_train_multi = pd.DataFrame({'y1': [0, 0, 0, 0, 1, 1, 1] * 10, 'y2': [1, 0, 0, 1, 1, 1, 1] * 10, 'y3': [0, 0, 1, 0, 1, 0, 1] * 10})
        x_col = ['col_1', 'col_2']
        y_col_mono = ['toto']
        y_col_multi = ['y1', 'y2', 'y3']
        # Pour les "valids" on ne s'embête pas, on réutilise les "trains"

        #
        lr = 0.123456

        ## Classification - Mono label - Mono Class
        model = ModelDenseClassifier(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False, keras_params={'learning_rate': lr, 'decay': 0.0})
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        model.fit(x_train, y_train_mono_2, x_valid=None, y_valid=None, with_shuffle=True)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(sorted(model.list_classes), [0, 1])
        self.assertTrue(model.model._is_compiled)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertEqual(lr, round(float(model.model.optimizer._decayed_lr(tensorflow.float32).numpy()), 6))
        remove_dir(model_dir)
        # avec valid
        model = ModelDenseClassifier(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        model.fit(x_train, y_train_mono_2, x_valid=x_train, y_valid=y_train_mono_2, with_shuffle=True)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(sorted(model.list_classes), [0, 1])
        self.assertTrue(model.model._is_compiled)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        remove_dir(model_dir)
        # Avec shuffle à False
        model = ModelDenseClassifier(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        model.fit(x_train, y_train_mono_2, x_valid=x_train, y_valid=y_train_mono_2, with_shuffle=False)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(sorted(model.list_classes), [0, 1])
        self.assertTrue(model.model._is_compiled)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        # Avec plusieurs iterations
        model = ModelDenseClassifier(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False, nb_iter_keras=3)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        self.assertEqual(model.nb_iter_keras, 3)
        model.fit(x_train, y_train_mono_2, x_valid=x_train, y_valid=y_train_mono_2, with_shuffle=False)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(sorted(model.list_classes), [0, 1])
        self.assertTrue(model.model._is_compiled)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best_1.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best_2.hdf5')))
        # Mauvais x_col
        with self.assertRaises(ValueError):
            model = ModelDenseClassifier(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False, x_col=['toto'])
            model.fit(x_train, y_train_mono_2, x_valid=None, y_valid=None, with_shuffle=True)
        # Test continue training
        model = ModelDenseClassifier(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        model.fit(x_train, y_train_mono_2, x_valid=x_train, y_valid=y_train_mono_2, with_shuffle=False)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        model.save()
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        # 2nd fit
        model.fit(x_train, y_train_mono_2, x_valid=x_train, y_valid=y_train_mono_2, with_shuffle=False) # On refit avec les mêmes données, pas grave
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 2)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations_fit_1.json')))
        # On ne save pas volontairement
        model_dir_2 = model.model_dir
        self.assertNotEqual(model_dir, model_dir_2)
        # 3rd fit
        model.fit(x_train, y_train_mono_2, x_valid=x_train, y_valid=y_train_mono_2, with_shuffle=False) # On refit avec les mêmes données, pas grave
        model.save()
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 3)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations_fit_1.json')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'configurations_fit_2.json')))
        model_dir_3 = model.model_dir
        self.assertNotEqual(model_dir_2, model_dir_3)
        remove_dir(model_dir)
        remove_dir(model_dir_2)
        remove_dir(model_dir_3)
        # Test erreur iterations
        model = ModelDenseClassifier(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False, nb_iter_keras=2)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        model.fit(x_train, y_train_mono_2, x_valid=x_train, y_valid=y_train_mono_2, with_shuffle=False)
        model.save()
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        with self.assertRaises(RuntimeError):
            model.fit(x_train, y_train_mono_2, x_valid=x_train, y_valid=y_train_mono_2, with_shuffle=False)
        self.assertEqual(model_dir, model.model_dir)
        remove_dir(model_dir)
        # Test erreur datas
        model = ModelDenseClassifier(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        model.fit(x_train, y_train_mono_2, x_valid=x_train, y_valid=y_train_mono_2, with_shuffle=False)
        model.save()
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        y_train_mono_2_fake = pd.Series([0, 0, 0, 0, 2, 2, 2] * 10)
        with self.assertRaises(AssertionError):
            model.fit(x_train, y_train_mono_2_fake, x_valid=x_train, y_valid=y_train_mono_2, with_shuffle=False)
        remove_dir(model_dir)





        ## Classification - Mono label - Multi Class
        model = ModelDenseClassifier(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False, keras_params={'learning_rate': lr, 'decay': 0.0})
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        model.fit(x_train, y_train_mono_3, x_valid=None, y_valid=None, with_shuffle=True)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(sorted(model.list_classes), [0, 1, 2])
        self.assertTrue(model.model._is_compiled)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertEqual(lr, round(float(model.model.optimizer._decayed_lr(tensorflow.float32).numpy()), 6))
        remove_dir(model_dir)
        # avec valid
        model = ModelDenseClassifier(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        model.fit(x_train, y_train_mono_3, x_valid=x_train, y_valid=y_train_mono_3, with_shuffle=True)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(sorted(model.list_classes), [0, 1, 2])
        self.assertTrue(model.model._is_compiled)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        remove_dir(model_dir)
        # Avec shuffle à False
        model = ModelDenseClassifier(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        model.fit(x_train, y_train_mono_3, x_valid=x_train, y_valid=y_train_mono_3, with_shuffle=False)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(sorted(model.list_classes), [0, 1, 2])
        self.assertTrue(model.model._is_compiled)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        # Avec plusieurs iterations
        model = ModelDenseClassifier(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False, nb_iter_keras=3)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        self.assertEqual(model.nb_iter_keras, 3)
        model.fit(x_train, y_train_mono_3, x_valid=x_train, y_valid=y_train_mono_3, with_shuffle=False)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(sorted(model.list_classes), [0, 1, 2])
        self.assertTrue(model.model._is_compiled)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best_1.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best_2.hdf5')))
        # Mauvais x_col
        with self.assertRaises(ValueError):
            model = ModelDenseClassifier(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False, x_col=['toto'])
            model.fit(x_train, y_train_mono_3, x_valid=None, y_valid=None, with_shuffle=True)
        # Test continue training
        model = ModelDenseClassifier(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        model.fit(x_train, y_train_mono_3, x_valid=x_train, y_valid=y_train_mono_3, with_shuffle=False)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        model.save()
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        # 2nd fit
        model.fit(x_train, y_train_mono_3, x_valid=x_train, y_valid=y_train_mono_3, with_shuffle=False) # On refit avec les mêmes données, pas grave
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 2)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations_fit_1.json')))
        # On ne save pas volontairement
        model_dir_2 = model.model_dir
        self.assertNotEqual(model_dir, model_dir_2)
        # 3rd fit
        model.fit(x_train, y_train_mono_3, x_valid=x_train, y_valid=y_train_mono_3, with_shuffle=False) # On refit avec les mêmes données, pas grave
        model.save()
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 3)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations_fit_1.json')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'configurations_fit_2.json')))
        model_dir_3 = model.model_dir
        self.assertNotEqual(model_dir_2, model_dir_3)
        remove_dir(model_dir)
        remove_dir(model_dir_2)
        remove_dir(model_dir_3)
        # Test erreur iterations
        model = ModelDenseClassifier(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False, nb_iter_keras=2)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        model.fit(x_train, y_train_mono_3, x_valid=x_train, y_valid=y_train_mono_3, with_shuffle=False)
        model.save()
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        with self.assertRaises(RuntimeError):
            model.fit(x_train, y_train_mono_3, x_valid=x_train, y_valid=y_train_mono_3, with_shuffle=False)
        self.assertEqual(model_dir, model.model_dir)
        remove_dir(model_dir)
        # Test erreur datas
        model = ModelDenseClassifier(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        model.fit(x_train, y_train_mono_3, x_valid=x_train, y_valid=y_train_mono_3, with_shuffle=False)
        model.save()
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        y_train_mono_3_fake = pd.Series([5, 5, 8, 8, 2, 2, 2] * 10)
        with self.assertRaises(AssertionError):
            model.fit(x_train, y_train_mono_3_fake, x_valid=x_train, y_valid=y_train_mono_3, with_shuffle=False)
        remove_dir(model_dir)





        ## Classification - Multi label
        model = ModelDenseClassifier(model_dir=model_dir, batch_size=8, epochs=2, multi_label=True, keras_params={'learning_rate': lr, 'decay': 0.0})
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        self.assertTrue(model.multi_label)
        model.fit(x_train, y_train_multi, x_valid=None, y_valid=None, with_shuffle=True)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(sorted(model.list_classes), y_col_multi)
        self.assertTrue(model.model._is_compiled)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertEqual(lr, round(float(model.model.optimizer._decayed_lr(tensorflow.float32).numpy()), 6))
        remove_dir(model_dir)
        # avec valid
        model = ModelDenseClassifier(model_dir=model_dir, batch_size=8, epochs=2, multi_label=True)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        self.assertTrue(model.multi_label)
        model.fit(x_train, y_train_multi, x_valid=x_train, y_valid=y_train_multi, with_shuffle=True)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(sorted(model.list_classes), y_col_multi)
        self.assertTrue(model.model._is_compiled)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        remove_dir(model_dir)
        # Avec shuffle à False
        model = ModelDenseClassifier(model_dir=model_dir, batch_size=8, epochs=2, multi_label=True)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        self.assertTrue(model.multi_label)
        model.fit(x_train, y_train_multi, x_valid=x_train, y_valid=y_train_multi, with_shuffle=False)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(sorted(model.list_classes), y_col_multi)
        self.assertTrue(model.model._is_compiled)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        # Avec plusieurs iterations
        model = ModelDenseClassifier(model_dir=model_dir, batch_size=8, epochs=2, multi_label=True, nb_iter_keras=3)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        self.assertTrue(model.multi_label)
        self.assertEqual(model.nb_iter_keras, 3)
        model.fit(x_train, y_train_multi, x_valid=x_train, y_valid=y_train_multi, with_shuffle=False)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(sorted(model.list_classes), y_col_multi)
        self.assertTrue(model.model._is_compiled)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best_1.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best_2.hdf5')))
        # Mauvais x_col
        with self.assertRaises(ValueError):
            model = ModelDenseClassifier(model_dir=model_dir, batch_size=8, epochs=2, multi_label=True, x_col=['toto'])
            model.fit(x_train, y_train_multi, x_valid=None, y_valid=None, with_shuffle=True)
        # Test continue training
        model = ModelDenseClassifier(model_dir=model_dir, batch_size=8, epochs=2, multi_label=True)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        model.fit(x_train, y_train_multi, x_valid=x_train, y_valid=y_train_multi, with_shuffle=False)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        model.save()
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        # 2nd fit
        model.fit(x_train, y_train_multi, x_valid=x_train, y_valid=y_train_multi, with_shuffle=False) # On refit avec les mêmes données, pas grave
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 2)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations_fit_1.json')))
        # On ne save pas volontairement
        model_dir_2 = model.model_dir
        self.assertNotEqual(model_dir, model_dir_2)
        # 3rd fit
        model.fit(x_train, y_train_multi, x_valid=x_train, y_valid=y_train_multi, with_shuffle=False) # On refit avec les mêmes données, pas grave
        model.save()
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 3)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations_fit_1.json')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'configurations_fit_2.json')))
        model_dir_3 = model.model_dir
        self.assertNotEqual(model_dir_2, model_dir_3)
        remove_dir(model_dir)
        remove_dir(model_dir_2)
        remove_dir(model_dir_3)
        # Test erreur iterations
        model = ModelDenseClassifier(model_dir=model_dir, batch_size=8, epochs=2, multi_label=True, nb_iter_keras=2)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        model.fit(x_train, y_train_multi, x_valid=x_train, y_valid=y_train_multi, with_shuffle=False)
        model.save()
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        with self.assertRaises(RuntimeError):
            model.fit(x_train, y_train_multi, x_valid=x_train, y_valid=y_train_multi, with_shuffle=False)
        self.assertEqual(model_dir, model.model_dir)
        remove_dir(model_dir)




        ## Regression
        model = ModelDenseRegressor(model_dir=model_dir, batch_size=8, epochs=2, keras_params={'learning_rate': lr, 'decay': 0.0})
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        model.fit(x_train, y_train_regressor, x_valid=None, y_valid=None, with_shuffle=True)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertTrue(model.model._is_compiled)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertEqual(lr, round(float(model.model.optimizer._decayed_lr(tensorflow.float32).numpy()), 6))
        remove_dir(model_dir)
        # avec valid
        model = ModelDenseRegressor(model_dir=model_dir, batch_size=8, epochs=2)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        model.fit(x_train, y_train_regressor, x_valid=x_train, y_valid=y_train_regressor, with_shuffle=True)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertTrue(model.model._is_compiled)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        remove_dir(model_dir)
        # Avec shuffle à False
        model = ModelDenseRegressor(model_dir=model_dir, batch_size=8, epochs=2)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        model.fit(x_train, y_train_regressor, x_valid=x_train, y_valid=y_train_regressor, with_shuffle=False)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertTrue(model.model._is_compiled)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        # Avec plusieurs iterations
        model = ModelDenseRegressor(model_dir=model_dir, batch_size=8, epochs=2, nb_iter_keras=3)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        self.assertEqual(model.nb_iter_keras, 3)
        model.fit(x_train, y_train_regressor, x_valid=x_train, y_valid=y_train_regressor, with_shuffle=False)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertTrue(model.model._is_compiled)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best_1.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best_2.hdf5')))
        # Mauvais x_col
        with self.assertRaises(ValueError):
            model = ModelDenseRegressor(model_dir=model_dir, batch_size=8, epochs=2, x_col=['toto'])
            model.fit(x_train, y_train_regressor, x_valid=None, y_valid=None, with_shuffle=True)
        # Test continue training
        model = ModelDenseRegressor(model_dir=model_dir, batch_size=8, epochs=2)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        model.fit(x_train, y_train_regressor, x_valid=x_train, y_valid=y_train_regressor, with_shuffle=False)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        model.save()
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        # 2nd fit
        model.fit(x_train, y_train_regressor, x_valid=x_train, y_valid=y_train_regressor, with_shuffle=False) # On refit avec les mêmes données, pas grave
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 2)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations_fit_1.json')))
        # On ne save pas volontairement
        model_dir_2 = model.model_dir
        self.assertNotEqual(model_dir, model_dir_2)
        # 3rd fit
        model.fit(x_train, y_train_regressor, x_valid=x_train, y_valid=y_train_regressor, with_shuffle=False) # On refit avec les mêmes données, pas grave
        model.save()
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 3)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations_fit_1.json')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'configurations_fit_2.json')))
        model_dir_3 = model.model_dir
        self.assertNotEqual(model_dir_2, model_dir_3)
        remove_dir(model_dir)
        remove_dir(model_dir_2)
        remove_dir(model_dir_3)
        # Test erreur iterations
        model = ModelDenseRegressor(model_dir=model_dir, batch_size=8, epochs=2, nb_iter_keras=2)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        model.fit(x_train, y_train_regressor, x_valid=x_train, y_valid=y_train_regressor, with_shuffle=False)
        model.save()
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        with self.assertRaises(RuntimeError):
            model.fit(x_train, y_train_regressor, x_valid=x_train, y_valid=y_train_regressor, with_shuffle=False)
        self.assertEqual(model_dir, model.model_dir)
        remove_dir(model_dir)


    def test03_model_keras_predict(self):
        '''Test de la fonction predict de ynov.models_training.model_keras.ModelKeras'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Set vars
        x_train = pd.DataFrame({'col_1': [-5, -1, 0, -2, 2, -6, 3] * 10, 'col_2': [2, -1, -8, 2, 3, 12, 2] * 10})
        y_train_mono_2 = pd.Series([0, 0, 0, 0, 1, 1, 1] * 10)
        y_train_mono_3 = pd.Series([0, 0, 0, 2, 1, 1, 1] * 10)
        y_train_regressor = pd.Series([-3, -2, -8, 0, 5, 6, 5] * 10)
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

        # Regressor
        model = ModelDenseRegressor(model_dir=model_dir, batch_size=8, epochs=2)
        model.fit(x_train, y_train_regressor)
        preds = model.predict(x_train)
        self.assertEqual(preds.shape, (len(x_train),))
        with self.assertRaises(ValueError):
            proba = model.predict(x_train, return_proba=True)
        preds = model.predict(x_train, experimental_version=True)
        self.assertEqual(preds.shape, (len(x_train),))
        with self.assertRaises(ValueError):
            proba = model.predict(x_train, return_proba=True, experimental_version=True)
        remove_dir(model_dir)
        # nb iter > 0
        model = ModelDenseRegressor(model_dir=model_dir, batch_size=8, epochs=2, nb_iter_keras=3)
        model.fit(x_train, y_train_regressor)
        preds = model.predict(x_train)
        self.assertEqual(preds.shape, (len(x_train),))
        with self.assertRaises(ValueError):
            proba = model.predict(x_train, return_proba=True)
        preds = model.predict(x_train, experimental_version=True)
        self.assertEqual(preds.shape, (len(x_train),))
        with self.assertRaises(ValueError):
            proba = model.predict(x_train, return_proba=True, experimental_version=True)
        remove_dir(model_dir)

        # Model needs to be fitted
        with self.assertRaises(AttributeError):
            model = ModelDenseClassifier(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False)
            model.predict(x_train)
        remove_dir(model_dir)


    def test04_model_keras_predict_proba(self):
        '''Test de la fonction predict_proba de ynov.models_training.model_keras.ModelPipeline'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Set vars
        x_train = pd.DataFrame({'col_1': [-5, -1, 0, -2, 2, -6, 3] * 10, 'col_2': [2, -1, -8, 2, 3, 12, 2] * 10})
        y_train_mono_2 = pd.Series([0, 0, 0, 0, 1, 1, 1] * 10)
        y_train_mono_3 = pd.Series([0, 0, 0, 2, 1, 1, 1] * 10)
        y_train_regressor = pd.Series([-3, -2, -8, 0, 5, 6, 5] * 10)
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

        # Regressor
        model = ModelDenseRegressor(model_dir=model_dir, batch_size=8, epochs=2)
        model.fit(x_train, y_train_regressor)
        with self.assertRaises(ValueError):
            proba = model.predict_proba(x_train)
        remove_dir(model_dir)

        # Model needs to be fitted
        with self.assertRaises(AttributeError):
            model = ModelDenseClassifier(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False)
            model.predict_proba('test')
        remove_dir(model_dir)


    def test05_model_keras_get_callbacks(self):
        '''Test de la fonction _get_callbacks de ynov.models_training.model_keras.ModelKeras'''

        # Create model
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        model = ModelKeras(model_dir=model_dir)

        # Fonctionnement nominal
        callbacks = model._get_callbacks()
        callbacks_types = [type(_) for _ in callbacks]
        self.assertTrue(tensorflow.python.keras.callbacks.EarlyStopping in callbacks_types)
        self.assertTrue(tensorflow.python.keras.callbacks.ModelCheckpoint in callbacks_types)
        self.assertTrue(tensorflow.python.keras.callbacks.CSVLogger in callbacks_types)
        self.assertTrue(tensorflow.python.keras.callbacks.TerminateOnNaN in callbacks_types)
        checkpoint = callbacks[callbacks_types.index(tensorflow.python.keras.callbacks.ModelCheckpoint)]
        csv_logger = callbacks[callbacks_types.index(tensorflow.python.keras.callbacks.CSVLogger)]
        self.assertEqual(checkpoint.filepath, os.path.join(model.model_dir, 'best.hdf5'))
        self.assertEqual(csv_logger.filename, os.path.join(model.model_dir, 'logger.csv'))

        # iter > 0
        callbacks = model._get_callbacks(2)
        callbacks_types = [type(_) for _ in callbacks]
        self.assertTrue(tensorflow.python.keras.callbacks.EarlyStopping in callbacks_types)
        self.assertTrue(tensorflow.python.keras.callbacks.ModelCheckpoint in callbacks_types)
        self.assertTrue(tensorflow.python.keras.callbacks.CSVLogger in callbacks_types)
        self.assertTrue(tensorflow.python.keras.callbacks.TerminateOnNaN in callbacks_types)
        checkpoint = callbacks[callbacks_types.index(tensorflow.python.keras.callbacks.ModelCheckpoint)]
        csv_logger = callbacks[callbacks_types.index(tensorflow.python.keras.callbacks.CSVLogger)]
        self.assertEqual(checkpoint.filepath, os.path.join(model.model_dir, 'best_2.hdf5'))
        self.assertEqual(csv_logger.filename, os.path.join(model.model_dir, 'logger_2.csv'))

        # level save 'LOW'
        model.level_save = 'LOW'
        callbacks = model._get_callbacks()
        callbacks_types = [type(_) for _ in callbacks]
        self.assertTrue(tensorflow.python.keras.callbacks.EarlyStopping in callbacks_types)
        self.assertFalse(tensorflow.python.keras.callbacks.ModelCheckpoint in callbacks_types)
        self.assertTrue(tensorflow.python.keras.callbacks.CSVLogger in callbacks_types)
        self.assertTrue(tensorflow.python.keras.callbacks.TerminateOnNaN in callbacks_types)
        csv_logger = callbacks[callbacks_types.index(tensorflow.python.keras.callbacks.CSVLogger)]
        self.assertEqual(csv_logger.filename, os.path.join(model.model_dir, 'logger.csv'))

        # Clean
        remove_dir(model_dir)


    def test06_model_keras_get_learning_rate_scheduler(self):
        '''Test de la fonction _get_learning_rate_scheduler de ynov.models_training.model_keras.ModelKeras'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Fonctionnement nominal
        model = ModelKeras(model_dir=model_dir)
        self.assertEqual(model._get_learning_rate_scheduler(), None)

        # Clean
        remove_dir(model_dir)


    def test07_model_keras_save(self):
        '''Test de la fonction save de ynov.models_training.model_keras.ModelKeras'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Fonctionnement nominal
        model = ModelKeras(model_dir=model_dir)
        model.model_type = 'classifier' # on ne test pas 'regressor', pareil
        model.save(json_data={'test': 8})
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        # self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5'))) -> pas de modèle trained
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, f"{model.model_name}.pkl")))
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
        self.assertTrue('list_classes' not in configs.keys()) # not in car on n'utilise pas le mixin Classifier
        self.assertTrue('dict_classes' not in configs.keys()) # not in car on n'utilise pas le mixin Classifier
        self.assertTrue('multi_label' not in configs.keys()) # not in car on n'utilise pas le mixin Classifier
        remove_dir(model_dir)

        # Fonctionnement avec custom_objects qui contient une fonction "partial"
        model = ModelKeras(model_dir=model_dir)
        model.model_type = 'classifier' # on ne test pas 'regressor', pareil
        custom_objects = utils_deep_keras.custom_objects
        custom_objects['fb_loss'] = utils_deep_keras.get_fb_loss(0.5)
        model.custom_objects = custom_objects
        model.save(json_data={'test': 8})
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        # self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5'))) -> pas de modèle trained
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, f"{model.model_name}.pkl")))
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
        self.assertTrue('list_classes' not in configs.keys()) # not in car on n'utilise pas le mixin Classifier
        self.assertTrue('dict_classes' not in configs.keys()) # not in car on n'utilise pas le mixin Classifier
        self.assertTrue('multi_label' not in configs.keys()) # not in car on n'utilise pas le mixin Classifier
        remove_dir(model_dir)


    def test08_model_keras_reload_model(self):
        '''Test de la fonction reload_model de ynov.models_training.model_keras.ModelKeras'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Set vars
        x_train = pd.DataFrame({'col_1': [-5, -1, 0, -2, 2, -6, 3] * 10, 'col_2': [2, -1, -8, 2, 3, 12, 2] * 10})
        y_train_mono_2 = pd.Series([0, 0, 0, 0, 1, 1, 1] * 10)
        y_train_mono_3 = pd.Series([0, 0, 0, 2, 1, 1, 1] * 10)
        y_train_regressor = pd.Series([-3, -2, -8, 0, 5, 6, 5] * 10)
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

        # # Regressor
        model = ModelDenseRegressor(model_dir=model_dir, batch_size=8, epochs=2)
        model.fit(x_train, y_train_regressor)
        model.save()
        # Reload keras
        hdf5_path = os.path.join(model.model_dir, 'best.hdf5')
        reloaded_model = model.reload_model(hdf5_path)
        np.testing.assert_almost_equal([list(_) for _ in reloaded_model.predict(x_train)], [[_] for _ in model.predict(x_train)], 5)
        # Test sans custom_objects
        model.custom_objects = None
        reloaded_model = model.reload_model(hdf5_path)
        np.testing.assert_almost_equal([list(_) for _ in reloaded_model.predict(x_train)], [[_] for _ in model.predict(x_train)], 5)
        # Clean
        remove_dir(model_dir)


# Execution des tests
if __name__ == '__main__':
    # Start tests
    unittest.main()