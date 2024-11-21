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
from ynov.models_training.classifiers.model_rf_classifier import ModelRFClassifier

# Disable logging
import logging
logging.disable(logging.CRITICAL)


def remove_dir(path):
    if os.path.isdir(path): shutil.rmtree(path)


class ModelRFClassifierTests(unittest.TestCase):
    '''Main class to test model_rf_classifier'''


    def setUp(self):
        '''SetUp fonction'''
        # On se place dans le bon répertoire
        # Change directory to script directory
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname)


    def test01_model_rf_classifier_init(self):
        '''Test de ynov.models_training.classifiers.model_rf_classifier.ModelRFClassifier.__init__'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Init., test all params
        model = ModelRFClassifier(model_dir=model_dir)
        self.assertEqual(model.model_dir, model_dir)
        self.assertTrue(os.path.isdir(model_dir))
        self.assertTrue(model.pipeline is not None)
        self.assertEqual(model.model_type, 'classifier')
        self.assertTrue(model.multiclass_strategy is None)
        # On fait juste un appel à display_if_gpu_activated et _is_gpu_activated
        self.assertTrue(type(model._is_gpu_activated()) == bool)
        model.display_if_gpu_activated()
        remove_dir(model_dir)

        # Check RF params en fonction multi-label & multiclass strategy
        model = ModelRFClassifier(model_dir=model_dir, multi_label=False, rf_params={'max_depth': 8, 'n_estimators': 10})
        self.assertEqual(model.pipeline['rf'].max_depth, 8)
        self.assertEqual(model.pipeline['rf'].n_estimators, 10)
        remove_dir(model_dir)
        model = ModelRFClassifier(model_dir=model_dir, multi_label=False, multiclass_strategy='ovr', rf_params={'max_depth': 8, 'n_estimators': 10})
        self.assertEqual(model.multiclass_strategy, 'ovr')
        self.assertEqual(model.pipeline['rf'].estimator.max_depth, 8)
        self.assertEqual(model.pipeline['rf'].estimator.n_estimators, 10)
        remove_dir(model_dir)
        model = ModelRFClassifier(model_dir=model_dir, multi_label=False, multiclass_strategy='ovo', rf_params={'max_depth': 8, 'n_estimators': 10})
        self.assertEqual(model.multiclass_strategy, 'ovo')
        self.assertEqual(model.pipeline['rf'].estimator.max_depth, 8)
        self.assertEqual(model.pipeline['rf'].estimator.n_estimators, 10)
        remove_dir(model_dir)
        #
        model = ModelRFClassifier(model_dir=model_dir, multi_label=True, rf_params={'max_depth': 8, 'n_estimators': 10})
        self.assertEqual(model.pipeline['rf'].max_depth, 8)
        self.assertEqual(model.pipeline['rf'].n_estimators, 10)
        self.assertEqual(model.multi_label, True)
        remove_dir(model_dir)
        model = ModelRFClassifier(model_dir=model_dir, multi_label=True, multiclass_strategy='ovr', rf_params={'max_depth': 8, 'n_estimators': 10})
        self.assertEqual(model.multiclass_strategy, 'ovr')
        self.assertEqual(model.pipeline['rf'].max_depth, 8)
        self.assertEqual(model.pipeline['rf'].n_estimators, 10)
        self.assertEqual(model.multi_label, True)
        remove_dir(model_dir)
        model = ModelRFClassifier(model_dir=model_dir, multi_label=True, multiclass_strategy='ovo', rf_params={'max_depth': 8, 'n_estimators': 10})
        self.assertEqual(model.multiclass_strategy, 'ovo')
        self.assertEqual(model.pipeline['rf'].max_depth, 8)
        self.assertEqual(model.pipeline['rf'].n_estimators, 10)
        self.assertEqual(model.multi_label, True)
        remove_dir(model_dir)

        # Error
        with self.assertRaises(ValueError):
            model = ModelRFClassifier(model_dir=model_dir, multi_label=False, multiclass_strategy='toto', rf_params={'max_depth': 8, 'n_estimators': 10})
        remove_dir(model_dir)

    def test02_model_rf_classifier_predict(self):
        '''Test de la fonction predict de ynov.models_training.classifiers.model_rf_classifier.ModelRFClassifier'''

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
        model = ModelRFClassifier(x_col=x_col, y_col=y_col_mono, model_dir=model_dir)
        model.fit(x_train, y_train_mono_2)
        preds = model.predict(x_train, return_proba=False)
        self.assertEqual(preds.shape, (len(x_train),))
        proba = model.predict(x_train, return_proba=True)
        self.assertEqual(proba.shape, (len(x_train), 2)) # 2 classes
        remove_dir(model_dir)
        model = ModelRFClassifier(x_col=x_col, y_col=y_col_mono, model_dir=model_dir, multiclass_strategy='ovr')
        model.fit(x_train, y_train_mono_2)
        preds = model.predict(x_train, return_proba=False)
        self.assertEqual(preds.shape, (len(x_train),))
        proba = model.predict(x_train, return_proba=True)
        self.assertEqual(proba.shape, (len(x_train), 2)) # 2 classes
        remove_dir(model_dir)
        model = ModelRFClassifier(x_col=x_col, y_col=y_col_mono, model_dir=model_dir, multiclass_strategy='ovo')
        model.fit(x_train, y_train_mono_2)
        preds = model.predict(x_train, return_proba=False)
        self.assertEqual(preds.shape, (len(x_train),))
        proba = model.predict(x_train, return_proba=True)
        self.assertEqual(proba.shape, (len(x_train), 2)) # 2 classes
        remove_dir(model_dir)

        # Classification - Mono label - Multi Class
        model = ModelRFClassifier(x_col=x_col, y_col=y_col_mono, model_dir=model_dir)
        model.fit(x_train, y_train_mono_3)
        preds = model.predict(x_train, return_proba=False)
        self.assertEqual(preds.shape, (len(x_train),))
        proba = model.predict(x_train, return_proba=True)
        self.assertEqual(proba.shape, (len(x_train), 3)) # 3 classes
        remove_dir(model_dir)
        model = ModelRFClassifier(x_col=x_col, y_col=y_col_mono, model_dir=model_dir, multiclass_strategy='ovr')
        model.fit(x_train, y_train_mono_3)
        preds = model.predict(x_train, return_proba=False)
        self.assertEqual(preds.shape, (len(x_train),))
        proba = model.predict(x_train, return_proba=True)
        self.assertEqual(proba.shape, (len(x_train), 3)) # 3 classes
        remove_dir(model_dir)
        model = ModelRFClassifier(x_col=x_col, y_col=y_col_mono, model_dir=model_dir, multiclass_strategy='ovo')
        model.fit(x_train, y_train_mono_3)
        preds = model.predict(x_train, return_proba=False)
        self.assertEqual(preds.shape, (len(x_train),))
        proba = model.predict(x_train, return_proba=True)
        self.assertEqual(proba.shape, (len(x_train), 3)) # 3 classes
        remove_dir(model_dir)

        # Classification - Multi label
        model = ModelRFClassifier(x_col=x_col, y_col=y_col_multi, model_dir=model_dir, multi_label=True)
        model.fit(x_train, y_train_multi)
        preds = model.predict(x_train)
        self.assertEqual(preds.shape, (len(x_train), len(y_col_multi)))
        proba = model.predict(x_train, return_proba=True)
        self.assertEqual(proba.shape, (len(x_train), len(y_col_multi)))
        remove_dir(model_dir)
        model = ModelRFClassifier(x_col=x_col, y_col=y_col_multi, model_dir=model_dir, multi_label=True, multiclass_strategy='ovr')
        model.fit(x_train, y_train_multi)
        preds = model.predict(x_train)
        self.assertEqual(preds.shape, (len(x_train), len(y_col_multi)))
        proba = model.predict(x_train, return_proba=True)
        self.assertEqual(proba.shape, (len(x_train), len(y_col_multi)))
        remove_dir(model_dir)
        model = ModelRFClassifier(x_col=x_col, y_col=y_col_multi, model_dir=model_dir, multi_label=True, multiclass_strategy='ovo')
        model.fit(x_train, y_train_multi)
        preds = model.predict(x_train)
        self.assertEqual(preds.shape, (len(x_train), len(y_col_multi)))
        proba = model.predict(x_train, return_proba=True)
        self.assertEqual(proba.shape, (len(x_train), len(y_col_multi)))
        remove_dir(model_dir)

        # Model needs to be fitted
        with self.assertRaises(AttributeError):
            model = ModelRFClassifier(x_col=x_col, y_col=y_col_mono, model_dir=model_dir)
            model.predict(pd.Series([-2, 3]))
        remove_dir(model_dir)

    def test03_model_rf_classifier_predict_proba(self):
        '''Test de la fonction predict_proba de ynov.models_training.classifiers.model_rf_classifier.ModelRFClassifier'''

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
        model = ModelRFClassifier(x_col=x_col, y_col=y_col_mono, model_dir=model_dir)
        model.fit(x_train, y_train_mono_2)
        preds = model.predict_proba(x_train)
        self.assertEqual(preds.shape, (len(x_train), 2)) # 2 classes
        self.assertTrue(isinstance(preds[0][0], (np.floating, float)))
        remove_dir(model_dir)
        model = ModelRFClassifier(x_col=x_col, y_col=y_col_mono, model_dir=model_dir, multiclass_strategy='ovr')
        model.fit(x_train, y_train_mono_2)
        preds = model.predict_proba(x_train)
        self.assertEqual(preds.shape, (len(x_train), 2)) # 2 classes
        self.assertTrue(isinstance(preds[0][0], (np.floating, float)))
        remove_dir(model_dir)
        model = ModelRFClassifier(x_col=x_col, y_col=y_col_mono, model_dir=model_dir, multiclass_strategy='ovo')
        model.fit(x_train, y_train_mono_2)
        preds = model.predict_proba(x_train)
        self.assertEqual(preds.shape, (len(x_train), 2)) # 2 classes
        self.assertTrue(isinstance(preds[0][0], (np.floating, float)))
        remove_dir(model_dir)

        # Classification - Mono label - Multi Class
        model = ModelRFClassifier(x_col=x_col, y_col=y_col_mono, model_dir=model_dir)
        model.fit(x_train, y_train_mono_3)
        preds = model.predict_proba(x_train)
        self.assertEqual(preds.shape, (len(x_train), 3)) # 3 classes
        self.assertTrue(isinstance(preds[0][0], (np.floating, float)))
        remove_dir(model_dir)
        model = ModelRFClassifier(x_col=x_col, y_col=y_col_mono, model_dir=model_dir, multiclass_strategy='ovr')
        model.fit(x_train, y_train_mono_3)
        preds = model.predict_proba(x_train)
        self.assertEqual(preds.shape, (len(x_train), 3)) # 3 classes
        self.assertTrue(isinstance(preds[0][0], (np.floating, float)))
        remove_dir(model_dir)
        model = ModelRFClassifier(x_col=x_col, y_col=y_col_mono, model_dir=model_dir, multiclass_strategy='ovo')
        model.fit(x_train, y_train_mono_3)
        preds = model.predict_proba(x_train)
        self.assertEqual(preds.shape, (len(x_train), 3)) # 3 classes
        self.assertTrue(isinstance(preds[0][0], (np.floating, float)))
        remove_dir(model_dir)

        # Classification - Multi label
        model = ModelRFClassifier(x_col=x_col, y_col=y_col_multi, model_dir=model_dir, multi_label=True)
        model.fit(x_train, y_train_multi)
        preds = model.predict_proba(x_train)
        self.assertEqual(preds.shape, (len(x_train), len(y_col_multi))) # 3 labels
        self.assertTrue(isinstance(preds[0][0], (np.floating, float)))
        remove_dir(model_dir)
        model = ModelRFClassifier(x_col=x_col, y_col=y_col_multi, model_dir=model_dir, multi_label=True, multiclass_strategy='ovr')
        model.fit(x_train, y_train_multi)
        preds = model.predict_proba(x_train)
        self.assertEqual(preds.shape, (len(x_train), len(y_col_multi))) # 3 labels
        self.assertTrue(isinstance(preds[0][0], (np.floating, float)))
        remove_dir(model_dir)
        model = ModelRFClassifier(x_col=x_col, y_col=y_col_multi, model_dir=model_dir, multi_label=True, multiclass_strategy='ovo')
        model.fit(x_train, y_train_multi)
        preds = model.predict_proba(x_train)
        self.assertEqual(preds.shape, (len(x_train), len(y_col_multi))) # 3 labels
        self.assertTrue(isinstance(preds[0][0], (np.floating, float)))
        remove_dir(model_dir)

        # Model needs to be fitted
        with self.assertRaises(AttributeError):
            model = ModelRFClassifier(x_col=x_col, y_col=y_col_mono, model_dir=model_dir)
            model.predict_proba('test')
        remove_dir(model_dir)

    def test04_model_rf_classifier_get_predict_position(self):
        '''Test de la fonction ynov.models_training.classifiers.model_rf_classifier.ModelRFClassifier.get_predict_position'''

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
        model = ModelRFClassifier(x_col=x_col, y_col=y_col_mono, model_dir=model_dir)
        model.fit(x_train, y_train_mono_2)
        predict_positions = model.get_predict_position(x_train, y_train_mono_2)
        self.assertEqual(predict_positions.shape, (len(x_train),))
        remove_dir(model_dir)
        model = ModelRFClassifier(x_col=x_col, y_col=y_col_mono, model_dir=model_dir, multiclass_strategy='ovr')
        model.fit(x_train, y_train_mono_2)
        predict_positions = model.get_predict_position(x_train, y_train_mono_2)
        self.assertEqual(predict_positions.shape, (len(x_train),))
        remove_dir(model_dir)
        model = ModelRFClassifier(x_col=x_col, y_col=y_col_mono, model_dir=model_dir, multiclass_strategy='ovo')
        model.fit(x_train, y_train_mono_2)
        predict_positions = model.get_predict_position(x_train, y_train_mono_2)
        self.assertEqual(predict_positions.shape, (len(x_train),))
        remove_dir(model_dir)

        # Classification - Mono label - Multi Class
        model = ModelRFClassifier(x_col=x_col, y_col=y_col_mono, model_dir=model_dir)
        model.fit(x_train, y_train_mono_3)
        predict_positions = model.get_predict_position(x_train, y_train_mono_2)
        self.assertEqual(predict_positions.shape, (len(x_train),))
        remove_dir(model_dir)
        model = ModelRFClassifier(x_col=x_col, y_col=y_col_mono, model_dir=model_dir, multiclass_strategy='ovr')
        model.fit(x_train, y_train_mono_3)
        predict_positions = model.get_predict_position(x_train, y_train_mono_2)
        self.assertEqual(predict_positions.shape, (len(x_train),))
        remove_dir(model_dir)
        model = ModelRFClassifier(x_col=x_col, y_col=y_col_mono, model_dir=model_dir, multiclass_strategy='ovo')
        model.fit(x_train, y_train_mono_3)
        predict_positions = model.get_predict_position(x_train, y_train_mono_2)
        self.assertEqual(predict_positions.shape, (len(x_train),))
        remove_dir(model_dir)

        # Classification - Multi label
        model = ModelRFClassifier(x_col=x_col, y_col=y_col_multi, model_dir=model_dir, multi_label=True)
        model.fit(x_train, y_train_multi)        # Pas dispo en multi-label
        with self.assertRaises(ValueError):
            model.get_predict_position(x_train, y_train_multi)
        remove_dir(model_dir)
        model = ModelRFClassifier(x_col=x_col, y_col=y_col_multi, model_dir=model_dir, multi_label=True, multiclass_strategy='ovr')
        model.fit(x_train, y_train_multi)        # Pas dispo en multi-label
        with self.assertRaises(ValueError):
            model.get_predict_position(x_train, y_train_multi)
        remove_dir(model_dir)
        model = ModelRFClassifier(x_col=x_col, y_col=y_col_multi, model_dir=model_dir, multi_label=True, multiclass_strategy='ovo')
        model.fit(x_train, y_train_multi)        # Pas dispo en multi-label
        with self.assertRaises(ValueError):
            model.get_predict_position(x_train, y_train_multi)
        remove_dir(model_dir)

    def test05_model_rf_classifier_save(self):
        '''Test de la fonction save de ynov.models_training.classifiers.model_rf_classifier.ModelRFClassifier'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Fonctionnement nominal
        model = ModelRFClassifier(model_dir=model_dir, multi_label=False, multiclass_strategy='ovr')
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
        self.assertEqual(configs['model_type'], 'classifier')
        self.assertTrue('trained' in configs.keys())
        self.assertTrue('nb_fit' in configs.keys())
        self.assertTrue('x_col' in configs.keys())
        self.assertTrue('y_col' in configs.keys())
        self.assertTrue('columns_in' in configs.keys())
        self.assertTrue('mandatory_columns' in configs.keys())
        self.assertTrue('level_save' in configs.keys())
        self.assertTrue('librairie' in configs.keys())
        self.assertEqual(configs['librairie'], 'scikit-learn')
        self.assertTrue('multiclass_strategy' in configs.keys())
        self.assertEqual(configs['multiclass_strategy'], 'ovr')
        self.assertTrue('list_classes' in configs.keys())
        self.assertTrue('dict_classes' in configs.keys())
        self.assertTrue('multi_label' in configs.keys())
        # Spécifique model utilisé
        self.assertTrue('rf_confs' in configs.keys())
        remove_dir(model_dir)

    def test06_model_rf_classifier_reload_from_standalone(self):
        '''Test de la fonction ynov.models_training.classifiers.model_rf_classifier.ModelRFClassifier.reload_from_standalone'''

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
        model = ModelRFClassifier(x_col=x_col, y_col=y_col_mono, model_dir=model_dir)
        rf = model.rf
        model.fit(x_train, y_train_mono_2)
        model.save()
        # Reload
        pkl_path = os.path.join(model.model_dir, f"{model.model_name}_standalone.pkl")
        conf_path = os.path.join(model.model_dir, "configurations.json")
        preprocess_pipeline_path = os.path.join(model.model_dir, "preprocess_pipeline.pkl")
        new_model = ModelRFClassifier()
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
        self.assertEqual(model.list_classes, new_model.list_classes)
        self.assertEqual(model.dict_classes, new_model.dict_classes)
        self.assertEqual(model.multi_label, new_model.multi_label)
        self.assertEqual(model.multiclass_strategy, new_model.multiclass_strategy)
        self.assertEqual(model.rf.get_params(), rf.get_params())
        self.assertTrue(new_model.preprocess_pipeline is not None)
        # On ne peut pas vraiment tester la pipeline, du coup on test les predictions
        self.assertEqual([list(_) for _ in model.predict_proba(x_train)], [list(_) for _ in new_model.predict_proba(x_train)])
        remove_dir(model_dir)
        remove_dir(new_model.model_dir)

        # multiclass_strategy 'ovr'
        # Create model
        model = ModelRFClassifier(x_col=x_col, y_col=y_col_mono, model_dir=model_dir, multiclass_strategy='ovr')
        rf = model.rf
        model.fit(x_train, y_train_mono_2)
        model.save()
        # Reload
        pkl_path = os.path.join(model.model_dir, f"{model.model_name}_standalone.pkl")
        conf_path = os.path.join(model.model_dir, "configurations.json")
        preprocess_pipeline_path = os.path.join(model.model_dir, "preprocess_pipeline.pkl")
        new_model = ModelRFClassifier()
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
        self.assertEqual(model.list_classes, new_model.list_classes)
        self.assertEqual(model.dict_classes, new_model.dict_classes)
        self.assertEqual(model.multi_label, new_model.multi_label)
        self.assertEqual(model.multiclass_strategy, new_model.multiclass_strategy)
        self.assertEqual(model.rf.get_params(), rf.get_params())
        self.assertTrue(new_model.preprocess_pipeline is not None)
        # On ne peut pas vraiment tester la pipeline, du coup on test les predictions
        self.assertEqual([list(_) for _ in model.predict_proba(x_train)], [list(_) for _ in new_model.predict_proba(x_train)])
        remove_dir(model_dir)
        remove_dir(new_model.model_dir)

        # multiclass_strategy 'ovo'
        # Create model
        model = ModelRFClassifier(x_col=x_col, y_col=y_col_mono, model_dir=model_dir, multiclass_strategy='ovo')
        rf = model.rf
        model.fit(x_train, y_train_mono_2)
        model.save()
        # Reload
        pkl_path = os.path.join(model.model_dir, f"{model.model_name}_standalone.pkl")
        conf_path = os.path.join(model.model_dir, "configurations.json")
        preprocess_pipeline_path = os.path.join(model.model_dir, "preprocess_pipeline.pkl")
        new_model = ModelRFClassifier()
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
        self.assertEqual(model.list_classes, new_model.list_classes)
        self.assertEqual(model.dict_classes, new_model.dict_classes)
        self.assertEqual(model.multi_label, new_model.multi_label)
        self.assertEqual(model.multiclass_strategy, new_model.multiclass_strategy)
        self.assertEqual(model.rf.get_params(), rf.get_params())
        self.assertTrue(new_model.preprocess_pipeline is not None)
        # On ne peut pas vraiment tester la pipeline, du coup on test les predictions
        self.assertEqual([list(_) for _ in model.predict_proba(x_train)], [list(_) for _ in new_model.predict_proba(x_train)])
        remove_dir(model_dir)
        remove_dir(new_model.model_dir)

        ############################################
        # Classification - Multi label
        ############################################

        # Create model
        model = ModelRFClassifier(x_col=x_col, y_col=y_col_multi, model_dir=model_dir, multi_label=True)
        rf = model.rf
        model.fit(x_train, y_train_multi)
        model.save()
        # Reload
        pkl_path = os.path.join(model.model_dir, f"{model.model_name}_standalone.pkl")
        conf_path = os.path.join(model.model_dir, "configurations.json")
        preprocess_pipeline_path = os.path.join(model.model_dir, "preprocess_pipeline.pkl")
        new_model = ModelRFClassifier()
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
        self.assertEqual(model.list_classes, new_model.list_classes)
        self.assertEqual(model.dict_classes, new_model.dict_classes)
        self.assertEqual(model.multi_label, new_model.multi_label)
        self.assertEqual(model.multiclass_strategy, new_model.multiclass_strategy)
        self.assertEqual(model.rf.get_params(), rf.get_params())
        self.assertTrue(new_model.preprocess_pipeline is not None)
        # On ne peut pas vraiment tester la pipeline, du coup on test les predictions
        self.assertEqual([list(_) for _ in model.predict_proba(x_train)], [list(_) for _ in new_model.predict_proba(x_train)])
        remove_dir(model_dir)
        remove_dir(new_model.model_dir)

        # multiclass_strategy 'ovr'
        # Create model
        model = ModelRFClassifier(x_col=x_col, y_col=y_col_multi, model_dir=model_dir, multi_label=True, multiclass_strategy='ovr')
        rf = model.rf
        model.fit(x_train, y_train_multi)
        model.save()
        # Reload
        pkl_path = os.path.join(model.model_dir, f"{model.model_name}_standalone.pkl")
        conf_path = os.path.join(model.model_dir, "configurations.json")
        preprocess_pipeline_path = os.path.join(model.model_dir, "preprocess_pipeline.pkl")
        new_model = ModelRFClassifier()
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
        self.assertEqual(model.list_classes, new_model.list_classes)
        self.assertEqual(model.dict_classes, new_model.dict_classes)
        self.assertEqual(model.multi_label, new_model.multi_label)
        self.assertEqual(model.multiclass_strategy, new_model.multiclass_strategy)
        self.assertEqual(model.rf.get_params(), rf.get_params())
        self.assertTrue(new_model.preprocess_pipeline is not None)
        # On ne peut pas vraiment tester la pipeline, du coup on test les predictions
        self.assertEqual([list(_) for _ in model.predict_proba(x_train)], [list(_) for _ in new_model.predict_proba(x_train)])
        remove_dir(model_dir)
        remove_dir(new_model.model_dir)

        # multiclass_strategy 'ovo'
        # Create model
        model = ModelRFClassifier(x_col=x_col, y_col=y_col_multi, model_dir=model_dir, multi_label=True, multiclass_strategy='ovo')
        rf = model.rf
        model.fit(x_train, y_train_multi)
        model.save()
        # Reload
        pkl_path = os.path.join(model.model_dir, f"{model.model_name}_standalone.pkl")
        conf_path = os.path.join(model.model_dir, "configurations.json")
        preprocess_pipeline_path = os.path.join(model.model_dir, "preprocess_pipeline.pkl")
        new_model = ModelRFClassifier()
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
        self.assertEqual(model.list_classes, new_model.list_classes)
        self.assertEqual(model.dict_classes, new_model.dict_classes)
        self.assertEqual(model.multi_label, new_model.multi_label)
        self.assertEqual(model.multiclass_strategy, new_model.multiclass_strategy)
        self.assertEqual(model.rf.get_params(), rf.get_params())
        self.assertTrue(new_model.preprocess_pipeline is not None)
        # On ne peut pas vraiment tester la pipeline, du coup on test les predictions
        self.assertEqual([list(_) for _ in model.predict_proba(x_train)], [list(_) for _ in new_model.predict_proba(x_train)])
        remove_dir(new_model.model_dir)
        # On ne remove pas model_dir pour tester les erreurs

        ############################################
        # Errors
        ############################################

        with self.assertRaises(FileNotFoundError):
            new_model = ModelRFClassifier()
            new_model.reload_from_standalone(configuration_path='toto.json', model_pipeline_path=pkl_path,
                                             preprocess_pipeline_path=preprocess_pipeline_path)
        with self.assertRaises(FileNotFoundError):
            new_model = ModelRFClassifier()
            new_model.reload_from_standalone(configuration_path=conf_path, model_pipeline_path='toto.pkl',
                                             preprocess_pipeline_path=preprocess_pipeline_path)
        with self.assertRaises(FileNotFoundError):
            new_model = ModelRFClassifier()
            new_model.reload_from_standalone(configuration_path=conf_path, model_pipeline_path=pkl_path,
                                             preprocess_pipeline_path='toto.pkl')

        # Clean
        remove_dir(model_dir)


# Execution des tests
if __name__ == '__main__':
    # Start tests
    unittest.main()