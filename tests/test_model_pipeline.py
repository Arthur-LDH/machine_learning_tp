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
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from ynov import utils
from ynov.models_training.model_pipeline import ModelPipeline

# Disable logging
import logging
logging.disable(logging.CRITICAL)


def remove_dir(path):
    if os.path.isdir(path): shutil.rmtree(path)


class ModelPipelineTests(unittest.TestCase):
    '''Main class to test model_pipeline'''


    def setUp(self):
        '''SetUp fonction'''
        # On se place dans le bon répertoire
        # Change directory to script directory
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname)


    def test01_model_pipeline_init(self):
        '''Test de l'initialisation de ynov.models_training.model_pipeline.ModelPipeline'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Init., test all params
        model = ModelPipeline(model_dir=model_dir)
        self.assertEqual(model.model_dir, model_dir)
        self.assertEqual(model.model_type, None)
        self.assertTrue(os.path.isdir(model_dir))
        self.assertEqual(model.pipeline, None)
        # On fait juste un appel à display_if_gpu_activated et _is_gpu_activated
        self.assertTrue(type(model._is_gpu_activated()) == bool)
        model.display_if_gpu_activated()
        remove_dir(model_dir)

        random_forest = RandomForestClassifier()
        pipeline = Pipeline([('rf', random_forest)])
        model = ModelPipeline(model_dir=model_dir, pipeline=pipeline)
        model.save()
        self.assertEqual(model.pipeline, pipeline)
        remove_dir(model_dir)


    def test02_model_pipeline_fit(self):
        '''Test de la fonction fit de ynov.models_training.model_pipeline.ModelPipeline'''

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
        rf_classifier = RandomForestClassifier()
        pipeline = Pipeline([('rf', rf_classifier)])
        model = ModelPipeline(x_col=x_col, y_col=y_col_mono, model_dir=model_dir, pipeline=pipeline)
        # On simule un classifier mono label
        model.model_type = 'classifier'
        model.multi_label = False
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        self.assertEqual(model.x_col, x_col)
        self.assertEqual(model.y_col, y_col_mono)
        self.assertFalse(hasattr(model.pipeline['rf'], "classes_"))
        model.fit(x_train, y_train_mono_2)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(model.x_col, x_col)
        self.assertEqual(model.y_col, y_col_mono)
        self.assertTrue(hasattr(model.pipeline['rf'], "classes_"))
        self.assertEqual(model.list_classes, [0, 1])
        self.assertEqual(model.dict_classes, {0: 0, 1: 1})
        remove_dir(model_dir)

        # Classification - Mono label - Multi Class
        rf_classifier = RandomForestClassifier()
        pipeline = Pipeline([('rf', rf_classifier)])
        model = ModelPipeline(x_col=x_col, y_col=y_col_mono, model_dir=model_dir, pipeline=pipeline)
        # On simule un classifier mono label
        model.model_type = 'classifier'
        model.multi_label = False
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        self.assertEqual(model.x_col, x_col)
        self.assertEqual(model.y_col, y_col_mono)
        self.assertFalse(hasattr(model.pipeline['rf'], "classes_"))
        model.fit(x_train, y_train_mono_3)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(model.x_col, x_col)
        self.assertEqual(model.y_col, y_col_mono)
        self.assertTrue(hasattr(model.pipeline['rf'], "classes_"))
        self.assertEqual(model.list_classes, [0, 1, 2])
        self.assertEqual(model.dict_classes, {0: 0, 1: 1, 2: 2})
        remove_dir(model_dir)

        # Classification - Multi Label
        rf_classifier = RandomForestClassifier()
        pipeline = Pipeline([('rf', rf_classifier)])
        # On vérifie aussi sans x_col & y_col
        model = ModelPipeline(model_dir=model_dir, pipeline=pipeline)
        # On simule un classifier multi label
        model.model_type = 'classifier'
        model.multi_label = True
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        self.assertEqual(model.x_col, None)
        self.assertEqual(model.y_col, None)
        self.assertFalse(hasattr(model.pipeline, "classes_"))
        model.fit(x_train, y_train_multi)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(model.x_col, x_col)
        self.assertEqual(model.y_col, y_col_multi)
        self.assertTrue(hasattr(model.pipeline, "classes_"))
        self.assertEqual(model.list_classes, y_col_multi)
        self.assertEqual(model.dict_classes, {0: 'y1', 1: 'y2', 2: 'y3'})
        remove_dir(model_dir)

        # Regressor
        rf_regressor = RandomForestRegressor()
        pipeline = Pipeline([('rf', rf_regressor)])
        model = ModelPipeline(x_col=x_col, y_col=y_col_mono, model_dir=model_dir, pipeline=pipeline)
        # On simule un regressor
        model.model_type = 'regressor'
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        self.assertEqual(model.x_col, x_col)
        self.assertEqual(model.y_col, y_col_mono)
        self.assertFalse(hasattr(model.pipeline['rf'], "feature_importances_"))
        model.fit(x_train, y_train_regressor)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(model.x_col, x_col)
        self.assertEqual(model.y_col, y_col_mono)
        self.assertTrue(hasattr(model.pipeline['rf'], "feature_importances_"))
        self.assertFalse(hasattr(model, "list_classes"))
        self.assertFalse(hasattr(model, "dict_classes"))
        remove_dir(model_dir)
        #
        ############
        # Test continue training
        rf_classifier = RandomForestClassifier()
        pipeline = Pipeline([('rf', rf_classifier)])
        model = ModelPipeline(x_col=x_col, y_col=y_col_mono, model_dir=model_dir, pipeline=pipeline)
        # On simule un classifier mono label
        model.model_type = 'classifier'
        model.multi_label = False
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        self.assertFalse(hasattr(model.pipeline['rf'], "classes_"))
        model.fit(x_train, y_train_mono_2)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertTrue(hasattr(model.pipeline['rf'], "classes_"))
        # second fit
        with self.assertRaises(RuntimeError):
            model.fit(x_train[:50], y_train_mono_2[:50])
        self.assertEqual(model_dir, model.model_dir)
        remove_dir(model_dir)

        # Gestion des erreurs
        rf_classifier = RandomForestClassifier()
        pipeline = Pipeline([('rf', rf_classifier)])
        model = ModelPipeline(x_col=x_col, y_col=y_col_mono, model_dir=model_dir, pipeline=pipeline)
        with self.assertRaises(ValueError):
            model.fit(x_train, y_train_mono_2)
        remove_dir(model_dir)


    def test03_model_pipeline_predict(self):
        '''Test de la fonction predict de ynov.models_training.model_pipeline.ModelPipeline'''

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
        rf_classifier = RandomForestClassifier(n_estimators=10)
        pipeline = Pipeline([('rf', rf_classifier)])
        model = ModelPipeline(x_col=x_col, y_col=y_col_mono, model_dir=model_dir, pipeline=pipeline)
        # On simule un classifier mono label
        model.model_type = 'classifier'
        model.multi_label = False
        model.fit(x_train, y_train_mono_2)
        preds = model.predict(x_train, return_proba=False)
        self.assertEqual(preds.shape, (len(x_train),))
        proba = model.predict(x_train, return_proba=True)
        self.assertEqual(proba.shape, (len(x_train), 2)) # 2 classes
        remove_dir(model_dir)

        # Classification - Mono label - Multi Class
        rf_classifier = RandomForestClassifier(n_estimators=10)
        pipeline = Pipeline([('rf', rf_classifier)])
        model = ModelPipeline(x_col=x_col, y_col=y_col_mono, model_dir=model_dir, pipeline=pipeline)
        # On simule un classifier mono label
        model.model_type = 'classifier'
        model.multi_label = False
        model.fit(x_train, y_train_mono_3)
        preds = model.predict(x_train, return_proba=False)
        self.assertEqual(preds.shape, (len(x_train),))
        proba = model.predict(x_train, return_proba=True)
        self.assertEqual(proba.shape, (len(x_train), 3)) # 3 classes
        remove_dir(model_dir)

        # Classification - Multi label
        rf_classifier = RandomForestClassifier(n_estimators=10)
        pipeline = Pipeline([('rf', rf_classifier)])
        model = ModelPipeline(x_col=x_col, y_col=y_col_multi, model_dir=model_dir, pipeline=pipeline)
        # On simule un classifier multi label
        model.model_type = 'classifier'
        model.multi_label = True
        model.fit(x_train, y_train_multi)
        preds = model.predict(x_train)
        self.assertEqual(preds.shape, (len(x_train), len(y_col_multi)))
        proba = model.predict(x_train, return_proba=True)
        self.assertEqual(proba.shape, (len(x_train), len(y_col_multi)))
        remove_dir(model_dir)

        # Regressor
        rf_regressor = RandomForestRegressor(n_estimators=10)
        pipeline = Pipeline([('rf', rf_regressor)])
        model = ModelPipeline(x_col=x_col, y_col=y_col_mono, model_dir=model_dir, pipeline=pipeline)
        # On simule un regressor
        model.model_type = 'regressor'
        model.fit(x_train, y_train_regressor)
        preds = model.predict(x_train)
        self.assertEqual(preds.shape, (len(x_train),))
        with self.assertRaises(ValueError):
            proba = model.predict(x_train, return_proba=True)
        remove_dir(model_dir)

        # Model needs to be fitted
        with self.assertRaises(AttributeError):
            model = ModelPipeline(x_col=x_col, y_col=y_col_mono, model_dir=model_dir, pipeline=pipeline)
            model.predict(pd.Series([-2, 3]))
        remove_dir(model_dir)


    def test04_model_pipeline_predict_proba(self):
        '''Test de la fonction predict_proba de ynov.models_training.model_pipeline.ModelPipeline'''

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
        rf_classifier = RandomForestClassifier(n_estimators=10)
        pipeline = Pipeline([('rf', rf_classifier)])
        model = ModelPipeline(x_col=x_col, y_col=y_col_mono, model_dir=model_dir, pipeline=pipeline)
        # On simule un classifier mono label
        model.model_type = 'classifier'
        model.multi_label = False
        model.fit(x_train, y_train_mono_2)
        preds = model.predict_proba(x_train)
        self.assertEqual(preds.shape, (len(x_train), 2)) # 2 classes
        self.assertTrue(isinstance(preds[0][0], (np.floating, float)))
        remove_dir(model_dir)

        # Classification - Mono label - Multi Class
        rf_classifier = RandomForestClassifier(n_estimators=10)
        pipeline = Pipeline([('rf', rf_classifier)])
        model = ModelPipeline(x_col=x_col, y_col=y_col_mono, model_dir=model_dir, pipeline=pipeline)
        # On simule un classifier mono label
        model.model_type = 'classifier'
        model.multi_label = False
        model.fit(x_train, y_train_mono_3)
        preds = model.predict_proba(x_train)
        self.assertEqual(preds.shape, (len(x_train), 3)) # 3 classes
        self.assertTrue(isinstance(preds[0][0], (np.floating, float)))
        remove_dir(model_dir)

        # Classification - Multi label
        rf_classifier = RandomForestClassifier(n_estimators=10)
        pipeline = Pipeline([('rf', rf_classifier)])
        model = ModelPipeline(x_col=x_col, y_col=y_col_multi, model_dir=model_dir, pipeline=pipeline)
        # On simule un classifier multi label
        model.model_type = 'classifier'
        model.multi_label = True
        model.fit(x_train, y_train_multi)
        preds = model.predict_proba(x_train)
        self.assertEqual(preds.shape, (len(x_train), len(y_col_multi))) # 3 labels
        self.assertTrue(isinstance(preds[0][0], (np.floating, float)))
        remove_dir(model_dir)

        # Regressor
        rf_regressor = RandomForestRegressor(n_estimators=10)
        pipeline = Pipeline([('rf', rf_regressor)])
        model = ModelPipeline(x_col=x_col, y_col=y_col_mono, model_dir=model_dir, pipeline=pipeline)
        # On simule un regressor
        model.model_type = 'regressor'
        model.fit(x_train, y_train_regressor)
        with self.assertRaises(ValueError):
            proba = model.predict_proba(x_train)
        remove_dir(model_dir)

        # Model needs to be fitted
        with self.assertRaises(AttributeError):
            model = ModelPipeline(x_col=x_col, y_col=y_col_mono, model_dir=model_dir, pipeline=pipeline)
            model.predict_proba('test')
        remove_dir(model_dir)


    def test05_model_pipeline_save(self):
        '''Test de la fonction save de ynov.models_training.model_pipeline.ModelPipeline'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        x_col = ['col_1', 'col_2']
        y_col_mono = ['toto']

        # Avec Pipeline - Classifier
        rf_classifier = RandomForestClassifier(n_estimators=10)
        pipeline = Pipeline([('rf', rf_classifier)])
        model = ModelPipeline(x_col=x_col, y_col=y_col_mono, model_dir=model_dir, pipeline=pipeline)
        model.model_type = 'classifier'
        # Save
        model.save(json_data={'test': 8})
        # Assert everything been saved
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
        self.assertTrue('list_classes' not in configs.keys()) # not in car on n'utilise pas le mixin Classifier
        self.assertTrue('dict_classes' not in configs.keys()) # not in car on n'utilise pas le mixin Classifier
        self.assertTrue('multi_label' not in configs.keys()) # not in car on n'utilise pas le mixin Classifier
        # Spécifique model utilisé
        self.assertTrue('rf_confs' in configs.keys())
        remove_dir(model_dir)

        # Avec Pipeline - Regressor
        rf_regressor = RandomForestRegressor(n_estimators=10)
        pipeline = Pipeline([('rf', rf_regressor)])
        model = ModelPipeline(x_col=x_col, y_col=y_col_mono, model_dir=model_dir, pipeline=pipeline)
        model.model_type = 'regressor'
        # Save
        model.save(json_data={'test': 8})
        # Assert everything been saved
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
        self.assertTrue('list_classes' not in configs.keys()) # not in car Regressor
        self.assertTrue('dict_classes' not in configs.keys()) # not in car Regressor
        self.assertTrue('multi_label' not in configs.keys()) # not in car Regressor
        # Spécifique model utilisé
        self.assertTrue('rf_confs' in configs.keys())
        remove_dir(model_dir)

        # Sans Pipeline
        model = ModelPipeline(x_col=x_col, y_col=y_col_mono, model_dir=model_dir, pipeline=None)
        model.model_type = 'classifier' # On ne test pas regressor, pareil
        model.save(json_data={'test': 8})
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, f"{model.model_name}.pkl")))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, f"{model.model_name}_standalone.pkl")))
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
        self.assertTrue('list_classes' not in configs.keys()) # not in car on n'utilise pas le mixin Classifier
        self.assertTrue('dict_classes' not in configs.keys()) # not in car on n'utilise pas le mixin Classifier
        self.assertTrue('multi_label' not in configs.keys()) # not in car on n'utilise pas le mixin Classifier
        # Spécifique model utilisé
        self.assertTrue('rf_confs' not in configs.keys())
        remove_dir(model_dir)

        # AVEC level_save = 'LOW'
        rf_classifier = RandomForestClassifier(n_estimators=10)
        pipeline = Pipeline([('rf', rf_classifier)])
        model = ModelPipeline(x_col=x_col, y_col=y_col_mono, model_dir=model_dir, pipeline=pipeline, level_save='LOW')
        model.model_type = 'classifier'
        model.multi_label = False
        # Save
        model.save(json_data={'test': 8})
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, f"{model.model_name}.pkl")))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, f"{model.model_name}_standalone.pkl")))
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
        self.assertTrue('list_classes' not in configs.keys()) # not in car on n'utilise pas le mixin Classifier
        self.assertTrue('dict_classes' not in configs.keys()) # not in car on n'utilise pas le mixin Classifier
        self.assertTrue('multi_label' not in configs.keys()) # not in car on n'utilise pas le mixin Classifier
        # Spécifique model utilisé
        self.assertTrue('rf_confs' in configs.keys())
        remove_dir(model_dir)


# Execution des tests
if __name__ == '__main__':
    # Start tests
    unittest.main()