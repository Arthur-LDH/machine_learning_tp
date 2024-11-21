#!/usr/bin/env python3

# Libs unittest
import unittest
from unittest.mock import patch
from unittest.mock import Mock

# Utils libs
import os
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer, make_column_selector
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_extraction.text import CountVectorizer
from ynov import utils
from ynov.preprocessing import preprocess

# Disable logging
import logging
logging.disable(logging.CRITICAL)


class PreprocessTests(unittest.TestCase):
    '''Main class to test all functions in ynov.preprocessing.preprocess'''


    def setUp(self):
        '''SetUp fonction'''
        # On se place dans le bon répertoire
        # Change directory to script directory
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname)


    def test01_get_pipelines_dict(self):
        '''Test de la fonction preprocess.get_pipelines_dict'''
        # Vals à tester
        # TODO: à modifier en fonction de votre projet !
        content = pd.DataFrame({'col_1': [-5, -1, 0, 2, -6, 3], 'col_2': [2, -1, -8, 3, 12, 2],
                                'text': ['toto', 'titi', 'tata', 'tutu', 'tyty', 'tete']})
        y = pd.Series([0, 1, 1, 1, 0, 0])

        # Fonctionnement nominal
        pipelines_dict = preprocess.get_pipelines_dict()
        self.assertEqual(type(pipelines_dict), dict)
        self.assertTrue('no_preprocess' in pipelines_dict.keys())

        # On test chaque fonctions retournées
        for p in pipelines_dict.values():
            p.fit(content, y)
            self.assertEqual(type(p.transform(content)), np.ndarray)
            self.assertEqual(type(p.transform(content)), np.ndarray)


    def test02_get_pipeline(self):
        '''Test de la fonction preprocess.get_pipeline'''
        # Vals à tester
        # On prend un preprocessing "au hasard"
        pipeline_str = list(preprocess.get_pipelines_dict().keys())[0]

        # Fonctionnement nominal
        pipeline = preprocess.get_pipeline(pipeline_str)
        # On test juste qu'on a bien une pipeline ...
        self.assertEqual(type(pipeline), ColumnTransformer)

        # Vérification du type du/des input(s)
        with self.assertRaises(ValueError):
            preprocess.get_pipeline('NOT A VALID PREPROCESS')


    def test03_retrieve_columns_from_pipeline(self):
        '''Test de la fonction preprocess.retrieve_columns_from_pipeline'''
        # Pipeline
        col_1_3_pipeline = make_pipeline(SimpleImputer(strategy='median'), StandardScaler())
        col_2_pipeline = make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder(handle_unknown='ignore'))
        text_pipeline = make_pipeline(CountVectorizer(), SelectKBest(k=2))
        transformers = [
            ('col_1_3', col_1_3_pipeline, ['col_1', 'col_3']),
            ('col_2', col_2_pipeline, ['col_2']),
            ('text', text_pipeline, 'text'),
        ]
        pipeline = ColumnTransformer(transformers, remainder='drop')
        # DataFrame
        df = pd.DataFrame({'col_1': [1, 5, 8, 4], 'col_2': [0.0, None, 1.0, 1.0], 'col_3': [-5, 6, 8, 6],
                           'toto': [4, 8, 9, 4],
                           'text': ['ceci est un test', 'un autre test', 'et un troisième test', 'et un dernier']})
        # Target
        y = pd.Series([1, 1, 1, 0])
        # Fit
        pipeline.fit(df, y)
        # transform
        transformed_df = pd.DataFrame(pipeline.transform(df))

        # Fonctionnement nominal
        new_transformed_df = preprocess.retrieve_columns_from_pipeline(transformed_df, pipeline)
        self.assertEqual(list(new_transformed_df.columns), ['col_1', 'col_3', 'col_2_0.0', 'col_2_1.0', 'vec_dernier', 'vec_test'])

        # Si pipeline pas fit, pas de moficiation
        tmp_pipeline = ColumnTransformer(transformers, remainder='drop')
        new_transformed_df = preprocess.retrieve_columns_from_pipeline(transformed_df, tmp_pipeline)
        pd.testing.assert_frame_equal(new_transformed_df, transformed_df)

        # Si pas le bon nombre de colonnes, pas de moficiation
        new_transformed_df = preprocess.retrieve_columns_from_pipeline(df, pipeline)
        pd.testing.assert_frame_equal(new_transformed_df, df)


    def test04_get_feature_out(self):
        '''Test de la fonction preprocess.get_feature_out'''

        # Fonctionnement nominal - non _VectorizerMixin - non SelectorMixin
        estimator = SimpleImputer(strategy='median')
        estimator.fit(pd.DataFrame({'col': [1, 0, 1, 1, None]}))
        feature_out = preprocess.get_feature_out(estimator, 'toto')
        self.assertEqual(feature_out, 'toto')
        feature_out = preprocess.get_feature_out(estimator, ['toto', 'tata'])
        self.assertEqual(feature_out, ['toto', 'tata'])

        # Fonctionnement nominal - _VectorizerMixin
        estimator = OneHotEncoder(handle_unknown='ignore')
        estimator.fit(pd.DataFrame({'col': [0, 0, 1, 1, 0]}))
        feature_out = preprocess.get_feature_out(estimator, ['toto'])
        self.assertEqual(list(feature_out), ['toto_0', 'toto_1'])

        # Fonctionnement nominal - SelectorMixin
        estimator = SelectKBest(k=2)
        estimator.fit(pd.DataFrame({'col_1': [0, 0, 1, 1, 1], 'col_2': [1, 1, 0, 0, 0], 'col_3': [0, 0, 0, 0, 0]}), pd.Series([-1, -1, 1, 1, 1]))
        feature_out = preprocess.get_feature_out(estimator, ['col_1', 'col_2', 'col_3'])
        self.assertEqual(list(feature_out), ['col_1', 'col_2'])


    def test05_get_ct_feature_names(self):
        '''Test de la fonction preprocess.get_ct_feature_names'''
        # Pipeline
        col_1_3_pipeline = make_pipeline(SimpleImputer(strategy='median'), StandardScaler())
        col_2_pipeline = make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder(handle_unknown='ignore'))
        text_pipeline = make_pipeline(CountVectorizer(), SelectKBest(k=2))
        transformers = [
            ('col_1_3', col_1_3_pipeline, ['col_1', 'col_3']),
            ('col_2', col_2_pipeline, ['col_2']),
            ('text', text_pipeline, 'text'),
        ]
        pipeline = ColumnTransformer(transformers, remainder='drop')
        # DataFrame
        df = pd.DataFrame({'col_1': [1, 5, 8, 4], 'col_2': [0.0, None, 1.0, 1.0], 'col_3': [-5, 6, 8, 6],
                           'toto': [4, 8, 9, 4],
                           'text': ['ceci est un test', 'un autre test', 'et un troisième test', 'et un dernier']})
        # Target
        y = pd.Series([1, 1, 1, 0])
        # Fit
        pipeline.fit(df, y)

        # Fonctionnement nominal
        output_features = preprocess.get_ct_feature_names(pipeline)
        self.assertEqual(output_features, ['col_1', 'col_3', 'col_2_0.0', 'col_2_1.0', 'vec_dernier', 'vec_test'])

        # remainder == 'passthrough'
        pipeline = ColumnTransformer(transformers, remainder='passthrough')
        pipeline.fit(df, y)
        output_features = preprocess.get_ct_feature_names(pipeline)
        self.assertEqual(output_features, ['col_1', 'col_3', 'col_2_0.0', 'col_2_1.0', 'vec_dernier', 'vec_test', 'toto'])


# Execution des tests
if __name__ == '__main__':
    # Start tests
    unittest.main()