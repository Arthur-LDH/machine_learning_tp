#!/usr/bin/env python3

# Libs unittest
import unittest
from unittest.mock import patch
from unittest.mock import Mock

# Utils libs
import os
import json
import numpy as np
import pandas as pd
from ynov import utils
from ynov.preprocessing import outlier_detection

# Disable logging
import logging
logging.disable(logging.CRITICAL)


class OutlierDetectionTests(unittest.TestCase):
    '''Main class to test all functions in ynov.preprocessing.outlier_detection'''


    def setUp(self):
        '''SetUp fonction'''
        # On se place dans le bon répertoire
        # Change directory to script directory
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname)


    def test01_check_for_outliers(self):
        '''Test de la fonction outlier_detection.check_for_outliers'''

        # Vals à tester
        df = pd.DataFrame({
            'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'y': [1, 2, 3, 4, 5, 6, 7, 100000, 100000000, 100000000],
            'z': [-10000, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        })
        df_copy = df.copy(deep=True)
        arr = df.to_numpy()
        arr_copy = arr.copy()
        expected_outliers = np.array([-1,  1,  1,  1,  1,  1,  1, -1, -1, -1])

        # Fonctionnement nominal
        outliers = outlier_detection.check_for_outliers(df)
        outliers_bis = outlier_detection.check_for_outliers(arr)
        np.testing.assert_array_equal(outliers, expected_outliers)
        np.testing.assert_array_equal(outliers_bis, expected_outliers)
        pd.testing.assert_frame_equal(df, df_copy)
        np.testing.assert_array_equal(arr, arr_copy)

        # Gestion erreurs
        with self.assertRaises(ValueError):
            outlier_detection.check_for_outliers('toto')


# Execution des tests
if __name__ == '__main__':
    # Start tests
    unittest.main()