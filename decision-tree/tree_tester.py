import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import optuna
from tqdm import tqdm
from sklearn.model_selection import cross_val_score
import lightgbm as lgb


class LGBMWrapper(lgb.LGBMRegressor):
    def __init__(self, num_leaves=31, max_depth=-1, learning_rate=0.1,
                 n_estimators=100, subsample=1.0, colsample_bytree=1.0,
                 min_child_samples=20, random_state=None, n_jobs=-1):
        super().__init__(
            num_leaves=num_leaves,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_samples=min_child_samples,
            random_state=random_state,
            n_jobs=n_jobs
        )

    def __sklearn_tags__(self):
        return {
            'allow_nan': True,
            'requires_fit': True,
            'requires_y': True,
            'enable_metadata_routing': False,
            '_skip_test': True
        }


class LightGBMTester:
    def __init__(self):
        # Configuration des features
        self.numeric_features = [
            'longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population',
            'households', 'median_income'
        ]
        self.categorical_features = ['ocean_proximity']
        self.target_column = 'median_house_value'
        self.column_to_drop = ['id']
        self.custom_features = []

        # Données
        self.train = None
        self.valid = None
        self.test = None
        self.X_train = None
        self.y_train = None
        self.X_valid = None
        self.y_valid = None
        self.X_test = None

        # Pipeline et modèle
        self.pipeline = None
        self.preprocessor = None
        self.metrics = {}
        self.best_params = None

        # Paramètres de recherche pour LightGBM
        self.param_grid = {
            'num_leaves': (20, 100),
            'max_depth': (5, 50),
            'learning_rate': (0.01, 0.1),
            'feature_fraction': (0.5, 1.0),
            'bagging_fraction': (0.5, 1.0),
            'min_data_in_leaf': (10, 100),
            'n_estimators': (100, 500)
        }

        self._update_preprocessor()

    def _update_preprocessor(self):
        ocean_categories = [['<1H OCEAN', 'INLAND', 'NEAR BAY', 'NEAR OCEAN', 'ISLAND']]

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]), self.numeric_features),
                ('cat', Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('encoder', OneHotEncoder(
                        drop='first',
                        handle_unknown='ignore',
                        sparse_output=False,
                        categories=ocean_categories
                    ))
                ]), self.categorical_features)
            ])

    def add_feature(self, feature_name, calculation_func):
        self.custom_features.append((feature_name, calculation_func))
        if feature_name not in self.numeric_features:
            self.numeric_features.append(feature_name)
            self._update_preprocessor()
        return self

    def _apply_custom_features(self, df):
        df_copy = df.copy()
        for feature_name, calculation_func in self.custom_features:
            df_copy[feature_name] = calculation_func(df_copy)
        return df_copy

    def load_data(self, train_path, test_path=None, valid_path=None):
        self.train = pd.read_csv(train_path)
        if test_path:
            self.test = pd.read_csv(test_path)
        if valid_path:
            self.valid = pd.read_csv(valid_path)

        if hasattr(self, 'train'):
            self.train = self.train.drop(columns=self.column_to_drop, errors='ignore')
        if hasattr(self, 'valid'):
            self.valid = self.valid.drop(columns=self.column_to_drop, errors='ignore')

        print("Colonnes disponibles:", self.train.columns.tolist())
        return self

    def prepare_data(self):
        missing_cols = [col for col in self.numeric_features + self.categorical_features
                        if col not in self.train.columns and col not in [f[0] for f in self.custom_features]]
        if missing_cols:
            raise ValueError(f"Colonnes manquantes dans le dataset: {missing_cols}")

        train_processed = self._apply_custom_features(self.train)
        self.X_train = train_processed[self.numeric_features + self.categorical_features]
        self.y_train = train_processed[self.target_column]

        if self.valid is not None:
            valid_processed = self._apply_custom_features(self.valid)
            self.X_valid = valid_processed[self.numeric_features + self.categorical_features]
            self.y_valid = valid_processed[self.target_column]

        if self.test is not None:
            test_processed = self._apply_custom_features(self.test)
            self.X_test = test_processed[self.numeric_features + self.categorical_features]

        return self

    def create_pipeline(self):
        model = LGBMWrapper(
            random_state=42,
            n_jobs=-1
        )

        if self.pipeline is None:
            self.pipeline = Pipeline([
                ('preprocessor', self.preprocessor),
                ('regressor', model)
            ])
        else:
            self.pipeline.steps[-1] = ('regressor', model)

        return self.pipeline

    def grid_search(self, n_iter=100):
        def objective(trial):
            params = {
                'regressor__num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'regressor__max_depth': trial.suggest_int('max_depth', 5, 50),
                'regressor__learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'regressor__colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'regressor__subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'regressor__min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
                'regressor__n_estimators': trial.suggest_int('n_estimators', 100, 500)
            }

            pipeline = self.create_pipeline()
            pipeline.set_params(**params)

            # Validation croisée avec early stopping
            scores = cross_val_score(
                pipeline,
                self.X_train,
                self.y_train,
                cv=5,
                scoring='neg_root_mean_squared_error',
                n_jobs=4
            )

            return scores.mean()

        study = optuna.create_study(direction='maximize')

        with tqdm(total=n_iter, desc="Optimisation") as pbar:
            def callback(study, trial):
                pbar.update(1)

            study.optimize(
                objective,
                n_trials=n_iter,
                callbacks=[callback],
                n_jobs=1
            )

        self.best_params = {
            key.replace('regressor__', ''): value
            for key, value in study.best_params.items()
        }

        self.pipeline = self.create_pipeline()
        final_params = {f'regressor__{k}': v for k, v in self.best_params.items()}
        self.pipeline.set_params(**final_params)

        # Ajout d'early stopping sur l'ensemble de validation
        if self.X_valid is not None:
            self.pipeline.fit(
                self.X_train,
                self.y_train,
                regressor__eval_set=[(self.X_valid, self.y_valid)],
                regressor__early_stopping_rounds=50,
                regressor__eval_metric='rmse'
            )
        else:
            self.pipeline.fit(self.X_train, self.y_train)

        self.evaluate()

        print("\nMeilleurs paramètres trouvés:")
        for param, value in self.best_params.items():
            print(f"{param}: {value}")
        print(f"Meilleur RMSE: {-study.best_value:.4f}")

        try:
            import plotly
            print("\nCréation des visualisations Optuna...")
            fig1 = optuna.visualization.plot_optimization_history(study)
            fig1.show()
            fig2 = optuna.visualization.plot_param_importances(study)
            fig2.show()
        except:
            print("Impossible de créer les visualisations. Vérifiez que plotly est installé.")

        return self.best_params

    def train_model(self, **params):
        self.pipeline = self.create_pipeline()
        params = {f'regressor__{k}': v for k, v in params.items()}
        self.pipeline.set_params(**params)

        # Utilisation de early stopping si un ensemble de validation est disponible
        if self.X_valid is not None:
            self.pipeline.fit(
                self.X_train,
                self.y_train,
                regressor__eval_set=[(self.X_valid, self.y_valid)],
                regressor__early_stopping_rounds=50,
                regressor__eval_metric='rmse'
            )
        else:
            self.pipeline.fit(self.X_train, self.y_train)

        return self

    def evaluate(self):
        y_pred_train = self.pipeline.predict(self.X_train)
        self.metrics['train_mse'] = mean_squared_error(self.y_train, y_pred_train)
        self.metrics['train_rmse'] = np.sqrt(self.metrics['train_mse'])
        self.metrics['train_r2'] = r2_score(self.y_train, y_pred_train)

        if self.X_valid is not None:
            y_pred_valid = self.pipeline.predict(self.X_valid)
            self.metrics['valid_mse'] = mean_squared_error(self.y_valid, y_pred_valid)
            self.metrics['valid_rmse'] = np.sqrt(self.metrics['valid_mse'])
            self.metrics['valid_r2'] = r2_score(self.y_valid, y_pred_valid)

        return self

    def print_metrics(self):
        print("\nMétriques de performance:")
        print("-" * 40)
        print(f"RMSE (train): {self.metrics['train_rmse']:.4f}")
        print(f"R²   (train): {self.metrics['train_r2']:.4f}")

        if 'valid_rmse' in self.metrics:
            print(f"RMSE (valid): {self.metrics['valid_rmse']:.4f}")
            print(f"R²   (valid): {self.metrics['valid_r2']:.4f}")

    def create_submission(self, path):
        if self.X_test is None:
            raise ValueError("Aucune donnée de test disponible")

        y_pred_test = self.pipeline.predict(self.X_test)
        submission = pd.DataFrame({
            'id': self.test['id'],
            'median_house_value': y_pred_test
        })
        submission.to_csv(path, index=False)
        print(f"Fichier de soumission enregistré: {path}")
        return self


def main():
    print("Initialisation du LightGBMTester...")
    tester = LightGBMTester()

    try:
        # Ajout des features
        tester.add_feature(
            'rooms_per_household',
            lambda df: df['total_rooms'] / df['households']
        )

        tester.add_feature(
            'income_per_household',
            lambda df: df['median_income'] / df['households']
        )

        tester.add_feature(
            'location_interaction',
            lambda df: df['longitude'] * df['latitude']
        )

        tester.add_feature(
            'age_income_interaction',
            lambda df: df['housing_median_age'] * df['median_income']
        )

        print("\nChargement des données...")
        tester.load_data(
            train_path="../ynov-data/train_housing_train.csv",
            test_path="../ynov-data/test_housing.csv",
            valid_path="../ynov-data/train_housing_valid.csv"
        )
        tester.prepare_data()
        best_params = tester.grid_search()

        print("\nEntraînement du LightGBM final...")
        tester.train_model(**best_params)
        tester.evaluate()
        tester.print_metrics()

        print("\nCréation du fichier de soumission...")
        tester.create_submission('../ynov-data/submission_lgbm.csv')

    except Exception as e:
        print(f"\nErreur: {str(e)}")
        print("\nColonnes requises:", tester.numeric_features + tester.categorical_features)
        raise


if __name__ == "__main__":
    main()