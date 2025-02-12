import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import optuna
from tqdm import tqdm
from sklearn.model_selection import cross_val_score


class TreeTester:
    # ===== Initialisation =====
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

        # Paramètres de recherche
        # Modifiez le param_grid dans __init__
        self.param_grid = {
            'max_depth': list(range(5, 50, 2)),
            'min_samples_split': list(range(2, 20, 2)),
            'min_samples_leaf': list(range(1, 20, 2)),
            'max_features': list(range(3, 8)),
            'n_estimators': [100, 200, 300, 500]
        }

        self._update_preprocessor()

    def _update_preprocessor(self):
        # Définir explicitement toutes les catégories possibles pour ocean_proximity
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

    # ===== Chargement et préparation des données =====
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
        """Prépare les données pour l'entraînement"""
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

    # ===== Pipeline et modèle =====
    def create_pipeline(self):
        model = RandomForestRegressor(
            random_state=42,
            bootstrap=True,
            oob_score=True,
            max_samples=0.7
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
        """Recherche les meilleurs paramètres avec Optuna"""

        def objective(trial):
            # Définir les paramètres à optimiser
            params = {
                'regressor__max_depth': trial.suggest_int('max_depth', 20, 40),
                'regressor__min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'regressor__min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'regressor__max_features': trial.suggest_int('max_features', 3, 8),
                'regressor__n_estimators': trial.suggest_int('n_estimators', 100, 500)
            }

            # Créer et configurer le pipeline
            pipeline = self.create_pipeline()
            pipeline.set_params(**params)

            # Réduire le nombre de jobs pour la validation croisée
            scores = cross_val_score(
                pipeline,
                self.X_train,
                self.y_train,
                cv=5,  # Réduit de 10 à 5
                scoring='neg_root_mean_squared_error',
                n_jobs=4  # Limite fixe au lieu de -1
            )

            return scores.mean()

        # Créer l'étude Optuna
        study = optuna.create_study(direction='maximize')

        # Configurer la barre de progression
        with tqdm(total=n_iter, desc="Optimisation") as pbar:
            def callback(study, trial):
                pbar.update(1)

            # Lancer l'optimisation
            study.optimize(
                objective,
                n_trials=n_iter,
                callbacks=[callback],
                n_jobs=1  # Force single-threaded optimization
            )

        # Récupérer les meilleurs paramètres
        self.best_params = {
            key.replace('regressor__', ''): value
            for key, value in study.best_params.items()
        }

        # Entraîner le modèle final avec les meilleurs paramètres
        self.pipeline = self.create_pipeline()
        final_params = {f'regressor__{k}': v for k, v in self.best_params.items()}
        self.pipeline.set_params(**final_params)
        self.pipeline.fit(self.X_train, self.y_train)
        self.evaluate()

        print("\nMeilleurs paramètres trouvés:")
        for param, value in self.best_params.items():
            print(f"{param}: {value}")
        print(f"Meilleur RMSE: {-study.best_value:.4f}")

        # Optionally create visualizations if plotly is available
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

    def train_model(self, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                    max_features=None, n_estimators=100):
        """Entraîne le modèle avec les paramètres spécifiés"""
        self.pipeline = self.create_pipeline()

        params = {
            'regressor__max_depth': max_depth,
            'regressor__min_samples_split': min_samples_split,
            'regressor__min_samples_leaf': min_samples_leaf,
            'regressor__max_features': max_features,
            'regressor__n_estimators': n_estimators,
            'regressor__n_jobs': -1,
            'regressor__random_state': 42
        }

        self.pipeline.set_params(**params)
        self.pipeline.fit(self.X_train, self.y_train)
        return self

    def evaluate(self):
        """Évalue le modèle"""
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

    # ===== Utilitaires =====
    def print_metrics(self):
        """Affiche les métriques de performance"""
        print("\nMétriques de performance:")
        print("-" * 40)
        print(f"RMSE (train): {self.metrics['train_rmse']:.4f}")
        print(f"R²   (train): {self.metrics['train_r2']:.4f}")

        if 'valid_rmse' in self.metrics:
            print(f"RMSE (valid): {self.metrics['valid_rmse']:.4f}")
            print(f"R²   (valid): {self.metrics['valid_r2']:.4f}")

    def create_submission(self, path):
        """Crée un fichier de soumission avec les prédictions"""
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
    print("Initialisation du TreeTester...")
    tester = TreeTester()

    try:
        # # Ajoutez ces nouvelles features
        tester.add_feature(
            'rooms_per_household',
            lambda df: df['total_rooms'] / df['households']
        )

        # tester.add_feature(
        #     'population_density',
        #     lambda df: df['population'] / df['households']
        # )
        #
        tester.add_feature(
            'income_per_household',
            lambda df: df['median_income'] / df['households']
        )

        # # Interactions géographiques
        tester.add_feature(
            'location_interaction',
            lambda df: df['longitude'] * df['latitude']
        )

        # tester.add_feature(
        #     'bedrooms_ratio',
        #     lambda df: df['total_bedrooms'] / df['total_rooms']
        # )
        #
        # # Features non-linéaires
        # tester.add_feature(
        #     'income_squared',
        #     lambda df: df['median_income'] ** 2
        # )
        #
        # # Features d'interaction avec l'âge
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

        print("\nEntraînement du RandomForestRegressor final...")
        tester.train_model(
            max_depth=best_params['max_depth'],
            min_samples_split=best_params['min_samples_split'],
            min_samples_leaf=best_params['min_samples_leaf'],
            max_features=best_params['max_features'],
            n_estimators=100
        )
        tester.evaluate()
        tester.print_metrics()

        print("\nCréation du fichier de soumission...")
        tester.create_submission('../ynov-data/submission.csv')

    except Exception as e:
        print(f"\nErreur: {str(e)}")
        print("\nColonnes requises:", tester.numeric_features + tester.categorical_features)
        raise


if __name__ == "__main__":
    main()