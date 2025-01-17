import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
import lightgbm as lgb
import optuna
from tqdm import tqdm


class LightGBMTester:
    def __init__(self):
        # Features de base
        self.numeric_features = [
            'longitude', 'latitude', 'housing_median_age', 'total_rooms',
            'total_bedrooms', 'population', 'households', 'median_income'
        ]
        self.categorical_features = ['ocean_proximity']
        self.target_column = 'median_house_value'

        # Pipeline de prétraitement
        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(drop='first', sparse_output=False))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ])

        # Création du pipeline de base
        self.model = Pipeline([
            ('preprocessor', self.preprocessor),
            ('regressor', lgb.LGBMRegressor(random_state=42))
        ])

    def load_and_prepare_data(self, train_path, valid_path, test_path=None):
        # Chargement des données
        self.train = pd.read_csv(train_path)
        self.valid = pd.read_csv(valid_path)
        if test_path:
            self.test = pd.read_csv(test_path)

        # Préparation des données d'entraînement
        self.X_train = self.train[self.numeric_features + self.categorical_features]
        self.y_train = self.train[self.target_column]

        # Préparation des données de validation
        self.X_valid = self.valid[self.numeric_features + self.categorical_features]
        self.y_valid = self.valid[self.target_column]

        # Combinaison des données train et valid pour l'entraînement final
        self.X_full = pd.concat([self.X_train, self.X_valid])
        self.y_full = pd.concat([self.y_train, self.y_valid])

        if test_path:
            self.X_test = self.test[self.numeric_features + self.categorical_features]

        print(f"Nombre d'échantillons d'entraînement: {len(self.X_train)}")
        print(f"Nombre d'échantillons de validation: {len(self.X_valid)}")
        print(f"Nombre total d'échantillons pour l'entraînement final: {len(self.X_full)}")

        return self

    def optimize_params(self, n_trials=100):
        def objective(trial):
            params = {
                'regressor__n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'regressor__learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'regressor__num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'regressor__max_depth': trial.suggest_int('max_depth', 3, 12),
                'regressor__min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'regressor__subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'regressor__colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0)
            }

            self.model.set_params(**params)

            # Validation croisée sur les données d'entraînement uniquement
            scores = cross_val_score(
                self.model,
                self.X_train,  # Utilisation des données d'entraînement uniquement
                self.y_train,
                scoring='neg_root_mean_squared_error',
                cv=5,
                n_jobs=-1
            )

            return scores.mean()

        study = optuna.create_study(direction='maximize')

        print("\nOptimisation des hyperparamètres...")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best_params = study.best_params
        print("\nMeilleurs paramètres trouvés:")
        for param, value in best_params.items():
            print(f"{param}: {value}")
        print(f"Meilleur RMSE: {-study.best_value:.2f}")

        best_params = {f'regressor__{k}': v for k, v in best_params.items()}
        self.model.set_params(**best_params)

        return self

    def train_evaluate(self):
        # Entraînement sur l'ensemble des données (train + valid)
        print("\nEntraînement sur l'ensemble complet des données (train + valid)...")
        self.model.fit(self.X_full, self.y_full)

        # Évaluation sur train+valid
        y_pred = self.model.predict(self.X_full)
        rmse = np.sqrt(mean_squared_error(self.y_full, y_pred))
        r2 = r2_score(self.y_full, y_pred)

        print(f"\nMétriques de performance (train + valid):")
        print(f"RMSE: {rmse:.2f}")
        print(f"R²: {r2:.4f}")

        return self

    def create_submission(self, output_path):
        if not hasattr(self, 'test'):
            raise ValueError("Pas de données de test chargées")

        predictions = self.model.predict(self.X_test)

        submission = pd.DataFrame({
            'id': self.test['id'],
            'median_house_value': predictions
        })

        submission.to_csv(output_path, index=False)
        print(f"\nFichier de soumission créé: {output_path}")
        return self


def main():
    # Initialisation
    tester = LightGBMTester()

    try:
        # Chargement et préparation des données
        tester.load_and_prepare_data(
            train_path="../ynov-data/train_housing_train.csv",
            valid_path="../ynov-data/train_housing_valid.csv",
            test_path="../ynov-data/test_housing.csv"
        )

        # Optimisation des hyperparamètres
        tester.optimize_params(n_trials=100)

        # Entraînement final et évaluation
        tester.train_evaluate()

        # Création de la soumission
        tester.create_submission('../ynov-data/submission_lgbm.csv')

    except Exception as e:
        print(f"\nErreur: {str(e)}")
        raise


if __name__ == "__main__":
    main()