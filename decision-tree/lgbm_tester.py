import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import lightgbm as lgb


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

        # Création du pipeline avec paramètres par défaut de LightGBM
        self.model = Pipeline([
            ('preprocessor', self.preprocessor),
            ('regressor', lgb.LGBMRegressor(
                n_estimators=1000,  # Augmenté car early stopping
                learning_rate=0.05,
                num_leaves=31,
                max_depth=8,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            ))
        ])

    def load_and_prepare_data(self, train_path, valid_path, test_path=None):
        # Chargement des données
        self.train = pd.read_csv(train_path)
        self.valid = pd.read_csv(valid_path)
        if test_path:
            self.test = pd.read_csv(test_path)

        # Préparation des données train et valid séparément pour l'early stopping
        self.X_train = self.train[self.numeric_features + self.categorical_features]
        self.y_train = self.train[self.target_column]

        self.X_valid = self.valid[self.numeric_features + self.categorical_features]
        self.y_valid = self.valid[self.target_column]

        # Combinaison pour l'entraînement final
        self.X_full = pd.concat([self.X_train, self.X_valid])
        self.y_full = pd.concat([self.y_train, self.y_valid])

        if test_path:
            self.X_test = self.test[self.numeric_features + self.categorical_features]

        print(f"Nombre d'échantillons d'entraînement: {len(self.X_train)}")
        print(f"Nombre d'échantillons de validation: {len(self.X_valid)}")
        print(f"Nombre total d'échantillons: {len(self.X_full)}")
        return self

    def train_evaluate(self, early_stopping_rounds=50):
        print("\nPremière phase: Entraînement avec early stopping...")
        # Prétraitement des données
        X_train_processed = self.preprocessor.fit_transform(self.X_train)
        X_valid_processed = self.preprocessor.transform(self.X_valid)

        # Récupération du modèle LightGBM
        lgb_model = self.model.named_steps['regressor']

        # Premier entraînement avec early stopping
        lgb_model.fit(
            X_train_processed, self.y_train,
            eval_set=[(X_train_processed, self.y_train), (X_valid_processed, self.y_valid)],
            eval_metric='rmse',
            callbacks=[lgb.early_stopping(early_stopping_rounds), lgb.log_evaluation(10)]
        )

        best_iteration = lgb_model.booster_.best_iteration
        print(f"\nMeilleure itération trouvée: {best_iteration}")

        print("\nPhase finale: Entraînement sur toutes les données...")
        # Mise à jour du nombre d'itérations pour l'entraînement final
        self.model.named_steps['regressor'].set_params(n_estimators=best_iteration)

        # Entraînement final sur toutes les données
        self.model.fit(self.X_full, self.y_full)

        # Évaluation finale
        y_pred = self.model.predict(self.X_full)
        rmse = np.sqrt(mean_squared_error(self.y_full, y_pred))
        r2 = r2_score(self.y_full, y_pred)

        print(f"\nMétriques de performance finales:")
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
    tester = LightGBMTester()

    try:
        # Chargement et préparation des données
        tester.load_and_prepare_data(
            train_path="../ynov-data/train_housing_train.csv",
            valid_path="../ynov-data/train_housing_valid.csv",
            test_path="../ynov-data/test_housing.csv"
        )

        # Entraînement avec early stopping et évaluation
        tester.train_evaluate(early_stopping_rounds=50)

        # Création de la soumission
        tester.create_submission('../ynov-data/submission_lgbm.csv')

    except Exception as e:
        print(f"\nErreur: {str(e)}")
        raise


if __name__ == "__main__":
    main()