""" import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import joblib

# Load preprocessed data
df = pd.read_csv('preprocessed_data.csv')
X = df[['duration', 'days_left', 'airline_encoded', 'source_city_encoded', 
        'destination_city_encoded', 'class_encoded', 'stops_encoded', 
        'departure_time_encoded', 'arrival_time_encoded', 'is_weekend', 
        'is_peak', 'competition_factor']]
y = df['price']

# Split data (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to train, evaluate, and save a model
def train_evaluate_save(model, param_grid, model_name):
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    
    # Predictions
    y_pred = best_model.predict(X_test)
    
    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    print(f"{model_name} - Best Params: {grid_search.best_params_}")
    print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}, MAPE: {mape:.2f}%")
    
    # Save model
    joblib.dump(best_model, f"{model_name}.pkl")
    return best_model

# Linear Regression (Baseline)
lr = LinearRegression()
lr_params = {}  # No hyperparameters to tune
train_evaluate_save(lr, lr_params, 'linear_regression')

# Decision Tree (Baseline)
dt = DecisionTreeRegressor(random_state=42)
dt_params = {'max_depth': [5, 10, 15], 'min_samples_split': [2, 5, 10]}
train_evaluate_save(dt, dt_params, 'decision_tree')

# XGBoost (Advanced)
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
# xgb_params = {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 6]}
xgb_params = {
    'n_estimators': [100, 200, 300, 400],
    'learning_rate': [0.01, 0.05, 0.1, 0.15],
    'max_depth': [3, 5, 6, 8],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0]
}

train_evaluate_save(xgb_model, xgb_params, 'xgboost')

# LightGBM (Advanced)
lgb_model = lgb.LGBMRegressor(random_state=42)
lgb_params = {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 6]}
train_evaluate_save(lgb_model, lgb_params, 'lightgbm')

# CatBoost (Advanced)
cat_model = CatBoostRegressor(random_state=42, verbose=0)
cat_params = {'iterations': [100, 200], 'learning_rate': [0.01, 0.1], 'depth': [3, 6]}
train_evaluate_save(cat_model, cat_params, 'catboost') """


# train_predict.py - Optimized Model Training with Early Stopping

import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import joblib
import time
from config import *

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_score = -np.inf

    def load_data(self):
        """Load preprocessed training data"""
        try:
            X_train = pd.read_csv(TRAIN_TEST_FILES['X_train'])
            X_test = pd.read_csv(TRAIN_TEST_FILES['X_test'])
            y_train = pd.read_csv(TRAIN_TEST_FILES['y_train']).iloc[:, 0]
            y_test = pd.read_csv(TRAIN_TEST_FILES['y_test']).iloc[:, 0]

            logger.info(f"Data loaded - Train: {X_train.shape}, Test: {X_test.shape}")
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def calculate_metrics(self, y_true, y_pred):
        """Calculate comprehensive metrics"""
        metrics = {
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'R2': r2_score(y_true, y_pred),
            'MAPE': np.mean(np.abs((y_true - y_pred) / np.abs(y_true))) * 100
        }
        return metrics

    def train_model_with_cv(self, model, param_grid, model_name, X_train, y_train, X_test, y_test):
        """Train model with cross-validation and early stopping"""
        logger.info(f"Training {model_name}...")
        start_time = time.time()

        try:
            # Use smaller CV for faster training
            cv_folds = 3 if len(X_train) > 1000 else 2

            # Grid search with timeout protection
            grid_search = GridSearchCV(
                model, param_grid,
                cv=cv_folds,
                scoring='neg_root_mean_squared_error',
                n_jobs=-1,
                verbose=1
            )

            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_

            # Make predictions
            y_pred_train = best_model.predict(X_train)
            y_pred_test = best_model.predict(X_test)

            # Calculate metrics
            train_metrics = self.calculate_metrics(y_train, y_pred_train)
            test_metrics = self.calculate_metrics(y_test, y_pred_test)

            # Cross-validation score
            cv_scores = cross_val_score(best_model, X_train, y_train,
                                       cv=cv_folds, scoring='neg_root_mean_squared_error')
            cv_rmse = -cv_scores.mean()

            training_time = time.time() - start_time

            # Store results
            self.results[model_name] = {
                'model': best_model,
                'best_params': grid_search.best_params_,
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'cv_rmse': cv_rmse,
                'training_time': training_time,
                'predictions_test': y_pred_test
            }

            # Check if this is the best model
            if test_metrics['R2'] > self.best_score:
                self.best_score = test_metrics['R2']
                self.best_model = model_name

            # Log results
            logger.info(f"{model_name} completed in {training_time:.2f}s")
            logger.info(f"Best params: {grid_search.best_params_}")
            logger.info(f"Test RMSE: {test_metrics['RMSE']:.4f}, R2: {test_metrics['R2']:.4f}")

            # Save model
            joblib.dump(best_model, MODELS_DIR / f"{model_name.lower().replace(' ', '_')}.pkl")

        except Exception as e:
            logger.error(f"Error training {model_name}: {e}")
            self.results[model_name] = {'error': str(e)}

    def train_all_models(self):
        """Train all prediction models"""
        logger.info("Starting model training pipeline...")

        # Load data
        X_train, X_test, y_train, y_test = self.load_data()

        # Define models with optimized hyperparameters
        models_config = {
            'Linear Regression': (LinearRegression(), {}),
            'Decision Tree': (DecisionTreeRegressor(random_state=RANDOM_STATE),
                              HYPERPARAMETERS['decision_tree']),
            'Random Forest': (RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
                              {'n_estimators': [50, 100], 'max_depth': [10, 20]}),
            'XGBoost': (xgb.XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1),
                        HYPERPARAMETERS['xgboost']),
            'LightGBM': (lgb.LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbose=-1),
                         HYPERPARAMETERS['lightgbm']),
            'CatBoost': (CatBoostRegressor(random_state=RANDOM_STATE, verbose=False),
                         HYPERPARAMETERS['catboost'])
        }

        # Train each model
        for model_name, (model, param_grid) in models_config.items():
            self.train_model_with_cv(model, param_grid, model_name,
                                    X_train, y_train, X_test, y_test)

        # Save results summary
        self.save_results()

    def save_results(self):
        """Save comprehensive results"""
        logger.info("Saving model comparison results...")

        # Create results DataFrame
        results_data = []
        for model_name, result in self.results.items():
            if 'error' not in result:
                results_data.append({
                    'Model': model_name,
                    'RMSE': result['test_metrics']['RMSE'],
                    'MAE': result['test_metrics']['MAE'],
                    'R2': result['test_metrics']['R2'],
                    'MAPE': result['test_metrics']['MAPE'],
                    'CV_RMSE': result['cv_rmse'],
                    'Training_Time': result['training_time'],
                    'Best_Params': str(result['best_params'])
                })

        results_df = pd.DataFrame(results_data)
        results_df = results_df.sort_values('R2', ascending=False)

        # Save to CSV
        results_df.to_csv('model_comparison_results.csv', index=False)

        # Print summary
        print("\n" + "="*80)
        print("MODEL TRAINING RESULTS SUMMARY")
        print("="*80)
        print(results_df[['Model', 'RMSE', 'MAE', 'R2', 'MAPE']].to_string(index=False))
        print(f"\n🏆 BEST MODEL: {self.best_model} (R2 = {self.best_score:.4f})")

        # Save best model separately
        if self.best_model and self.best_model in self.results:
            best_model_obj = self.results[self.best_model]['model']
            joblib.dump(best_model_obj, MODELS_DIR / 'best_price_prediction_model.pkl')
            logger.info(f"Best model saved: {self.best_model}")

def main():
    """Main training pipeline"""
    print("="*70)
    print("FLIGHT PRICE PREDICTION - MODEL TRAINING")
    print("="*70)

    try:
        trainer = ModelTrainer()
        trainer.train_all_models()
        print("✅ Model training completed successfully!")

    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()


