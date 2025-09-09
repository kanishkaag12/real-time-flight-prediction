""" import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('Clean_Dataset.csv')

def complete_preprocessing_pipeline(df, target_rows=10000):
   
    print("Starting preprocessing pipeline...")
    print(f"Original dataset shape: {df.shape}")
    
    # Step 1: Preprocessing
    def preprocess_data(df):
        df_clean = df.drop(['flight'], axis=1, errors='ignore')
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        
        for col in numeric_cols:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
        
        for col in categorical_cols:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
        
        return df_clean
    
    df_clean = preprocess_data(df)
    print(f"After cleaning: {df_clean.shape}")
    
    # Step 2: Encoding
    def encode_categorical(df):
        le = LabelEncoder()
        categorical_cols = ['airline', 'source_city', 'departure_time', 
                           'arrival_time', 'destination_city', 'class']
        
        for col in categorical_cols:
            if col in df.columns:
                df[col + '_encoded'] = le.fit_transform(df[col])
        
        stops_mapping = {'zero': 0, 'one': 1, 'two_or_more': 2}
        df['stops_encoded'] = df['stops'].map(stops_mapping)
        
        return df
    
    df_encoded = encode_categorical(df_clean)
    print(f"After encoding: {df_encoded.shape}")
    
    # Step 3: Reduce size (optional)
    def reduce_dataset(df, target_rows):
        if len(df) > target_rows:
            sample_df = df.groupby('airline', group_keys=False).apply(
                lambda x: x.sample(min(len(x), int(target_rows * len(x) / len(df))), 
                                  random_state=42)
            )
            if len(sample_df) > target_rows:
                sample_df = sample_df.sample(target_rows, random_state=42)
            return sample_df
        return df
    
    df_reduced = reduce_dataset(df_encoded, target_rows)
    print(f"After reduction: {df_reduced.shape}")
    
    # Step 4: Standardization
    numeric_features = ['duration', 'days_left', 'price']
    scaler = StandardScaler()
    df_standardized = df_reduced.copy()
    df_standardized[numeric_features] = scaler.fit_transform(df_reduced[numeric_features])
    
    # Step 5: PCA
    # Get all encoded features and numeric features
    encoded_features = [col for col in df_standardized.columns if 'encoded' in col]
    numeric_features_pca = ['duration', 'days_left']
    features_for_pca = encoded_features + numeric_features_pca
    
    X = df_standardized[features_for_pca]
    pca = PCA(n_components=0.95)  # Keep 95% variance
    X_pca = pca.fit_transform(X)
    
    # Create final dataset
    final_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])
    final_df['price'] = df_standardized['price'].values
    final_df['airline'] = df_standardized['airline'].values
    
    print(f"Final processed dataset: {final_df.shape}")
    print(f"Variance explained: {sum(pca.explained_variance_ratio_):.4f}")
    
    # Save results
    final_df.to_csv('flight_data_final_processed.csv', index=False)
    df_standardized.to_csv('flight_data_standardized.csv', index=False)
    
    return final_df, df_standardized

# Run the complete pipeline
final_dataset, standardized_data = complete_preprocessing_pipeline(df, target_rows=10000)

print("Pipeline completed successfully!")
print(f"Final dataset shape: {final_dataset.shape}")

# ... [all your previous preprocessing code] ...

# ==================== STEP 1: VERIFY PROCESSED DATA ====================
print("\n" + "="*50)
print("STEP 1: VERIFYING PROCESSED DATA")
print("="*50)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your processed data
try:
    final_data = pd.read_csv('flight_data_final_processed.csv')
    print("✅ Successfully loaded final processed data!")
    print("Final dataset info:")
    print(final_data.info())
    print("\nFirst 5 rows:")
    print(final_data.head())
    
    # Basic statistics
    print("\n📊 Basic statistics:")
    print(f"Dataset shape: {final_data.shape}")
    print(f"Number of PCA components: {len([col for col in final_data.columns if col.startswith('PC')])}")
    print(f"Price range: ₹{final_data['price'].min():.2f} to ₹{final_data['price'].max():.2f}")
    
except FileNotFoundError:
    print("❌ Error: Could not find flight_data_final_processed.csv")
    print("Make sure the file was created successfully in the previous steps.")
except Exception as e:
    print(f"❌ Error loading file: {e}")

print("="*50)

# ==================== STEP 2: SPLIT DATA FOR MACHINE LEARNING ====================
print("\n" + "="*60)
print("STEP 2: SPLITTING DATA FOR MACHINE LEARNING")
print("="*60)

from sklearn.model_selection import train_test_split

# Separate features and target
X = final_data.drop(['price', 'airline'], axis=1)  # All PCA components (features)
y = final_data['price']  # Target variable (price)

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

print(f"✅ Data split successfully!")
print(f"Training set: {X_train.shape} (80% of data)")
print(f"Testing set: {X_test.shape} (20% of data)")
print(f"Total samples: {len(X_train) + len(X_test)}")

# Save the split datasets for later use
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

print("✅ Split datasets saved as:")
print("   - X_train.csv (training features)")
print("   - X_test.csv (testing features)") 
print("   - y_train.csv (training target)")
print("   - y_test.csv (testing target)")

print("="*60) """



# preprocessing.py - Enhanced Unified Preprocessing Pipeline

import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path
from config import *

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FlightDataPreprocessor:
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.pca = None
        self.feature_names = []
        self.is_fitted = False
        
    def load_data(self, filepath):
        """Load and validate dataset"""
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Successfully loaded data with shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def clean_data(self, df):
        """Clean and validate data"""
        logger.info("Starting data cleaning...")
        
        # Remove unwanted columns
        df_clean = df.drop(['flight'], axis=1, errors='ignore')
        
        # Handle missing values
        for col in df_clean.select_dtypes(include=[np.number]).columns:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
                
        for col in df_clean.select_dtypes(include=['object']).columns:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
        
        # Remove duplicates
        df_clean = df_clean.drop_duplicates()
        
        logger.info(f"Data cleaning completed. Shape: {df_clean.shape}")
        return df_clean
    
    def encode_categorical_features(self, df, fit=True):
        """Encode categorical features consistently"""
        logger.info("Encoding categorical features...")
        
        df_encoded = df.copy()
        
        # Encode standard categorical features
        for col in CATEGORICAL_FEATURES:
            if col in df_encoded.columns:
                if fit:
                    if col not in self.encoders:
                        self.encoders[col] = LabelEncoder()
                    df_encoded[f'{col}_encoded'] = self.encoders[col].fit_transform(df_encoded[col])
                else:
                    if col in self.encoders:
                        try:
                            df_encoded[f'{col}_encoded'] = self.encoders[col].transform(df_encoded[col])
                        except ValueError:
                            mode_val = self.encoders[col].classes_[0]
                            df_encoded[col] = df_encoded[col].fillna(mode_val)
                            df_encoded.loc[~df_encoded[col].isin(self.encoders[col].classes_), col] = mode_val
                            df_encoded[f'{col}_encoded'] = self.encoders[col].transform(df_encoded[col])
        
        # Handle stops encoding
        if 'stops' in df_encoded.columns:
            df_encoded['stops_encoded'] = df_encoded['stops'].map(STOPS_MAPPING)
            df_encoded['stops_encoded'] = df_encoded['stops_encoded'].fillna(0)  # Default to 'zero'
        
        return df_encoded
    
    def engineer_features(self, df):
        """Create additional engineered features"""
        logger.info("Engineering additional features...")
        
        df_features = df.copy()
        
        # Weekend indicator (simplified)
        if 'days_left' in df_features.columns:
            df_features['is_weekend'] = (df_features['days_left'] % 7).isin([0, 6]).astype(int)
        
        # Peak time indicator
        if 'departure_time' in df_features.columns:
            peak_times = ['Morning', 'Evening']
            df_features['is_peak'] = df_features['departure_time'].isin(peak_times).astype(int)
        
        # Route popularity (only during training)
        if self.is_fitted and 'price' in df_features.columns:
            route_cols = ['source_city', 'destination_city', 'class', 'stops']
            if all(col in df_features.columns for col in route_cols):
                df_features['route_avg_price'] = df_features.groupby(route_cols)['price'].transform('mean')
        else:
            df_features['route_avg_price'] = 0  # Default for new data
        
        return df_features
    
    def scale_features(self, df, fit=True):
        """Scale numerical features"""
        logger.info("Scaling numerical features...")
        
        df_scaled = df.copy()
        
        for col in NUMERICAL_FEATURES:
            if col in df_scaled.columns:
                if fit:
                    if col not in self.scalers:
                        self.scalers[col] = StandardScaler()
                    df_scaled[[col]] = self.scalers[col].fit_transform(df_scaled[[col]])
                else:
                    if col in self.scalers:
                        df_scaled[[col]] = self.scalers[col].transform(df_scaled[[col]])
        
        return df_scaled
    
    def apply_pca(self, X, fit=True):
        """Apply PCA transformation"""
        logger.info("Applying PCA transformation...")
        
        if fit:
            self.pca = PCA(n_components=PCA_VARIANCE_THRESHOLD, random_state=RANDOM_STATE)
            X_pca = self.pca.fit_transform(X)
            logger.info(f"PCA reduced features from {X.shape[1]} to {X_pca.shape[1]}")
            logger.info(f"Variance explained: {self.pca.explained_variance_ratio_.sum():.4f}")
        else:
            if self.pca is None:
                raise ValueError("PCA not fitted. Call fit_transform first.")
            X_pca = self.pca.transform(X)
        
        return X_pca
    
    def fit_transform(self, df, target_col='price'):
        """Complete preprocessing pipeline for training data"""
        logger.info("Starting complete preprocessing pipeline...")
        
        # Clean data
        df_clean = self.clean_data(df)
        
        # Encode categorical features
        df_encoded = self.encode_categorical_features(df_clean, fit=True)
        
        # Engineer features
        df_features = self.engineer_features(df_encoded)
        
        # Scale features
        df_scaled = self.scale_features(df_features, fit=True)
        
        # Prepare feature matrix
        feature_cols = [col for col in df_scaled.columns 
                       if col.endswith('_encoded') or col in ['duration', 'days_left', 'is_weekend', 'is_peak', 'route_avg_price']]
        
        X = df_scaled[feature_cols].fillna(0)  # Handle any remaining NaNs
        self.feature_names = feature_cols
        
        # Apply PCA
        X_pca = self.apply_pca(X, fit=True)
        
        # Create final dataframe
        pca_cols = [f'PC{i+1}' for i in range(X_pca.shape[1])]
        final_df = pd.DataFrame(X_pca, columns=pca_cols)
        
        if target_col in df_scaled.columns:
            final_df[target_col] = df_scaled[target_col].values
            
        if 'airline' in df_clean.columns:
            final_df['airline'] = df_clean['airline'].values
        
        self.is_fitted = True
        logger.info(f"Preprocessing completed. Final shape: {final_df.shape}")
        
        return final_df
    
    def transform(self, df):
        """Transform new data using fitted preprocessor"""
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")
        
        logger.info("Transforming new data...")
        
        # Apply same pipeline without fitting
        df_clean = self.clean_data(df)
        df_encoded = self.encode_categorical_features(df_clean, fit=False)
        df_features = self.engineer_features(df_encoded)
        df_scaled = self.scale_features(df_features, fit=False)
        
        # Select same features
        X = df_scaled[self.feature_names].fillna(0)
        
        # Apply PCA
        X_pca = self.apply_pca(X, fit=False)
        
        return X_pca
    
    def save(self, filepath):
        """Save preprocessor state"""
        preprocessor_state = {
            'scalers': self.scalers,
            'encoders': self.encoders,
            'pca': self.pca,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }
        joblib.dump(preprocessor_state, filepath)
        logger.info(f"Preprocessor saved to {filepath}")
    
    def load(self, filepath):
        """Load preprocessor state"""
        preprocessor_state = joblib.load(filepath)
        self.scalers = preprocessor_state['scalers']
        self.encoders = preprocessor_state['encoders'] 
        self.pca = preprocessor_state['pca']
        self.feature_names = preprocessor_state['feature_names']
        self.is_fitted = preprocessor_state['is_fitted']
        logger.info(f"Preprocessor loaded from {filepath}")

def main():
    """Main preprocessing pipeline"""
    print("="*60)
    print("FLIGHT PRICE PREDICTION - DATA PREPROCESSING")
    print("="*60)
    
    # Initialize preprocessor
    preprocessor = FlightDataPreprocessor()
    
    # Load and preprocess data
    try:
        df = preprocessor.load_data(RAW_DATA_FILE)
        final_df = preprocessor.fit_transform(df)
        
        # Save preprocessed data
        final_df.to_csv(PROCESSED_DATA_FILE, index=False)
        
        # Split data for ML
        X = final_df.drop(['price', 'airline'], axis=1, errors='ignore')
        y = final_df['price'] if 'price' in final_df.columns else None
        
        if y is not None:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
            )
            
            # Save split data
            X_train.to_csv(TRAIN_TEST_FILES['X_train'], index=False)
            X_test.to_csv(TRAIN_TEST_FILES['X_test'], index=False)
            y_train.to_csv(TRAIN_TEST_FILES['y_train'], index=False)
            y_test.to_csv(TRAIN_TEST_FILES['y_test'], index=False)
            
            print(f"✅ Data split: Train {X_train.shape}, Test {X_test.shape}")
        
        # Save preprocessor
        preprocessor.save(MODELS_DIR / 'preprocessor.pkl')
        
        print("✅ Preprocessing completed successfully!")
        print(f"✅ Final data shape: {final_df.shape}")
        print(f"✅ Files saved: {PROCESSED_DATA_FILE}")
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise

if __name__ == "__main__":
    main()
