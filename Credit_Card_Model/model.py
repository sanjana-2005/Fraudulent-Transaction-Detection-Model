import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
import joblib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

def load_data(path):
    logging.info("Loading dataset...")
    data = pd.read_csv(path)
    return data

def preprocess_data(data):
    logging.info("Preprocessing data...")
    features = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
    X = data[features].copy()
    y = data['Class'].copy()

    # PCA for dimensionality reduction
    pca = PCA(n_components=10, random_state=42)
    reduced_features = pca.fit_transform(X[[f'V{i}' for i in range(1, 29)]])  # Apply PCA only on V1-V28

    # Save the PCA model
    joblib.dump(pca, r"C:\Users\shalu\Downloads\pca.pkl")  # Save the PCA model

    # KMeans clustering
    kmeans = KMeans(n_clusters=5, n_init=5, random_state=42)
    clusters = kmeans.fit_predict(reduced_features)
    X['Cluster'] = clusters

    # Scale 'Time' and 'Amount'
    scaler = StandardScaler()
    X[['Time', 'Amount']] = scaler.fit_transform(X[['Time', 'Amount']])

    return X, y, scaler, kmeans, pca

def handle_imbalance(X, y):
    logging.info("Handling class imbalance with SMOTE...")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

def train_model(X_train, y_train):
    logging.info("Training RandomForest model with RandomizedSearchCV...")
    param_dist = {
        'n_estimators': [50, 75, 100],
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2']
    }

    random_search = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced'),
        param_distributions=param_dist,
        n_iter=5,
        cv=3,
        n_jobs=-1,
        random_state=42
    )
    random_search.fit(X_train, y_train)
    return random_search.best_estimator_

def evaluate_model(model, X_train, X_test, y_train, y_test):
    logging.info("Evaluating model...")
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    logging.info(f"Training Accuracy: {train_accuracy * 100:.2f}%")
    logging.info(f"Testing Accuracy: {test_accuracy * 100:.2f}%")
    logging.info("\nClassification Report (Training Data):\n" + classification_report(y_train, y_train_pred))
    logging.info("\nClassification Report (Testing Data):\n" + classification_report(y_test, y_test_pred))

    return train_accuracy, test_accuracy

def save_artifacts(model, scaler, kmeans, pca, model_path, scaler_path, kmeans_path, pca_path):
    logging.info("Saving model, scaler, KMeans, and PCA...")
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(kmeans, kmeans_path)
    joblib.dump(pca, pca_path)  # Save the PCA model
    logging.info("Artifacts saved successfully.")

# Main workflow
if __name__ == "__main__":
    data_path = r"C:\Users\shalu\Downloads\creditcard.csv"
    model_path = r"C:\Users\shalu\Downloads\quantum_probability_classifier_model.pkl"
    scaler_path = r"C:\Users\shalu\Downloads\scaler.pkl"
    kmeans_path = r"C:\Users\shalu\Downloads\kmeans.pkl"
    pca_path = r"C:\Users\shalu\Downloads\pca.pkl"  # Path for PCA model

    data = load_data(data_path)
    X, y, scaler, kmeans, pca = preprocess_data(data)  # Now we also return PCA
    X_resampled, y_resampled = handle_imbalance(X[['Time', 'Amount', 'Cluster']], y)

    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_train, X_test, y_train, y_test)
    save_artifacts(model, scaler, kmeans, pca, model_path, scaler_path, kmeans_path, pca_path)
