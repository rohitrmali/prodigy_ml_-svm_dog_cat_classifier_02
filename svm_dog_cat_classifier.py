import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import cv2
import os
from tqdm import tqdm
import pickle

class DogCatSVMClassifier:
    def __init__(self, img_size=(64, 64)):
        """
        Initialize the SVM classifier for dog/cat images
        
        Args:
            img_size: Tuple of (height, width) for resizing images
        """
        self.img_size = img_size
        self.scaler = StandardScaler()
        self.svm_model = svm.SVC(kernel='rbf', C=10, gamma='scale', 
                                  probability=True, random_state=42)
        
    def extract_features(self, img_path):
        """
        Extract features from an image
        
        Args:
            img_path: Path to the image file
            
        Returns:
            Flattened feature vector
        """
        try:
            # Read image
            img = cv2.imread(img_path)
            if img is None:
                return None
            
            # Resize image
            img = cv2.resize(img, self.img_size)
            
            # Convert to grayscale for simpler features
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Flatten the image
            features = gray.flatten()
            
            return features
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            return None
    
    def load_dataset(self, train_path, limit=None):
        """
        Load and preprocess the dataset
        
        Args:
            train_path: Path to training directory containing 'dogs' and 'cats' folders
            limit: Optional limit on number of images per class
            
        Returns:
            X: Feature matrix, y: Labels
        """
        X = []
        y = []
        
        # Load cat images (label 0)
        cat_path = os.path.join(train_path, 'cats')
        if os.path.exists(cat_path):
            cat_files = os.listdir(cat_path)[:limit] if limit else os.listdir(cat_path)
            print(f"Loading {len(cat_files)} cat images...")
            for img_name in tqdm(cat_files):
                if img_name.endswith(('.jpg', '.jpeg', '.png')):
                    features = self.extract_features(os.path.join(cat_path, img_name))
                    if features is not None:
                        X.append(features)
                        y.append(0)  # Cat
        
        # Load dog images (label 1)
        dog_path = os.path.join(train_path, 'dogs')
        if os.path.exists(dog_path):
            dog_files = os.listdir(dog_path)[:limit] if limit else os.listdir(dog_path)
            print(f"Loading {len(dog_files)} dog images...")
            for img_name in tqdm(dog_files):
                if img_name.endswith(('.jpg', '.jpeg', '.png')):
                    features = self.extract_features(os.path.join(dog_path, img_name))
                    if features is not None:
                        X.append(features)
                        y.append(1)  # Dog
        
        # Alternative: Load from single directory with filenames like 'cat.1.jpg', 'dog.1.jpg'
        if not os.path.exists(cat_path) and not os.path.exists(dog_path):
            print(f"Loading images from {train_path}...")
            files = os.listdir(train_path)
            if limit:
                files = files[:limit*2]
            
            for img_name in tqdm(files):
                if img_name.endswith(('.jpg', '.jpeg', '.png')):
                    features = self.extract_features(os.path.join(train_path, img_name))
                    if features is not None:
                        X.append(features)
                        # Determine label from filename
                        if img_name.startswith('cat'):
                            y.append(0)
                        elif img_name.startswith('dog'):
                            y.append(1)
        
        return np.array(X), np.array(y)
    
    def train(self, X_train, y_train):
        """
        Train the SVM model
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        print("Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        print("Training SVM model...")
        self.svm_model.fit(X_train_scaled, y_train)
        print("Training complete!")
    
    def predict(self, X_test):
        """
        Make predictions on test data
        
        Args:
            X_test: Test features
            
        Returns:
            Predictions
        """
        X_test_scaled = self.scaler.transform(X_test)
        return self.svm_model.predict(X_test_scaled)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: True labels
        """
        predictions = self.predict(X_test)
        
        accuracy = accuracy_score(y_test, predictions)
        print(f"\nAccuracy: {accuracy:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, predictions, 
                                   target_names=['Cat', 'Dog']))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, predictions)
        print(cm)
        
        return accuracy, predictions
    
    def save_model(self, filepath):
        """Save the trained model"""
        with open(filepath, 'wb') as f:
            pickle.dump({'model': self.svm_model, 'scaler': self.scaler, 
                        'img_size': self.img_size}, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.svm_model = data['model']
            self.scaler = data['scaler']
            self.img_size = data['img_size']
        print(f"Model loaded from {filepath}")


# Example usage
if __name__ == "__main__":
    # Initialize classifier
    classifier = DogCatSVMClassifier(img_size=(64, 64))
    
    # Path to your dataset
    # Option 1: Dataset structure: train_path/cats/ and train_path/dogs/
    # Option 2: Dataset structure: train_path/ with files named cat.*.jpg and dog.*.jpg
    train_path = "path/to/your/dataset"
    
    # Load dataset (use limit for faster training during testing)
    print("Loading dataset...")
    X, y = classifier.load_dataset(train_path, limit=1000)  # Remove limit for full dataset
    
    print(f"\nDataset loaded: {len(X)} images")
    print(f"Feature shape: {X[0].shape}")
    print(f"Class distribution: Cats={np.sum(y==0)}, Dogs={np.sum(y==1)}")
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set: {len(X_train)} images")
    print(f"Test set: {len(X_test)} images")
    
    # Train the model
    classifier.train(X_train, y_train)
    
    # Evaluate the model
    print("\n" + "="*50)
    print("Evaluating on test set...")
    print("="*50)
    classifier.evaluate(X_test, y_test)
    
    # Save the model
    classifier.save_model('dog_cat_svm_model.pkl')
    
    # Example: Predict on a single image
    # test_img_path = "path/to/test/image.jpg"
    # features = classifier.extract_features(test_img_path)
    # if features is not None:
    #     prediction = classifier.predict(features.reshape(1, -1))
    #     label = "Dog" if prediction[0] == 1 else "Cat"
    #     print(f"Prediction: {label}")
