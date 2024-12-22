import csv
import numpy as np
import pandas as pd
import regex as re
import contractions
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import pairwise_distances, accuracy_score
from scipy.stats import mode
from tqdm import trange

# Download and configure stopwords once
nltk.download('stopwords')
stopwords = set(stopwords.words('english'))  # Use a set for O(1) lookup
# Remove specific stopwords 'no', 'nor', and 'not'
stopwords -= {'no', 'nor', 'not'}

# Function to preprocess reviews
def preprocess_reviews(reviews, stopwords):
    processed_reviews = []
    for review in tqdm(reviews):
        if isinstance(review, str):
            review = re.sub('(<[\w\s]*/?>)', "", review)  # Remove HTML tags
            review = contractions.fix(review)  # Expand contractions
            review = re.sub('[^a-zA-Z0-9\s]+', "", review)  # Remove special characters
            review = re.sub('\d+', "", review)  # Remove digits
            # Lowercase, remove stopwords and short words
            processed_reviews.append(
                " ".join([word.lower() for word in review.split() if word not in stopwords and len(word) >= 3])
            )
        else:
            processed_reviews.append("")  # Handle non-string entries
    return processed_reviews

# Load and preprocess the training dataset
df = pd.read_csv('/scratch/mbiswas2/CS 580/train.csv')
df['Processed_Reviews'] = preprocess_reviews(df['Reviews'], stopwords)
df['Processed_Reviews'] = df['Processed_Reviews'].str.strip()
processed_df = pd.DataFrame({'Reviews': df['Processed_Reviews'], 'Ratings': df['Ratings']})

# Load and preprocess the test dataset
df1 = pd.read_csv('/scratch/mbiswas2/CS 580/test.csv')
df1['Processed_Reviews'] = preprocess_reviews(df1['Reviews'], stopwords)
df1['Processed_Reviews'] = df1['Processed_Reviews'].str.strip()
test_data = df1

# Vectorization using TfidfVectorizer
combined_data = pd.concat([processed_df['Reviews'], test_data['Processed_Reviews']], ignore_index=True)
vectorizer = TfidfVectorizer(lowercase=True, stop_words='english', ngram_range=(1, 2), max_df=0.9, min_df=5, max_features=10000, sublinear_tf=True)
X_combined = vectorizer.fit_transform(combined_data)

# Split into training and test sets
n_train_samples = len(processed_df)
X_train = X_combined[:n_train_samples]
X_test = X_combined[n_train_samples:]

# Labels - Convert Ratings to numpy array
y_train = processed_df['Ratings'].values

# KNN Implementation optimized for better batch management
class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train  # Sparse matrix for training data
        self.y_train = np.array(y_train)  # Ensure labels are in numpy array

    def predict(self, X_test, batch_size=1000):  # Increase batch size for performance
        predictions = []
        n_test_samples = X_test.shape[0]
        
        for start in tqdm(range(0, n_test_samples, batch_size)):
            end = min(start + batch_size, n_test_samples)
            X_test_batch = X_test[start:end]

            # Efficient distance calculation using cosine similarity
            distances = pairwise_distances(X_test_batch, self.X_train, metric='cosine', n_jobs=-1)

            # Find the k nearest neighbors
            nearest_indices = np.argpartition(distances, self.k, axis=1)[:, :self.k]
            nearest_labels = self.y_train[nearest_indices]

            # Determine the most common label for each test sample
            batch_predictions = mode(nearest_labels, axis=1)[0].flatten()
            predictions.extend(batch_predictions)
        
        return predictions



# Custom K-Fold Cross Validation Implementation
def custom_cross_validation(X, y, k, n_splits=5, batch_size=1000):
    fold_size = X.shape[0] // n_splits
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    
    accuracies = []
    
    for fold in range(n_splits):
        print(f"Processing fold {fold + 1}/{n_splits}")
        
        # Create train and validation indices for the current fold
        val_start = fold * fold_size
        val_end = (fold + 1) * fold_size if fold != n_splits - 1 else X.shape[0]
        
        val_indices = indices[val_start:val_end]
        train_indices = np.concatenate([indices[:val_start], indices[val_end:]])
        
        # Split the data
        X_train_fold = X[train_indices]
        X_val_fold = X[val_indices]
        y_train_fold = y[train_indices]
        y_val_fold = y[val_indices]
        
        # Train the KNN model on the training fold
        knn = KNN(k=k)
        knn.fit(X_train_fold, y_train_fold)
        
        # Predict on the validation fold
        val_predictions = knn.predict(X_val_fold, batch_size=batch_size)
        
        # Calculate accuracy for this fold
        accuracy = accuracy_score(y_val_fold, val_predictions)
        accuracies.append(accuracy)
        
        print(f"Fold {fold + 1} accuracy: {accuracy:.4f}")
    
    # Compute the average accuracy across all folds
    average_accuracy = np.mean(accuracies)
    print(f"Average accuracy across {n_splits} folds: {average_accuracy:.4f}")
    
    return average_accuracy

# Perform custom cross-validation
# average_accuracy = custom_cross_validation(X_train, y_train, k=700, n_splits=5, batch_size=1000)



# Final KNN model on full training data
knn = KNN(k=700)
knn.fit(X_train, y_train)

# Predict on the test set
test_predictions = knn.predict(X_test, batch_size=1000)

# Save predictions to file as integers
with open('/home/mbiswas2/CS_584/result.dat', 'w') as prediction_file:
    for prediction in test_predictions:
        prediction_file.write(f"{str(prediction)}\n")

print(f"Predictions for {X_test.shape[0]} samples have been saved to 'result.dat'")
