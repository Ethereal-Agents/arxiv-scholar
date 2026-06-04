import pandas as pd
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import sys
import os

# Add src and root to path so we can import arxiv_scholar and configs
sys.path.append(os.path.abspath("src"))
sys.path.append(os.path.abspath("."))

from arxiv_scholar.embedding.st_embedder import SentenceTransformerEmbedder

def main():
    # 1. Load your collected dataset
    with open('data/router_dataset/queries.json', 'r') as f:
        data = json.load(f)

    # Convert labels from strings to integers
    # DECOMPOSE (Class 1) and DIRECT (Class 0)
    for item in data:
        item['label'] = 1 if item['label'] == 'DECOMPOSE' else 0

    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} queries.")

    # 2. Initialize embedding model (Using the same one as production!)
    print("Loading SentenceTransformerEmbedder('BAAI/bge-m3')...")
    embedding_model = SentenceTransformerEmbedder("BAAI/bge-m3")

    # 3. Generate vectors for the training set
    print("Generating embeddings...")
    embeddings_list = embedding_model.embed(df["text"].tolist())
    X = np.array(embeddings_list)
    y = df["label"].values

    print(f"X is now a matrix of shape {X.shape}")
    print(f"y is a 1D array of shape {y.shape}")

    # 4. Train the classifier
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    classifier = LogisticRegression(C=1.0, max_iter=1000)
    classifier.fit(X_train, y_train)

    # 5. Check performance
    accuracy = classifier.score(X_test, y_test)
    print(f"Classifier Test Accuracy: {accuracy * 100:.2f}%")

    # 6. Save the trained model artifact to disk
    joblib.dump(classifier, "data/router_dataset/query_router_model.joblib")
    print("Model saved to data/router_dataset/query_router_model.joblib")

if __name__ == "__main__":
    main()
