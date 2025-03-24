# Install Dependencies
!pip install pandas numpy tensorflow keras nltk scikit-learn

# Import Libraries
import pandas as pd
import numpy as np
import tensorflow as tf
import nltk
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Download Stopwords
nltk.download('stopwords')

# Load Dataset
def load_data(filepath):
    df = pd.read_csv(filepath)
    df = df[['text', 'label']]  # Ensure correct columns
    return df

# Preprocess Text Data
def preprocess_text(df):
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(df['text'])

    sequences = tokenizer.texts_to_sequences(df['text'])
    max_length = 100  # Adjust based on dataset
    X = pad_sequences(sequences, maxlen=max_length, padding='post')

    y = np.array(df['label'])  # Labels (0: Real, 1: Fake)
    return X, y, tokenizer, max_length

# Train the BiLSTM Model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Sequential([
        Embedding(input_dim=5000, output_dim=100, input_length=X.shape[1]),
        Bidirectional(LSTM(100, return_sequences=True)),
        Dropout(0.5),
        Bidirectional(LSTM(50)),
        Dense(1, activation='sigmoid')  # Binary classification
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
    
    model.save("fake_news_model.h5")
    print("✅ Model training complete and saved!")

    return model

# Predict Fake News
def predict_fake_news(model, tokenizer, max_length, news_text):
    sequence = tokenizer.texts_to_sequences([news_text])
    sequence = pad_sequences(sequence, maxlen=max_length, padding='post')
    prediction = model.predict(sequence)[0][0]
    return "🛑 Fake News" if prediction > 0.5 else "✅ Real News"

# Run Everything
if __name__ == "__main__":
    dataset_path = "dataset/fake_news.csv"  # Replace with your dataset path
    df = load_data(dataset_path)
    X, y, tokenizer, max_length = preprocess_text(df)
    model = train_model(X, y)

    # Example Test
    sample_news = "Breaking: Scientists discover a new way to generate clean energy from ocean waves!"
    print(f"Prediction: {predict_fake_news(model, tokenizer, max_length, sample_news)}")
