"""Optional deep-learning training script.

Install requirements-ml.txt before running this file.
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding, SpatialDropout1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

from complaint_ai.data.ingest import filter_training_rows, load_complaints
from complaint_ai.data.preprocess import prepare_text_classification_frame


def train_lstm(input_path: str, output_dir: str = "models/lstm", max_words: int = 40000) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    df = prepare_text_classification_frame(filter_training_rows(load_complaints(input_path)))
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(df["text_clean"])
    X = pad_sequences(tokenizer.texts_to_sequences(df["text_clean"]), maxlen=300)

    label_encoder = LabelEncoder()
    y = to_categorical(label_encoder.fit_transform(df["label_clean"]))

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Sequential(
        [
            Embedding(max_words, 128, input_length=300),
            SpatialDropout1D(0.2),
            LSTM(128, dropout=0.2, recurrent_dropout=0.2),
            Dense(128, activation="relu"),
            Dropout(0.3),
            Dense(y.shape[1], activation="softmax"),
        ]
    )
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=5,
        batch_size=128,
        callbacks=[EarlyStopping(patience=2, restore_best_weights=True)],
    )
    model.save(output / "lstm_model.keras")
