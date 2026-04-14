import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from scaler import scale_deployment

MIN_REPLICAS = 1
MAX_REPLICAS = 3
HIGH_TRAFFIC_THRESHOLD = 70
SEQ_LEN = 10


def build_sequences(data, seq_len=10):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len])
    return np.array(X), np.array(y)


def load_and_scale_data(filepath):
    df = pd.read_csv(filepath)
    if 'user_count' not in df.columns:
        raise ValueError(f"'user_count' column not found in {filepath}")
    data = df['user_count'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler


def build_model(seq_len):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(seq_len, 1)),
        LSTM(50),
        Dense(1),
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


def predict_and_scale(model, scaler, X):
    last_sequence = X[-1].reshape(1, SEQ_LEN, 1)
    predicted_scaled = model.predict(last_sequence, verbose=0)
    predicted_users = int(scaler.inverse_transform(predicted_scaled)[0][0])
    predicted_users = max(0, predicted_users)

    print(f"🔮 Predicted future user count: {predicted_users}")

    if predicted_users > HIGH_TRAFFIC_THRESHOLD:
        print(f"⚠️ High traffic predicted ({predicted_users} users)! Scaling up to {MAX_REPLICAS} replicas.")
        scale_deployment(MAX_REPLICAS)
    else:
        print(f"✅ Normal traffic ({predicted_users} users). Scaling down to {MIN_REPLICAS} replica.")
        scale_deployment(MIN_REPLICAS)


def main():
    scaled_data, scaler = load_and_scale_data('network_data.csv')

    X, y = build_sequences(scaled_data, SEQ_LEN)
    if len(X) == 0:
        raise ValueError("X is empty — ensure data generation ran successfully.")

    model = build_model(SEQ_LEN)
    model.fit(X, y, epochs=5, batch_size=32, verbose=1)
    print("🧠 AI Model is ready to predict traffic!")

    predict_and_scale(model, scaler, X)


if __name__ == "__main__":
    main()