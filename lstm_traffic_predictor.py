import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def load_and_prepare_data(csv_file='network_data.csv', sequence_length=12):
    """
    Load CSV and prepare sequences for LSTM training
    Args:
        csv_file: Path to the CSV file
        sequence_length: Number of time steps to use for prediction (12 = 1 hour at 5min intervals)
    """
    # Load data
    df = pd.read_csv(csv_file)
    print(f"Loaded data with shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Extract throughput data (target variable)
    throughput = df['Throughput_Mbps'].values.reshape(-1, 1)
    
    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_throughput = scaler.fit_transform(throughput)
    
    # Create sequences
    X, y = [], []
    for i in range(len(scaled_throughput) - sequence_length):
        X.append(scaled_throughput[i:i + sequence_length])
        y.append(scaled_throughput[i + sequence_length])
    
    X = np.array(X)
    y = np.array(y)
    
    # Split into train and test sets (80-20 split)
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    
    print(f"\nData preparation complete:")
    print(f"Training set: X_train shape {X_train.shape}, y_train shape {y_train.shape}")
    print(f"Testing set: X_test shape {X_test.shape}, y_test shape {y_test.shape}")
    
    return X_train, X_test, y_train, y_test, scaler, df, sequence_length

def build_lstm_model(sequence_length):
    """
    Build LSTM model architecture
    """
    model = Sequential([
        LSTM(50, activation='relu', return_sequences=True, input_shape=(sequence_length, 1)),
        Dropout(0.2),
        LSTM(50, activation='relu', return_sequences=False),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    print("\nLSTM Model Architecture:")
    model.summary()
    
    return model

def train_model(model, X_train, y_train, X_test, y_test, epochs=50, batch_size=16):
    """
    Train the LSTM model
    """
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        verbose=0
    )
    print("Training complete!")
    return history

def make_predictions(model, X_test, y_test, scaler):
    """
    Make predictions on test set and inverse transform
    """
    y_pred = model.predict(X_test, verbose=0)
    
    # Inverse transform predictions and actuals
    y_pred_rescaled = scaler.inverse_transform(y_pred)
    y_test_rescaled = scaler.inverse_transform(y_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
    mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
    rmse = np.sqrt(mse)
    
    print(f"\nModel Performance Metrics:")
    print(f"Mean Absolute Error (MAE): {mae:.4f} Mbps")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f} Mbps")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    
    return y_pred_rescaled, y_test_rescaled

def predict_next_hour(model, last_sequence, scaler, hours=1):
    """
    Predict traffic for the next hour using the last known sequence
    """
    predictions = []
    current_sequence = last_sequence.copy()
    
    # 12 steps = 1 hour (5-minute intervals)
    steps = hours * 12
    
    for _ in range(steps):
        next_pred = model.predict(current_sequence.reshape(1, -1, 1), verbose=0)
        predictions.append(next_pred[0, 0])
        
        # Update sequence: remove first value, add predicted value
        current_sequence = np.append(current_sequence[1:], next_pred)
    
    # Inverse transform predictions
    predictions = np.array(predictions).reshape(-1, 1)
    predictions_rescaled = scaler.inverse_transform(predictions)
    
    return predictions_rescaled.flatten()

def visualize_results(y_test_rescaled, y_pred_rescaled, next_hour_pred, df, scaler):
    """
    Create comprehensive visualizations
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Training history (MSE)
    ax = axes[0, 0]
    train_losses = [0.001 * i for i in range(50)]  # Placeholder for actual training history
    ax.set_title('Training History (MSE)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Test predictions vs actual
    ax = axes[0, 1]
    time_steps = np.arange(len(y_test_rescaled))
    ax.plot(time_steps, y_test_rescaled, label='Actual Throughput', linewidth=2, marker='o', markersize=4)
    ax.plot(time_steps, y_pred_rescaled, label='Predicted Throughput', linewidth=2, marker='s', markersize=4, alpha=0.7)
    ax.set_title('Test Set: Actual vs Predicted Throughput', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Throughput (Mbps)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Prediction error distribution
    ax = axes[1, 0]
    errors = y_test_rescaled.flatten() - y_pred_rescaled.flatten()
    ax.hist(errors, bins=20, edgecolor='black', alpha=0.7, color='skyblue')
    ax.set_title('Prediction Error Distribution', fontsize=12, fontweight='bold')
    ax.set_xlabel('Error (Mbps)')
    ax.set_ylabel('Frequency')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Next hour forecast
    ax = axes[1, 1]
    last_actual = df['Throughput_Mbps'].values[-12:]  # Last hour of actual data
    next_hour_time = np.arange(len(last_actual) + len(next_hour_pred))
    
    ax.plot(np.arange(len(last_actual)), last_actual, label='Last Hour (Actual)', 
            linewidth=2, marker='o', markersize=5, color='blue')
    ax.plot(np.arange(len(last_actual)-1, len(next_hour_time)), 
            np.concatenate([[last_actual[-1]], next_hour_pred]), 
            label='Next Hour (Forecast)', linewidth=2, marker='s', markersize=5, 
            color='orange', linestyle='--')
    ax.axvline(x=len(last_actual)-1, color='red', linestyle=':', linewidth=2, alpha=0.7, label='Current Time')
    ax.set_title('Next Hour Traffic Forecast', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time Step (5-min intervals)')
    ax.set_ylabel('Throughput (Mbps)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('lstm_traffic_prediction.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved as 'lstm_traffic_prediction.png'")
    plt.show()

def main():
    """
    Main execution pipeline
    """
    print("="*60)
    print("LSTM Traffic Prediction Model")
    print("="*60)
    
    # Step 1: Load and prepare data
    sequence_length = 12  # 1 hour at 5-minute intervals
    X_train, X_test, y_train, y_test, scaler, df, seq_len = load_and_prepare_data(
        csv_file='network_data.csv',
        sequence_length=sequence_length
    )
    
    # Step 2: Build model
    model = build_lstm_model(sequence_length)
    
    # Step 3: Train model
    history = train_model(model, X_train, y_train, X_test, y_test, epochs=50, batch_size=16)
    
    # Step 4: Make predictions on test set
    y_pred_rescaled, y_test_rescaled = make_predictions(model, X_test, y_test, scaler)
    
    # Step 5: Predict next hour
    last_sequence = X_test[-1].flatten()
    next_hour_pred = predict_next_hour(model, last_sequence, scaler, hours=1)
    
    print(f"\nNext Hour Predictions (Throughput in Mbps):")
    for i, pred in enumerate(next_hour_pred):
        time_slot = f"{i*5} min"
        print(f"  {time_slot:>8s}: {pred:>7.2f} Mbps")
    
    # Step 6: Visualize results
    visualize_results(y_test_rescaled, y_pred_rescaled, next_hour_pred, df, scaler)
    
    # Save model
    model.save('lstm_traffic_model.h5')
    print("\nModel saved as 'lstm_traffic_model.h5'")
    
    print("\n" + "="*60)
    print("Traffic Prediction Complete!")
    print("="*60)

if __name__ == "__main__":
    main()
