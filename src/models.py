import numpy as np
import pandas as pd
import pmdarima as pm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler

# --- Metrics ---
def calculate_metrics(y_true, y_pred):
    """
    Calculates MAE, RMSE, and MAPE.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    # Avoid division by zero
    non_zero = y_true != 0
    mape = np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100
    
    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape
    }

# --- Classical Models ---
def fit_auto_arima(train_series, m=1):
    """
    Fits an auto_arima model and returns the fitted model.
    """
    print("Fitting auto_arima...")
    model = pm.auto_arima(train_series, seasonal=m > 1, m=m, trace=True,
                          error_action='ignore', suppress_warnings=True)
    print(f"Best ARIMA order: {model.order} Seasonal order: {model.seasonal_order}")
    return model

def forecast_arima(model, steps):
    """
    Generates forecasts and confidence intervals from a fitted auto_arima model.
    """
    forecasts, conf_int = model.predict(n_periods=steps, return_conf_int=True)
    return forecasts, conf_int

# --- Deep Learning (NumPy LSTM) ---
# A clean, custom LSTM implementation in pure NumPy to avoid any PyTorch/TensorFlow DLL loading errors.
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

class NumpyLSTM:
    def __init__(self, input_dim=1, hidden_dim=16, output_dim=1):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # LSTM weights
        limit = np.sqrt(1.0 / (input_dim + hidden_dim))
        self.W = np.random.uniform(-limit, limit, (input_dim + hidden_dim, 4 * hidden_dim))
        self.b = np.zeros((1, 4 * hidden_dim))
        
        # Dense output weights
        self.W_out = np.random.uniform(-np.sqrt(1.0 / hidden_dim), np.sqrt(1.0 / hidden_dim), (hidden_dim, output_dim))
        self.b_out = np.zeros((1, output_dim))

    def _forward_sequence(self, x_seq):
        # x_seq: (T, input_dim)
        T = x_seq.shape[0]
        h = np.zeros((T + 1, self.hidden_dim))
        c = np.zeros((T + 1, self.hidden_dim))
        
        f_gates = []
        i_gates = []
        c_tilde_gates = []
        o_gates = []
        
        for t in range(T):
            # Combine input and previous hidden state
            combined = np.hstack((x_seq[t].reshape(1, -1), h[t].reshape(1, -1))) # (1, input_dim + hidden_dim)
            gates = np.dot(combined, self.W) + self.b # (1, 4 * hidden_dim)
            
            f = sigmoid(gates[:, :self.hidden_dim])
            i = sigmoid(gates[:, self.hidden_dim:2*self.hidden_dim])
            c_tilde = np.tanh(gates[:, 2*self.hidden_dim:3*self.hidden_dim])
            o = sigmoid(gates[:, 3*self.hidden_dim:])
            
            c[t+1] = f * c[t] + i * c_tilde
            h[t+1] = o * np.tanh(c[t+1])
            
            f_gates.append(f)
            i_gates.append(i)
            c_tilde_gates.append(c_tilde)
            o_gates.append(o)
            
        y_pred = np.dot(h[-1].reshape(1, -1), self.W_out) + self.b_out
        
        cache = {
            "x_seq": x_seq, "h": h, "c": c,
            "f": f_gates, "i": i_gates, "c_tilde": c_tilde_gates, "o": o_gates
        }
        return y_pred[0, 0], cache

    def train_model(self, X, y, epochs=5, lr=0.01):
        """
        Trains the model using simplified backpropagation through time.
        """
        # Downsample to the most recent 400 days for fast training on CPU
        if len(X) > 400:
            X = X[-400:]
            y = y[-400:]
            
        print(f"Training Custom NumPy LSTM with hidden_dim={self.hidden_dim} on {len(X)} samples...")
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            # Gradients accumulator
            dW = np.zeros_like(self.W)
            db = np.zeros_like(self.b)
            dW_out = np.zeros_like(self.W_out)
            db_out = np.zeros_like(self.b_out)
            
            for i in range(len(X)):
                x_seq = X[i] # (T, input_dim)
                y_target = y[i]
                
                # Forward pass
                y_pred, cache = self._forward_sequence(x_seq)
                loss = (y_pred - y_target) ** 2
                epoch_loss += loss
                
                # Backprop output layer
                dy = 2 * (y_pred - y_target)
                h_last = cache["h"][-1].reshape(-1, 1)
                dW_out += np.dot(h_last, np.array([[dy]]))
                db_out += dy
                
                # Backprop LSTM gates (simplified BPTT for the last few steps to speed up and stabilize)
                dh_last = np.dot(self.W_out, np.array([[dy]])).flatten()
                
                T = x_seq.shape[0]
                dh = np.zeros(self.hidden_dim)
                dc = np.zeros(self.hidden_dim)
                
                # We backpropagate the gradients back through time
                for t in reversed(range(T)):
                    dh = dh + (dh_last if t == T - 1 else 0)
                    
                    c_curr = cache["c"][t+1]
                    c_prev = cache["c"][t]
                    f = cache["f"][t].flatten()
                    i = cache["i"][t].flatten()
                    c_tilde = cache["c_tilde"][t].flatten()
                    o = cache["o"][t].flatten()
                    
                    tanh_c = np.tanh(c_curr)
                    
                    do = dh * tanh_c
                    dc_curr = dc + dh * o * (1 - tanh_c**2)
                    
                    df = dc_curr * c_prev
                    di = dc_curr * c_tilde
                    dc_tilde = dc_curr * i
                    
                    # Derivatives of activations
                    df_raw = df * f * (1 - f)
                    di_raw = di * i * (1 - i)
                    dc_tilde_raw = dc_tilde * (1 - c_tilde**2)
                    do_raw = do * o * (1 - o)
                    
                    d_gates = np.hstack((df_raw, di_raw, dc_tilde_raw, do_raw)).reshape(1, -1)
                    
                    combined = np.hstack((x_seq[t].reshape(1, -1), cache["h"][t].reshape(1, -1)))
                    dW += np.dot(combined.T, d_gates)
                    db += d_gates
                    
                    # Gradient for next step
                    dh_next = np.dot(d_gates, self.W.T)
                    dh = dh_next[0, self.input_dim:]
                    dc = dc_curr * f
            
            # Weight updates
            self.W -= lr * dW / len(X)
            self.b -= lr * db / len(X)
            self.W_out -= lr * dW_out / len(X)
            self.b_out -= lr * db_out / len(X)
            
            epoch_loss /= len(X)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.6f}")

    def predict(self, X):
        preds = []
        for x_seq in X:
            y_pred, _ = self._forward_sequence(x_seq)
            preds.append(y_pred)
        return np.array(preds)

def prepare_lstm_data(series, window_size=60):
    """
    Prepares sequence data for LSTM. Scales the data to [0, 1].
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(series.values.reshape(-1, 1))
    
    X, y = [], []
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i-window_size:i])
        y.append(scaled_data[i, 0])
        
    X, y = np.array(X), np.array(y)
    return X, y, scaler

def train_lstm(X_train, y_train, epochs=5, batch_size=32, lr=0.01, hidden_dim=16, num_layers=1):
    """
    Wrapper to match the interface and train the custom NumPy LSTM.
    """
    model = NumpyLSTM(input_dim=1, hidden_dim=hidden_dim, output_dim=1)
    model.train_model(X_train, y_train, epochs=epochs, lr=lr)
    return model, "cpu"

def forecast_lstm_test(model, X_test, scaler, device):
    """
    Forecasts for the test set.
    """
    preds = model.predict(X_test)
    preds_orig = scaler.inverse_transform(preds.reshape(-1, 1))
    return preds_orig.flatten()

def forecast_lstm_future(model, last_sequence, steps, scaler, device):
    """
    Generates multi-step future forecasts by iteratively feeding predictions back.
    """
    current_sequence = last_sequence.copy() # shape: (window_size, 1)
    future_preds = []
    
    for _ in range(steps):
        pred, _ = model._forward_sequence(current_sequence)
        future_preds.append(pred)
        
        # Roll sequence and append prediction
        current_sequence = np.roll(current_sequence, -1, axis=0)
        current_sequence[-1, 0] = pred
            
    future_preds = np.array(future_preds).reshape(-1, 1)
    future_preds_orig = scaler.inverse_transform(future_preds)
    return future_preds_orig.flatten()
