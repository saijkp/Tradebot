# trade_bot_app.py
import os
import datetime
from flask import Flask, jsonify
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, EMAIndicator
import xgboost as xgb
import vectorbt as vbt

app = Flask(__name__)

# PARAMETERS
TICKER = "SPY"
START = "2016-01-01"
END = datetime.date.today().isoformat()
PROB_THRESHOLD = 0.55
MAX_POS_PCT = 0.05

ALPACA_KEY = os.environ.get("ALPACA_API_KEY")
ALPACA_SECRET = os.environ.get("ALPACA_SECRET_KEY")

# HELPER FUNCTIONS
def fetch_data():
    df = yf.download(TICKER, start=START, end=END, progress=False)
    df = df.dropna()
    df['return'] = df['Adj Close'].pct_change()
    df = df.dropna()
    return df

def add_features(df):
    df['sma20'] = SMAIndicator(df['Adj Close'], window=20).sma_indicator()
    df['sma50'] = SMAIndicator(df['Adj Close'], window=50).sma_indicator()
    df['ema12'] = EMAIndicator(df['Adj Close'], window=12).ema_indicator()
    df['rsi14'] = RSIIndicator(df['Adj Close'], window=14).rsi()
    df['vol_20'] = df['Volume'].rolling(20).mean()
    for lag in [1,2,3,5]:
        df[f'lagr_{lag}'] = df['return'].shift(lag)
    df = df.dropna()
    df['target'] = (df['return'].shift(-1) > 0).astype(int)
    df = df.dropna()
    return df

def train_model(df):
    split_idx = int(len(df) * 0.8)
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    features = ['sma20','sma50','ema12','rsi14','vol_20','lagr_1','lagr_2','lagr_3','lagr_5']
    X_train, y_train = train[features], train['target']
    X_test, y_test = test[features], test['target']

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    model.fit(X_train, y_train, early_stopping_rounds=20, eval_set=[(X_test, y_test)], verbose=False)
    return model, features, df

# ROUTES
@app.route("/")
def index():
    return "Trade Bot is running!"

@app.route("/run")
def run_backtest():
    try:
        df = fetch_data()
        df = add_features(df)
        model, features, df = train_model(df)
        probs = model.predict_proba(df[features])[:,1]
        df['pred_prob'] = probs
        df['signal'] = (df['pred_prob'] > PROB_THRESHOLD).astype(int)
        df['signal_executed'] = df['signal'].shift(1).fillna(0)

        price = df['Adj Close']
        size = df['signal_executed'] * MAX_POS_PCT

        pf = vbt.Portfolio.from_signals(
            close=price,
            entries=df['signal_executed'] == 1,
            exits=df['signal_executed'] == 0,
            init_cash=100000,
            fees=0.0005,
            slippage=0.0005,
            size=size,
            freq='1D'
        )

        stats = pf.stats().to_dict()
        latest_signal = int(df['signal_executed'].iloc[-1])
        return jsonify({"latest_signal": latest_signal, "backtest_stats": stats})
    except Exception as e:
        return jsonify({"error": str(e)})

# START APP
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
