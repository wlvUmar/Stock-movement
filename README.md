# 📈 Stock Price Movement Predictor

Predicts whether a stock price will go **up or down** using LSTM + Multi-Head Attention.  
Includes ETL pipeline, model training (full + incremental), and FastAPI API.

---

## 🔧 Features

- 🧠 LSTM + MHA (Multi-Head Attention) model
- 📊 ETL pipeline for stock CSV data
- 🔁 Incremental & recent-only training support
- 🚀 FastAPI endpoint for live prediction
- 📉 Predict next-day movement

---

## 🗂️ Project Structure


<pre><code> 
┌───app/
│   │   main.py
│   │   scheduler.py
│   ├───db/
│   │      connection.py
│   │      models.py
│   │
│   ├───routers/
│   │      router.py
│   │
│   ├───schemas/
│   │       stock.py
│   │
│   └───utils/
│           config.py
│    
└───ml/
        dataloader.py
        model.py
        training.py
</code> 
</pre>

---

## 📦 Installation

```bash
pip install -r requirements.txt
````

---

## 🧪 Usage

### 1. Train model from scratch

```python
from main import train_stock_model
train_stock_model("temp.csv", epochs=100)
```

### 2. Incremental training

```python
from main import incremental_train
incremental_train("temp.csv", "stock_model.pth", epochs=10)
```

### 3. Train on recent data only

```python
from main import train_on_new_data_only
train_on_new_data_only("temp.csv", "stock_model.pth", days_back=30)
```

### 4. Predict next day's movement

```python
from main import predict_next_day
predict_next_day("temp.csv", "stock_model.pth")
```

---

## 🌐 Run the API

```bash
uvicorn api.app:app --reload
```

---

## 📌 Notes
* Model is not trained
* Model supports 1-minute interval stock data since 2006.
* Classification task: Binary (UP or DOWN).
* Can update daily using incremental or recent data training.
* FastAPI makes deployment easy and fast.

---

## 🛠 Future Ideas

* 📈 Add visualization dashboard
* ☁️ Deploy to cloud (e.g., Render, Hugging Face Spaces)
* 💾 Add live data streaming

