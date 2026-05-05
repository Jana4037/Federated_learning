# Federated Book Recommender System
### Current Trends in Artificial Intelligence — VUB Course Project
**Team:** Michal Dokupil, Janarthanan Nagarajan, Nishant Kumar

---

## Project Overview

This project builds a **Federated Learning-based Book Recommender System** using the
[Book Recommendation Dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset/data)
from Kaggle.

The core idea: imagine small independent bookstores in different cities around the world.
Each store has its own local customer base and ratings data. They want to collaborate on a
shared recommendation model — without sharing their customers' private data with each other.
Federated Learning makes this possible.

---

## Setup Instructions

### 1. Install dependencies

```bash
pip install flwr torch pandas numpy scikit-learn tqdm matplotlib seaborn
```

### 2. Download the dataset

Download from Kaggle:
https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset/data

Place these three CSV files inside the `data/` folder:
- `Books.csv`
- `Users.csv`
- `Ratings.csv`

### 3. Run data preparation (exploration + partitioning)

```bash
python data_preparation.py
```

This will:
- Load and explore the raw dataset
- Partition users into "town nodes" based on their city
- Apply assortment simulation (global + local books)
- Skew rating distributions per node to simulate different store cultures
- Save all partitioned data to `data/partitions/`

### 4. Run centralized baseline

```bash
python centralized.py
```

Trains a Neural Collaborative Filtering model on all data combined.
Prints final RMSE and MAE on the test set.

### 5. Run federated simulation (easiest way — one script)

```bash
python simulation.py
```

This runs everything in-process using Flower's virtual client engine.

### 6. Run federated training with separate server + clients

Start the server first:
```bash
python server.py
```

Then in separate terminals, start clients:
```bash
python client.py --node-id city_london
python client.py --node-id city_paris
```

---

## File Structure

```
federated_book_recommender/
│
├── data/                        # Put Kaggle CSVs here
│   └── partitions/              # Auto-generated after data_preparation.py
│
├── data_preparation.py          # Data loading, exploration, partitioning
├── model.py                     # Neural Collaborative Filtering model (PyTorch)
├── utils.py                     # Shared utility functions
├── centralized.py               # Centralized training baseline
├── client.py                    # Flower federated client
├── server.py                    # Flower federated server
├── simulation.py                # Run full federation locally in one script
└── README.md
```

---

## Architecture

```
                    ┌─────────────────────┐
                    │   Central Server    │
                    │  (Flower Server)    │
                    │  FedAvg Aggregation │
                    └────────┬────────────┘
                             │  global model weights
              ┌──────────────┼──────────────┐
              │              │              │
     ┌────────▼──┐   ┌───────▼──┐   ┌──────▼────┐
     │ London    │   │  Paris   │   │  Berlin   │   ...
     │ Bookstore │   │Bookstore │   │ Bookstore │
     │ (Client)  │   │ (Client) │   │ (Client)  │
     └───────────┘   └──────────┘   └───────────┘
      local data       local data     local data
      stays here       stays here     stays here
```

---

## Research Question

> Can federated learning achieve comparable recommendation accuracy to centralized training,
> while keeping user data private within each local bookstore?

**Hypothesis:** FedAvg aggregation across heterogeneous town partitions will reach within
5-10% RMSE of the centralized baseline, without any raw user data leaving local nodes.

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| RMSE   | Root Mean Squared Error on ratings |
| MAE    | Mean Absolute Error on ratings |
| Rounds | Number of federated communication rounds |

---

## Dataset Notes

The Kaggle dataset contains:
- ~270,000 books
- ~278,000 users
- ~1,100,000 ratings (0 = implicit, 1-10 = explicit)

Users have a `Location` field (City, State, Country) — we use **City** as the federated partition key.
