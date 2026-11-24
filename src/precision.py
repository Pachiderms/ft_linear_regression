import json
import sys

import pandas as pd

DATA = sys.argv[1] if len(sys.argv) > 1 else "data.csv"


def main():
    df = pd.read_csv(DATA)
    with open("thetas.json") as f:
        t = json.load(f)

    y = df["price"].to_numpy(dtype=float)
    y_hat = t["theta0"] + t["theta1"] * df["km"].to_numpy(dtype=float)
    m = len(y)

    mae = sum(abs(a - b) for a, b in zip(y_hat, y)) / m
    mse = sum((a - b) ** 2 for a, b in zip(y_hat, y)) / m
    mean = sum(y) / m
    ss_tot = sum((a - mean) ** 2 for a in y)
    r2 = 1 - (mse * m) / ss_tot

    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {mse ** 0.5:.2f}")
    print(f"R2   : {r2:.4f}")


if __name__ == "__main__":
    main()