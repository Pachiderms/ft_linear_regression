import json
import sys

import numpy as np
import pandas as pd

from src.main import MyLinearRegression

DATA = sys.argv[1] if len(sys.argv) > 1 else "data.csv"
OUT = "thetas.json"


def main():
    df = pd.read_csv(DATA)
    x = df["km"].to_numpy(dtype=float).reshape(-1, 1)
    y = df["price"].to_numpy(dtype=float).reshape(-1, 1)

    mean, std = x.mean(), x.std()
    x_n = (x - mean) / std

    lr = MyLinearRegression(alpha=0.1, max_iter=2000).fit_(x_n, y)
    t0_n, t1_n = lr.thetas.flatten()

    theta1 = t1_n / std
    theta0 = t0_n - t1_n * mean / std

    with open(OUT, "w") as f:
        json.dump({"theta0": theta0, "theta1": theta1}, f)
    print(f"theta0 = {theta0:.4f}, theta1 = {theta1:.6f} -> {OUT}")

    y_hat = theta0 + theta1 * x
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    print(f"R2 = {1 - ss_res / ss_tot:.4f}")


if __name__ == "__main__":
    main()