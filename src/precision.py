import json
import sys

import pandas as pd

from src.loss import mae_, rmse_, r2score_

DATA = sys.argv[1] if len(sys.argv) > 1 else "data.csv"


def main():
    df = pd.read_csv(DATA)
    with open("thetas.json") as f:
        t = json.load(f)

    y = df["price"].to_numpy(dtype=float)
    y_hat = t["theta0"] + t["theta1"] * df["km"].to_numpy(dtype=float)

    print(f"MAE  : {mae_(y, y_hat):.2f}")
    print(f"RMSE : {rmse_(y, y_hat):.2f}")
    print(f"R2   : {r2score_(y, y_hat):.4f}")


if __name__ == "__main__":
    main()