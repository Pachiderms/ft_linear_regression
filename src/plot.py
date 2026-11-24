import json
import sys

import matplotlib

import pandas as pd

DATA = sys.argv[1] if len(sys.argv) > 1 else "data.csv"
SAVE = "--save" in sys.argv


def main():
    if SAVE:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(DATA)
    with open("thetas.json") as f:
        t = json.load(f)

    xs = [df["km"].min(), df["km"].max()]
    ys = [t["theta0"] + t["theta1"] * x for x in xs]

    plt.scatter(df["km"], df["price"], label="données")
    plt.plot(xs, ys, color="red", label="régression linéaire")
    plt.xlabel("Kilométrage (km)")
    plt.ylabel("Prix")
    plt.legend()
    if SAVE:
        plt.savefig("regression.png", dpi=120)
        print("regression.png enregistré")
    else:
        plt.show()


if __name__ == "__main__":
    main()