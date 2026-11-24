import json
import sys

import matplotlib

import pandas as pd

DATA = next((a for a in sys.argv[1:] if not a.startswith("--")), "data.csv")
SAVE = "--save" in sys.argv


def main():
    if SAVE:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(DATA)
    try:
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
    except:
        print("No thetas found")
        return


if __name__ == "__main__":
    main()