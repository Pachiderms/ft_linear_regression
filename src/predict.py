import json


def estimate_price(mileage, theta0, theta1):
    return theta0 + theta1 * mileage


def main():
    try:
        with open("thetas.json") as f:
            t = json.load(f)
        theta0, theta1 = t["theta0"], t["theta1"]
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        theta0, theta1 = 0.0, 0.0

    try:
        mileage = float(input("Kilométrage : "))
    except ValueError:
        print("Veuillez entrer un nombre.")
        return
    if mileage < 0:
        print("Le kilométrage doit être positif.")
        return
    print(f"Prix estimé : {estimate_price(mileage, theta0, theta1):.2f}")


if __name__ == "__main__":
    main()