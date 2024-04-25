# NOTE: Get nutrients
# Use openfoodfacts to get missing nutrients
# (1) energy, (2) fat, (3) saturated fat, (4) carbohydrates, (5) sugar, (6) protein, and (7) sodium 

import pandas as pd
import requests

def get_nutrients(product_no, upc):
    ndf = pd.read_csv("data/Nutrients.csv")
    prodf = ndf.loc[ndf["NDB_No"] == product_no]
    big_7 = ["Energy", "Total lipid (fat)", "Fatty acids, total saturated", "Carbohydrate, by difference", "Sugars, total", "Protein", "Sodium, Na"]
    values = {}
    for index, row in prodf.iterrows():
        nutrient = row["Nutrient_name"]
        if nutrient in big_7:
            if nutrient == "Sodium, Na":
                values[nutrient] = row["Output_value"]/1000
            else: values[nutrient] = row["Output_value"]
            big_7.remove(nutrient)
    if len(big_7) > 0:
        for nutrient in big_7:
            result = fetch_nutrient(nutrient=nutrient, upc=upc)
            if result != "N/A":
                values[nutrient] = fetch_nutrient(nutrient=nutrient, upc=upc)
                big_7.remove(nutrient)


    if len(big_7) > 0:
        print("Failed")
    
    return values


def fetch_nutrient(nutrient, upc):
    mapping = {"Energy": "energy-kcal_100g",
               "Total lipid (fat)": "fat_100g",
               "Fatty acids, total saturated": "saturated-fat_100g",
               "Carbohydrate, by difference": "carbohydrates_100g",
               "Sugars, total": "sugars_100g",
               "Protein": "proteins_100g",
               "Sodium, Na": "sodium_100g"}
    
    url = f"https://world.openfoodfacts.org/api/v2/product/{upc}.json"
    resp = requests.get(url)
    result = "N/A"
    if resp.ok:
        data = resp.json()
        if mapping[nutrient] not in data["product"]["nutriments"]:
            result = 0
        else:
            result = data["product"]["nutriments"][mapping[nutrient]]
        # if nutrient == "Sodium, Na":
        #     result *= 1000
    return result


# print(get_nutrients(45127487, "00030100585541"))
