
import json
import requests

def nutritionixAPI(query):
    url = "https://trackapi.nutritionix.com/v2/natural/nutrients"

    payload = json.dumps({
        "query": "100g " + query
    })

    headers = {
        "x-app-id": "b3444f3c",
        "x-app-key": "8dee6a36d45fb133556413d7e9bd65ab",
        "Content-Type": "application/json"
    }

    response = requests.request("POST", url, headers=headers, data=payload)
     
    data = response.json()
    response_dict = {}
    # TODO ensure these values are out of 100g

    if (data.get("message")):
        print(data.get("message"))

    response_dict["protein (g)"] = data.get("foods", [{}])[0].get("nf_protein") or 0
    response_dict["sodium (g)"] = (data.get("foods", [{}])[0].get("nf_sodium") or 0)/1000
    response_dict["carbohydrate (g)"] = data.get("foods", [{}])[0].get("nf_total_carbohydrate") or 0
    response_dict["sugars (g)"] = data.get("foods", [{}])[0].get("nf_sugars") or 0
    response_dict["fat (g)"] = data.get("foods", [{}])[0].get("nf_total_fat") or 0
    response_dict["calories"] = data.get("foods", [{}])[0].get("nf_calories") or 0
    response_dict["saturated fat (g)"] = data.get("foods", [{}])[0].get("nf_saturated_fat") or 0
    return response_dict
