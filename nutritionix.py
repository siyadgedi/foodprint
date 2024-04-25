
import json
import requests
import csv
import os

big7 = ["protein (g)", "sodium (g)", "carbohydrate (g)", "sugars (g)", "fat (g)", "calories", "saturated fat (g)"]
def nutritionixAPI(query):
    #check if the query has been called before and is the stored csv file
    stored = check_csv(query)
    if stored:
        stored_tag, stored_data = stored
        response_dict = {}
        for i in range(7):
            response_dict[big7[i]] = stored_data[i]
        return stored_tag, response_dict

    url = "https://trackapi.nutritionix.com/v2/natural/nutrients"

    payload = json.dumps({
        "query": "100g " + query
    })

    f = open('api_key.json')
    headers = json.load(f)

    response = requests.request("POST", url, headers=headers, data=payload)
     
    data = response.json()
    response_dict = {}
    # TODO ensure these values are out of 100g

    if (data.get("message")):
        print(data.get("message"))
    
    response_dict[big7[0]] = data.get("foods", [{}])[0].get("nf_protein") or 0
    response_dict[big7[1]] = (data.get("foods", [{}])[0].get("nf_sodium") or 0)/1000
    response_dict[big7[2]] = data.get("foods", [{}])[0].get("nf_total_carbohydrate") or 0
    response_dict[big7[3]] = data.get("foods", [{}])[0].get("nf_sugars") or 0
    response_dict[big7[4]] = data.get("foods", [{}])[0].get("nf_total_fat") or 0
    response_dict[big7[5]] = data.get("foods", [{}])[0].get("nf_calories") or 0
    response_dict[big7[6]] = data.get("foods", [{}])[0].get("nf_saturated_fat") or 0

    # add the ingredient to the stored csv file
    item_tag = data.get("foods", [{}])[0].get("tags", {}).get("item")
    stored_list = [query, item_tag]

    for key, value in response_dict.items():
        stored_list.append(value)
    write_to_csv(stored_list)
    
    if (data.get("message") and data.get("message") == "usage limits exceeded" ):
        item_tag = data.get("message")
        
    return item_tag, response_dict

def check_csv(query):
    file_path = 'data/stored_ingredients.csv'
    if os.path.isfile(file_path):
        with open(file_path, 'r', newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                if query in row:
                    return row[1], row[2:]
    return None

def write_to_csv(data):
    file_path = 'data/stored_ingredients.csv'
    with open(file_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(data)

