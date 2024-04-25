from percentages import *

def calculate_GHG(num):
    total_CED = 0
    total_GHG = 0
    covered_weight = 0
    total_weight = 0
    covered_ingredients = []
    name, ingredients, food_composition, ingredient_weights = setup(num)
    weights = calculate_percent(ingredients, food_composition, ingredient_weights, name)

    for ingredient in weights:
        weight = weights[ingredient]
        result = check_csv(ingredient)
        if (result):
            ghg_val, ced_val = result
            total_GHG += (float(ghg_val)/1000) * float(weight[:-1])

            total_CED += (float(ced_val)/1000) * float(weight[:-1])
            
            covered_weight += float(weight[:-1])
            covered_ingredients.append(ingredient)
        total_weight += float(weight[:-1])
    
    total_GHG = (total_GHG/total_weight) * 1000
    total_CED = (total_CED/total_weight) * 1000
    print("Total GHG: " + str(round(total_GHG, 2)) + " CO2 eq /kg ")
    print("Total CED: " + str(round(total_CED, 2)) + " MJ /kg ")
    print("Coverage: " + str(round((covered_weight/total_weight) * 100, 2)) + "%")
    print("Covered Ingredients: ")
    print(covered_ingredients)


def check_csv(query):
    file_path = 'data/datafield_simple.csv'
    if os.path.isfile(file_path):
        with open(file_path, 'r', newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                if query in row:
                    return row[2], row[3]
    return None


calculate_GHG(190796)

# 28451

# 190796

# 193925

# 19064
