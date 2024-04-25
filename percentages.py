import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import torch
import torch.nn.functional as F
import csv

from collections import OrderedDict
from torch.autograd import Variable
from nutritionix import *
from ingredients import get_info
import csv

big7_components = [
    'protein (g)',
    'sodium (g)',
    'carbohydrate (g)',
    'sugars (g)',
    'fat (g)',
    'calories',
    'saturated fat (g)'
]

def remap_big7(target_big7_old):
    target_big7_new = {}
    target_big7_new['protein (g)'] = target_big7_old["Protein"]
    target_big7_new['sodium (g)'] = target_big7_old["Sodium, Na"]
    target_big7_new['carbohydrate (g)'] = target_big7_old["Carbohydrate, by difference"]
    target_big7_new['sugars (g)'] = target_big7_old["Sugars, total"]
    target_big7_new['fat (g)'] = target_big7_old["Total lipid (fat)"]
    target_big7_new['calories'] = target_big7_old["Energy"]
    target_big7_new['saturated fat (g)'] = target_big7_old["Fatty acids, total saturated"]
    return target_big7_new
    
def small_ingredient(ingredient):
    small = ["emulsifier", "flavor", "acid", "enzymes", "color", "artificial", "dextrin", "vitamin"]
    for s in small:
        if s in ingredient:
            return True
    return False

def setup(num):
    bfsd_info = get_info(num)
    if (bfsd_info):
        ingredients = bfsd_info['ingredients']
        target = bfsd_info["name"]
    else:
        return None

    food_composition = {}
    for i in range(len(ingredients)):
        ingredient_tag, composition = nutritionixAPI(ingredients[i])
        # for large loops 
        # if (ingredient_tag and ingredient_tag == "usage limits exceeded"):
        #     target = ingredient_tag
        #     break
        if ingredient_tag:
            ingredients[i] = ingredient_tag
        food_composition[ingredients[i]] = composition

    food_composition[target] = remap_big7(bfsd_info['nutrients'])

    ingredient_weights = [None]*len(ingredients)

    return target, ingredients, food_composition, ingredient_weights


def setup_ingredients_only(num):
    bfsd_info = get_info(num)
    if (bfsd_info):
        ingredients = bfsd_info['ingredients']
    else:
        ingredients = []

    return ingredients

# Formulate the problem as an optimization task:
# - linear model without bias
# - one training sample per nutritional component (proteins, carbs, etc.)
# - each sample Xi from X is a row vector containing the percentage of a given nutritional component in each ingredient
# - each output Yi from Y is a scalar corresponding to the total amount of this nutrinional component in the final product
# - after training, the weights are the amount of unknown ingredients in grams
def calculate_percent(ingredients, food_composition, ingredient_weights, target):
    X = torch.zeros(len(big7_components), len(ingredients))
    W = torch.zeros(len(ingredients), 1)
    Y = torch.zeros(len(big7_components), 1)

    for i, nutritional_component in enumerate(big7_components):
        for j, ingredient in enumerate(ingredients):
            X[i,j] = float(food_composition[ingredient][nutritional_component])
        Y[i,0] = food_composition[target][nutritional_component]

    # normalization to be more efficient on smaller nutritional amounts
    Y_scaler = Y.clone()
    Y_scaler[Y_scaler == 0] = 1
    X.div_(Y_scaler)
    _ = Y.div_(Y_scaler)

    X = Variable(X, requires_grad=False)
    W = Variable(W, requires_grad=True)
    Y = Variable(Y, requires_grad=False)

    epochs = 5000
    loss_history = np.zeros((epochs))
    for epoch in range(epochs):
        Y_pred = X.mm(W)
        
        loss = (Y_pred - Y).pow(2).sum()
        
        # try to go and stay at 100g total
        loss += (W.sum()-1.).abs()*15
        
        loss_history[epoch] = loss.item()
        loss.backward()

        for i in range(W.size(0)):
            if ingredient_weights[i] is None: # update only unknown quantities
                W.data[i].sub_(1e-5 * W.grad[i].data)

                if small_ingredient(ingredients[i]): # (max 1% of "small" ingredients)
                    W.data[i].clamp_(0., 0.01)
                    
                # keep mass positives
                W.data[i].clamp_(0., 100.)
                
                # keep ingredients in order
                upper_lim = 100. if i==0 else W.data[i-1,0]
                lower_lim = 0. if (i+1)==len(W) else W.data[i+1,0]
                
                W.data[i].clamp_(lower_lim, upper_lim)
                
        W.grad.data.zero_()

    # plot loss
    # fig = plt.figure(figsize=(15,5))
    # ax = fig.add_subplot(1,1,1)
    # ax.set_title("Loss = f(epoch)")
    # ax.plot(loss_history)
    # ax.legend()
    # plt.show()

    weights = OrderedDict([(ingredients[i], "%.1fg" % (W.data[i,0]*100)) for i in range(len(ingredients))])
    write_weights_to_csv(weights)

    print("estimated composition for %s =" % target, json.dumps(weights, indent=2))
    print("total: %.1fg" % np.sum([W.data[i,0]*100 for i in range(len(ingredients))]))

    print("final loss: %.1f" % loss.item())

    return weights

def write_weights_to_csv(ordered_dict):
    csv_file_name = 'data/stored_ingredient_weights_v2.csv'
    
    # Attempt to load existing data, or create a new DataFrame if the file does not exist
    try:
        existing_df = pd.read_csv(csv_file_name)
    except FileNotFoundError:
        existing_df = pd.DataFrame(columns=['ingredient', 'total weight'])
    
    # Ensure 'total weight' is float for sum operations
    existing_df['total weight'] = existing_df['total weight'].astype(float)
    
    # Prepare new data from OrderedDict
    new_data = pd.DataFrame(list(ordered_dict.items()), columns=['ingredient', 'total weight'])
    new_data['total weight'] = new_data['total weight'].str.replace('g', '').astype(float)

    # Combine new and existing data
    combined_df = pd.concat([existing_df, new_data])
    
    # Group by ingredient and sum weights, reset index to turn groupby object back into DataFrame
    result_df = combined_df.groupby('ingredient', as_index=False)['total weight'].sum()
    
    # Write the result back to CSV without appending 'g' to weights
    result_df.to_csv(csv_file_name, index=False)

def single_product(x):
    name, ingredients, food_composition, ingredient_weights = setup(x)
    calculate_percent(ingredients, food_composition, ingredient_weights, name)

def loop(x):
# # loop through database OLD
    for i in range(x, 200000, 10000):
        try:
            name, ingredients, food_composition, ingredient_weights = setup(i)
            if name == "usage limits exceeded":
                print("starting loop at x: " + str(x))
                print("currently at i = " + str(i))
                break
            calculate_percent(ingredients, food_composition, ingredient_weights, name)
        except:
            print("skipped")


# for i in range(62, 72):
#     loop(i)


# single_product(193977)

#4255
#28451