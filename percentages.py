import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import torch
import torch.nn.functional as F

from collections import OrderedDict
from torch.autograd import Variable
from nutritionix import *
#from ingredients import get_info

big7_components = [
    'protein (g)',
    'sodium (g)',
    'carbohydrate (g)',
    'sugars (g)',
    'fat (g)',
    'calories',
    'saturated fat (g)'
]


def setup(target, num):
    ingredients = ["sugar", "palm_oil", "hazelnut", "low_fat_cocoa", "nonfat_milk_powder"]
    # ingredients = ['sugar', 'palm oil', 'hazelnuts', 'cocoa', 'skim milk', 'whey', 'lecithin emulsifier', 'artificial flavor']

    # TODO
    # bfsd_info = get_info(num)
    # ingredients = bfsd_info['ingredients']

    food_composition = {}
    for ingredient in ingredients:
        food_composition[ingredient] = nutritionixAPI(ingredient)

    # TODO: use bfsd_info['nutrients'] instead of manual entry - need to fix mapping
    food_composition[target] = {'protein (g)': 5.4, 'sodium (g)': 0.04, 'carbohydrate (g)': 62.1, 'sugars (g)': 54, 'fat (g)': 29.7, 'calories': 540, 'saturated fat (g)': 29.7}
    ingredient_weights = [None]*len(ingredients)

    return ingredients, food_composition, ingredient_weights

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
    X.div_(Y_scaler)
    _ = Y.div_(Y_scaler)

    X = Variable(X, requires_grad=False)
    W = Variable(W, requires_grad=True)
    Y = Variable(Y, requires_grad=False)

    epochs = 5000
    loss_history = np.zeros((epochs))
    for epoch in range(epochs):
        Y_pred = X.mm(W)
        
        # l2 loss
        loss = (Y_pred - Y).pow(2).sum()
        
        # try to go and stay at 100g total
        loss += (W.sum()-1.).abs()*5
        
        loss_history[epoch] = loss.item()
        loss.backward()

        for i in range(W.size(0)):
            if ingredient_weights[i] is None: # update only unknown quantities
                W.data[i].sub_(1e-5 * W.grad[i].data)
                
                # keep mass positives
                W.data[i].clamp_(0., 100.)
                
                # keep ingredients in order
                upper_lim = 100. if i==0 else W.data[i-1,0]
                lower_lim = 0. if (i+1)==len(W) else W.data[i+1,0]
                
                W.data[i].clamp_(lower_lim, upper_lim)
                
        W.grad.data.zero_()

    weights = OrderedDict([(ingredients[i], "%.1fg" % (W.data[i,0]*100)) for i in range(len(ingredients))])
    print("estimated composition for %s =" % target, json.dumps(weights, indent=2))
    print("total: %.1fg" % np.sum([W.data[i,0]*100 for i in range(len(ingredients))]))

    print("final loss: %.1f" % loss.item())

# something isn't working during setup or calculate_percent
# when the sample data from below is used, calculate_percent works
# but when setup is used, calculate_percent is not correct
# not sure if the error is in setup or something is wrong in calculate_percent
ingredients, food_composition, ingredient_weights = setup("nutella", 28451)

# SAMPLE data
# food_composition2 = {
#     "sugar": {'protein (g)': 0, 'sodium (g)': 0, 'carbohydrate (g)': 100, 'sugars (g)': 100, 'fat (g)': 0, 'calories': 387, 'saturated fat (g)': 0},
#     "palm_oil": {'protein (g)': 0, 'sodium (g)': 0, 'carbohydrate (g)': 0, 'sugars (g)': 0, 'fat (g)': 100, 'calories': 865.8, 'saturated fat (g)': 47.8},
#     "hazelnut": {'protein (g)': 15.4, 'sodium (g)': 0, 'carbohydrate (g)': 17.8, 'sugars (g)': 5, 'fat (g)': 64.3, 'calories': 653.3, 'saturated fat (g)': 4.6},
#     "low_fat_cocoa": {'protein (g)': 20, 'sodium (g)': 0, 'carbohydrate (g)': 60, 'sugars (g)': 0, 'fat (g)': 10, 'calories': 420, 'saturated fat (g)': 0},
#     "nonfat_milk_powder": {'protein (g)': 36.9, 'sodium (g)': 0.57, 'carbohydrate (g)': 57.9, 'sugars (g)': 44.8, 'fat (g)': 47.4, 'calories': 842.7, 'saturated fat (g)': 21.1},
#     "nutella": {'protein (g)': 5.4, 'sodium (g)': 0.04, 'carbohydrate (g)': 62.1, 'sugars (g)': 54, 'fat (g)': 29.7, 'calories': 540, 'saturated fat (g)': 29.7}
# }


# ingredients = ["sugar", "palm_oil", "hazelnut", "low_fat_cocoa", "nonfat_milk_powder"]
# ingredient_weights = [None]*len(ingredients)
# target = "nutella"


calculate_percent(ingredients, food_composition, ingredient_weights, "nutella")