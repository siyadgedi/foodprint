import numpy as np
from scipy.optimize import linprog


## FYI: This is not working yet 

# sample data
ingredients = ["sugar", "palm oil", "hazelnuts", "skim milk powder", "cocoa"]
big7key = ["energy in kcal", "fat", "saturated fat", "carbohydrates", "sugar", "protein", "salt"]

# Nutritional values for each ingredient
sugar_big7 = [387, 0, 0, 100, 100, 0, 0]
palm_oil_big7 = [857, 100, 47.9, 0, 0, 0, 0]
hazelnuts_big7 = [653.3, 64.3, 4.6, 17.8, 5, 13.5, 0]
skim_milk_powder_big7 = [357, 0.7, 0.5, 51.5, 51.5, 35.3, 0.54]
cocoa_big7 = [420, 10, 0, 60, 0, 20, 0]

# nutritional values for main product
big7 = [541, 29.7, 10.8, 59.5, 56.8, 5.41, 0.041]

data = [sugar_big7, palm_oil_big7, hazelnuts_big7, skim_milk_powder_big7, cocoa_big7]

c = np.zeros(len(data))

A_ub = np.array(data).T
b_ub = np.array(big7)

# Equality constraint to ensure the sum of percentages is 1
A_eq = np.ones((1, len(data)))
b_eq = 1.0

# Bounds for linear programming
bounds = [(0, 1) for _ in range(len(data))] 

result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

ingredient_composition = result

print("\nIngredient Composition:")
print(ingredient_composition)