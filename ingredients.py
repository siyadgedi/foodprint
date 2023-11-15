import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))


def parse_text(text):
    print(text)
    text = text.lower()
    word_tokens = word_tokenize(text)
    stopwords_in_text = [w for w in word_tokens if w.lower() in stop_words]
    text = text.replace(" and", ",")
    
    for sw in stopwords_in_text:
        text = text.replace(" " + sw, "")

    # remove parenthesis and everything within them
    text = re.sub("[\(\[].*?[\)\]]", "", text)
    
    text = re.split(r'[,.]', text)
    text_result = []

    for item in text:
        item = item.strip()
        if "contains one following" in item.lower():
            continue  # Skip items with general descriptors
        if "contains less" in item.lower():
            break  # Stop parsing when "contains less" is encountered
        item = re.sub(r'\s*[\)\]]*$', '', item)  # Remove trailing parentheses and whitespaces
        text_result.append(re.sub(r'.*: ', '', re.sub(r'.*\).*$', '', item)))

    text_result = list(filter(None, text_result))
    return text_result



prodf = pd.read_csv("Products.csv")
unique_ingredients = set()

print(parse_text(prodf.at[777, "ingredients_english"]))

# for i,rows in prodf.iterrows():
#     if type(prodf.at[i, "ingredients_english"]) != str:
#         continue
#     ingredients = parse_text(prodf.at[i, "ingredients_english"])
#     for ingredient in ingredients:
#         unique_ingredients.add(ingredient)

# print(len(list(unique_ingredients)))
# print(list(unique_ingredients)[1:101])

# NOTE: Get nutrients
# Use openfoodfacts to get missing nutrients
# (1) energy, (2) fat, (3) saturated fat, (4) carbohydrates, (5) sugar, (6) protein, and (7) sodium 