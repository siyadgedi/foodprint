import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nutrients import get_nutrients
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))


def parse_text(text):
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


def get_info(index):
    prodf = pd.read_csv("Products.csv")
    unique_ingredients = set()
    if type(prodf.at[index, "ingredients_english"]) != str:
        return
    ingredients = parse_text(prodf.at[index, "ingredients_english"])
    full_data = {"ingredients": ingredients}
    for ingredient in ingredients:
        unique_ingredients.add(ingredient)

    full_data["name"] = prodf.at[index, "long_name"]
    full_data["nutrients"] = get_nutrients(prodf.at[index, "NDB_Number"], prodf.at[index, "gtin_upc"])

    return full_data

# print(get_info(3678))




