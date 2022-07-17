from pyrebase import pyrebase
import json

import config


########################################################################################################################
# server authentication
########################################################################################################################
serverCreds = {
  "apiKey":"AIzaSyBp6wFgPJKiRWQA_HP1fLWI4g2omFy7sM8",
  "authDomain": "projectregalhospitality-default-rtdb.firebaseapp.com",
  "databaseURL": "https://projectregalhospitality-default-rtdb.firebaseio.com",
  "storageBucket": "projectregalhospitality.appspot.com"
}
# authenticate_user()
serverBase = pyrebase.initialize_app(serverCreds)
stdb = serverBase.database()


########################################################################################################################
# set initial menu structure
########################################################################################################################
def set_menu_structure():
    product_list: list = [
        'Meals',
        'Snacks',
        'Electronics',
        'Toiletries'
    ]
    meals_list: list = [
        'Cuisine',
        'Genre',
        'Type'
    ]
    cuisine_type_list: list = [
        'American',
        'Italian',
        'Mexican',
        'Chinese',
        'Japanese',
        'Indian',
        'Korean',
        'Vietnamese',
        'Thai',
        'Mediterranean',
        'Asian',
        'Greek',
        'Latin American',
        'Southern',
        'French'
    ]
    pop_type_list: list = [
        'Breakfast',
        'Fast Food',
        'Healthy'
        'Bakery',
        'Vegan',
        'Deli',
        'BBQ',
        'Gluten Free',
        'Pub',
        'Appetizers',
        'Vegetarian'
    ]
    foods_type_list: list = [
        'Seafood',
        'Wings',
        'Chicken',
        'Burgers',
        'Steak',
        'Bowls',
        'Sandwiches',
        'Salad',
        'Pasta',
        'Desserts',
        'Pizza',
        'Soup',
        'Noodles',
        'Ice Cream',
        'Frozen Yogurt',
        'Sushi',
        'Burritos',
        'Tacos',
        'Donuts',
        'Pretzels',
        'Cup Cakes'
    ]
    snacks_list: list = [
        'Bakery',
        'Fruit',
        'Nuts',
        'Candy',
        'Donuts',
        'Muffins',
        'Cookies',
        'Chips & Crackers',
        'Healthy',
        'Nutrition Bars',
        'Popcorn',
        'Chocolate',
        'Chocolate',
        'Ice Cream',
        'Coffees',
        'Juices',
        'Teas',
        'Smoothies',
        'Energy Drinks',
        'Sports Drinks',
        'Bottled Water',
        'Sodas'
    ]
    elx_list: list = [
        'Chargers',
        'Headphones',
        'Power Adapters',
        'Earbuds',
        'Speakers',
        'Batteries',
        'Phone Cases',
        'Other'
    ]
    toi_list: list = [
        'Toiletry Bags',
        'Moisturizers',
        'Shaving Accessories',
        'Sunscreen',
        'Facial Cleanser',
        'Toothbrushes & Toothpaste',
        'Oral Hygiene',
        'Shampoo & Conditioner',
        'Makeup',
        'Hand Sanitizer',
        'Chapstick',
        'Sewing Kit',
        'Skin Care',
        'First Aid Kits',
        'Perfumes & Colognes',
        'Hand & Feet Care',
        'Deodorants'
    ]
    stdb.child('configuration').child('menus').child('productList').set(product_list)
    stdb.child('configuration').child('menus').child('mealTypeList').set(meals_list)
    stdb.child('configuration').child('menus').child('cuisineTypeList').set(cuisine_type_list)
    stdb.child('configuration').child('menus').child('popularTypeList').set(pop_type_list)
    stdb.child('configuration').child('menus').child('foodsTypeList').set(foods_type_list)
    stdb.child('configuration').child('menus').child('snacksTypeList').set(snacks_list)
    stdb.child('configuration').child('menus').child('electronicsTypeList').set(elx_list)
    stdb.child('configuration').child('menus').child('toiletriesTypeList').set(toi_list)


def get_menu_structure():
    all_data = stdb.child('configuration').get()
    for d in all_data.each():
        data = d.val()
        json_str = json.dumps(data)
        resp = json.loads(json_str)
        sorted_obj = dict(resp)

        # create dict
        if "productList" in sorted_obj:
            config.product_list = sorted_obj['productList']
        if "mealTypeList" in sorted_obj:
            config.meal_type_list = sorted_obj['mealTypeList']
        if "cuisineTypeList" in sorted_obj:
            config.cuisine_type_list = sorted_obj['cuisineTypeList']
        if "popularTypeList" in sorted_obj:
            config.pop_type_list = sorted_obj['popularTypeList']
        if "foodsTypeList" in sorted_obj:
            config.foods_type_list = sorted_obj['foodsTypeList']
        if "snacksTypeList" in sorted_obj:
            config.snacks_type_list = sorted_obj['snacksTypeList']
        if "electronicsTypeList" in sorted_obj:
            config.elx_type_list = sorted_obj['electronicsTypeList']
        if "toiletriesTypeList" in sorted_obj:
            config.toi_type_list = sorted_obj['toiletriesTypeList']
        # if "pA" in sorted_obj:
        #     ct.currents0 = sorted_obj["pA"]


def add_menu_item(item: str, item_list_name: str):
    if item_list_name == "product":
        config.product_list.append(item)
        stdb.child('configuration').child('menus').child('productList').set(config.product_list)

