# -*- coding: utf-8 -*-

import streamlit as st
from copy import deepcopy

import server
import config

########################################################################################################################
# download data configs
########################################################################################################################

server.get_menu_structure()

########################################################################################################################
# build interface - capitalized variables indicate parameter inputs for the GA.
########################################################################################################################

title_label = st.header("ConciAIRge Database Builder")
sep0 = st.markdown('___')
CONFIG_TYPE = st.radio(
    "Select which action to perform:",
    ["View Data Option", "Edit Data Options"],
    key='config_type',
)

sep1 = st.markdown('___')
DEBUG_CONFIG: bool = False

########################################################################################################################
# interface logic
########################################################################################################################
def reset_item_add_interface():
    global GEN_TYPE
    global gen_type_button
    global gen_type_caption
    global GEN_MEALS_TYPE
    global gen_meals_caption
    global CUISINE_TYPE
    global cuisine_type_caption
    global cuisine_type_button
    global POP_TYPE
    global pop_type_caption
    global pop_type_button
    global FOODS_TYPE
    global foods_type_caption
    global foods_type_button
    global SNACKS_TYPE
    global snacks_type_caption
    global ELX_TYPE
    global elx_type_caption
    global TOI_TYPE
    global toi_type_caption

    # GEN_TYPE = st.radio(
    #     "Select the type of product:",
    #     (
    #         'Meals',
    #         'Snacks',
    #         'Electronics',
    #         'Toiletries'
    #     )
    # )
    # gen_type_button = st.button('What are these?')
    # gen_type_caption = st.empty()

    FOODS_TYPE = st.empty()
    foods_type_button = st.empty()
    foods_type_caption = st.empty()

    CUISINE_TYPE = st.empty()
    cuisine_type_button = st.empty()
    cuisine_type_caption = st.empty()

    POP_TYPE = st.empty()
    pop_type_button = st.empty()
    pop_type_caption = st.empty()

    CUISINE_TYPE = st.empty()
    # cuisine_type_button = st.button('What are these?', key='cuisine_type_button')
    cuisine_type_caption = st.empty()

    POP_TYPE = st.empty()
    # pop_type_button = st.button('What are these?', key='pop_type_button')
    pop_type_caption = st.empty()

    FOODS_TYPE = st.empty()
    # foods_type_button = st.button('What are these?', key='foods_type_button')
    foods_type_caption = st.empty()

    SNACKS_TYPE = st.empty()
    # snacks_type_button = st.button('What are these?', key='snacks_type_button')
    snacks_type_caption = st.empty()

    ELX_TYPE = st.empty()
    # elx_type_button = st.button('What are these?', key='elx_type_button')
    elx_type_caption = st.empty()

    TOI_TYPE = st.empty()
    # toi_type_button = st.button('What are these?', key='toi_type_button')
    toi_type_caption = st.empty()



if CONFIG_TYPE == "View Data Option":
    desc_label = st.subheader("Select the category of menu item you would like to view:")
    GEN_TYPE = st.radio(
        "Select the type of product:",
        config.product_list
        , key='gen_type'
    )
    gen_type_button = st.empty()
    gen_type_caption = st.empty()

    if GEN_TYPE == "Meals":

        GEN_MEALS_TYPE = st.radio(
            "Select the meals category:",
            config.meal_type_list,
            key='meals_type'
        )
        # gen_meals_type = st.button('What are these?', key='gen_meals_type_button')
        gen_meals_caption = st.empty()

        if GEN_MEALS_TYPE == "Cuisine":

            CUISINE_TYPE = st.radio(
                "Select the cuisine:",
                config.cuisine_type_list,
                key='cuisine_type'
            )
            # cuisine_type_button = st.button('What are these?', key='cuisine_type_button')
            cuisine_type_caption = st.empty()

            POP_TYPE = st.empty()
            # pop_type_button = st.button('What are these?', key='pop_type_button')
            pop_type_caption = st.empty()

            FOODS_TYPE = st.empty()
            # foods_type_button = st.button('What are these?', key='foods_type_button')
            foods_type_caption = st.empty()

        elif GEN_MEALS_TYPE == "Genre":
            POP_TYPE = st.radio(
                "Select the genre of food:",
                config.pop_type_list,
                key='popular_type'
            )
            # pop_type_button = st.button('What are these?', key='pop_type_button')
            pop_type_caption = st.empty()


            CUISINE_TYPE = st.empty()
            # cuisine_type_button = st.button('What are these?', key='cuisine_type_button')
            cuisine_type_caption = st.empty()

            FOODS_TYPE = st.empty()
            # foods_type_button = st.button('What are these?', key='foods_type_button')
            foods_type_caption = st.empty()

        elif GEN_MEALS_TYPE == "Type":

            FOODS_TYPE = st.radio(
                "Select the category of food:",
                config.foods_type_list,
                key='foods_type'
            )
            # foods_type_button = st.button('What are these?', key='foods_type_button')
            foods_type_caption = st.empty()

            CUISINE_TYPE = st.empty()
            # cuisine_type_button = st.button('What are these?', key='cuisine_type_button')
            cuisine_type_caption = st.empty()

            POP_TYPE = st.empty()
            # pop_type_button = st.button('What are these?', key='pop_type_button')
            pop_type_caption = st.empty()

    elif GEN_TYPE == "Snacks":
        SNACKS_TYPE = st.radio(
            "Select the category of snacks:",
            config.snacks_type_list,
            key='snacks_type'
        )
        snacks_type_button = st.button('What are these?', key='snacks_type_button')
        snacks_type_caption = st.empty()

        ELX_TYPE = st.empty()
        elx_type_button = st.button('What are these?', key='elx_type_button')
        elx_type_caption = st.empty()

        TOI_TYPE = st.empty()
        toi_type_button = st.button('What are these?', key='toi_type_button')
        toi_type_caption = st.empty()

    elif GEN_TYPE == "Electronics":

        ELX_TYPE = st.radio(
            "Select the type of electronics:",
            config.elx_type_list,
            key='elx_type'
        )
        # elx_type_button = st.button('What are these?', key='elx_type_button')
        elx_type_caption = st.empty()

        SNACKS_TYPE = st.empty()
        # snacks_type_button = st.button('What are these?', key='snacks_type_button')
        snacks_type_caption = st.empty()

        TOI_TYPE = st.empty()
        # toi_type_button = st.button('What are these?', key='toi_type_button')
        toi_type_caption = st.empty()

    elif GEN_TYPE == "Toiletries":

        TOI_TYPE = st.radio(
            "Select the toiletries type:",
            config.toi_type_list,
            key='toi_type'
        )
        # toi_type_button = st.button('What are these?', key='toi_type_button')
        toi_type_caption = st.empty()

        SNACKS_TYPE = st.empty()
        # snacks_type_button = st.button('What are these?', key='snacks_type_button')
        snacks_type_caption = st.empty()

        ELX_TYPE = st.empty()
        # elx_type_button = st.button('What are these?', key='elx_type_button')
        elx_type_caption = st.empty()

    else:
        CUISINE_TYPE = st.empty()
        # cuisine_type_button = st.button('What are these?', key='cuisine_type_button')
        cuisine_type_caption = st.empty()

        POP_TYPE = st.empty()
        # pop_type_button = st.button('What are these?', key='pop_type_button')
        pop_type_caption = st.empty()

        FOODS_TYPE = st.empty()
        # foods_type_button = st.button('What are these?', key='foods_type_button')
        foods_type_caption = st.empty()

        SNACKS_TYPE = st.empty()
        # snacks_type_button = st.button('What are these?', key='snacks_type_button')
        snacks_type_caption = st.empty()

        ELX_TYPE = st.empty()
        # elx_type_button = st.button('What are these?', key='elx_type_button')
        elx_type_caption = st.empty()

        TOI_TYPE = st.empty()
        # toi_type_button = st.button('What are these?', key='toi_type_button')
        toi_type_caption = st.empty()

    add_button = st.button('Add Item')
    success_label = st.empty()

    if add_button:
        # warning_label.empty()
        reset_item_add_interface()
        success_label.success('Adding menu item to database')

elif CONFIG_TYPE == "Edit Data Options":
    reset_item_add_interface()

    desc_label = st.subheader("Select the category of menu item you would like to add:")
    GEN_TYPE = st.radio(
        "Select the type of product:",
        config.product_list,
        key='gen_type'
    )
    gen_type_button = st.button('+', key='gen_type_button')
    gen_type_caption = st.empty()
    if gen_type_button:
        gen_type_text = st.text_input(label="New product type",
                                      value="",
                                      key="gen_type_text")
        gen_type_text_button = st.button('Done', key="gen_type_text_button")
        if gen_type_text_button:
            server.add_menu_item(gen_type_text, "product")
            gen_type_text = st.empty()
            gen_type_text_button = st.empty()
            gen_type_caption.success("Success")

    if GEN_TYPE == "Meals":

        GEN_MEALS_TYPE = st.radio(
            "Select the meals category:",
            config.meal_type_list,
            key='meals_type'
        )
        gen_meals_button = st.button('+', key='gen_meals_type_button')
        gen_meals_caption = st.empty()
        if gen_meals_button:
            gen_meals_text = st.text_input(label="New meals category",
                                           value="",
                                           key="gen_meals_text")
            gen_meals_text_button = st.button('Done', key="gen_meals_text_button")
            if gen_meals_text_button:
                server.add_menu_item(gen_meals_text, "meal_type")
                gen_meals_text = st.empty()
                gen_meals_text_button = st.empty()
                gen_meals_caption.success("Success")

        if GEN_MEALS_TYPE == "Cuisine":

            CUISINE_TYPE = st.radio(
                "Select the cuisine:",
                config.cuisine_type_list,
                key='cuisine_type'
            )
            cuisine_type_button = st.button('+', key='cuisine_type_button')
            cuisine_type_caption = st.empty()
            if cuisine_type_button:
                cuisine_type_text = st.text_input(label="New cuisine",
                                                  value="",
                                                  key="cuisine_type_text")
                cuisine_type_text_button = st.button('Done', key="cuisine_type_text_button")
                if cuisine_type_text_button:
                    server.add_menu_item(cuisine_type_text, "cuisine_type")
                    cuisine_type_text = st.empty()
                    cuisine_type_text_button = st.empty()
                    cuisine_type_caption.success("Success")

            POP_TYPE = st.empty()
            pop_type_button = st.empty()
            pop_type_caption = st.empty()

            FOODS_TYPE = st.empty()
            foods_type_button = st.empty()
            foods_type_caption = st.empty()

        elif GEN_MEALS_TYPE == "Genre":
            POP_TYPE = st.radio(
                "Select the genre of food:",
                config.pop_type_list,
                key='popular_type'
            )
            pop_type_button = st.button('+', key='pop_type_button')
            pop_type_caption = st.empty()
            if pop_type_button:
                pop_type_text = st.text_input(label="New food genre",
                                              value="",
                                              key="pop_type_text")
                pop_type_text_button = st.button('Done', key="pop_type_text_button")
                if pop_type_text_button:
                    server.add_menu_item(pop_type_text, "pop_type")
                    pop_type_text = st.empty()
                    pop_type_text_button = st.empty()
                    pop_type_caption.success("Success")

            CUISINE_TYPE = st.empty()
            cuisine_type_button = st.empty()
            cuisine_type_caption = st.empty()

            FOODS_TYPE = st.empty()
            foods_type_button = st.empty()
            foods_type_caption = st.empty()

        elif GEN_MEALS_TYPE == "Type":

            FOODS_TYPE = st.radio(
                "Select the category of food:",
                config.foods_type_list,
                key='foods_type'
            )
            foods_type_button = st.button('+', key='foods_type_button')
            foods_type_caption = st.empty()
            if foods_type_button:
                foods_type_text = st.text_input(label="New food category",
                                                value="",
                                                key="foods_type_text")
                foods_type_text_button = st.button('Done', key="foods_type_text_button")
                if foods_type_text_button:
                    server.add_menu_item(foods_type_text, "foods_type")
                    foods_type_text = st.empty()
                    foods_type_text_button = st.empty()
                    foods_type_caption.success("Success")

            CUISINE_TYPE = st.empty()
            cuisine_type_button = st.empty()
            cuisine_type_caption = st.empty()

            POP_TYPE = st.empty()
            pop_type_button = st.empty()
            pop_type_caption = st.empty()

    elif GEN_TYPE == "Snacks":
        SNACKS_TYPE = st.radio(
            "Select the category of snacks:",
            config.snacks_type_list,
            key='snacks_type'
        )
        snacks_type_button = st.button('+', key='snacks_type_button')
        snacks_type_caption = st.empty()

        if snacks_type_button:
            snacks_type_text = st.text_input(label="New snacks type",
                                             value="",
                                             key="snacks_type_text")
            snacks_type_text_button = st.button('Done', key="snacks_type_text_button")
            if snacks_type_text_button:
                server.add_menu_item(snacks_type_text, "snacks_type")
                snacks_type_text = st.empty()
                snacks_type_text_button = st.empty()
                snacks_type_caption.success("Success")

        ELX_TYPE = st.empty()
        elx_type_button = st.empty()
        elx_type_caption = st.empty()

        TOI_TYPE = st.empty()
        toi_type_button = st.empty()
        toi_type_caption = st.empty()

    elif GEN_TYPE == "Electronics":

        ELX_TYPE = st.radio(
            "Select the type of electronics:",
            config.elx_type_list,
            key='elx_type'
        )
        elx_type_button = st.button('+', key='elx_type_button')
        elx_type_caption = st.empty()

        if elx_type_button:
            elx_type_text = st.text_input(label="New electronics category",
                                          value="",
                                          key="elx_type_text")
            elx_type_text_button = st.button('Done', key="elx_type_text_button")
            if elx_type_text_button:
                server.add_menu_item(elx_type_text, "elx_type")
                elx_type_text = st.empty()
                elx_type_text_button = st.empty()
                elx_type_caption.success("Success")

        SNACKS_TYPE = st.empty()
        snacks_type_button = st.empty()
        snacks_type_caption = st.empty()

        TOI_TYPE = st.empty()
        toi_type_button = st.empty()
        toi_type_caption = st.empty()

    elif GEN_TYPE == "Toiletries":

        TOI_TYPE = st.radio(
            "Select the toiletries type:",
            config.toi_type_list,
            key='toi_type'
        )
        toi_type_button = st.button('+', key='toi_type_button')
        toi_type_caption = st.empty()

        if toi_type_button:
            toi_type_text = st.text_input(label="New toiletries category",
                                          value="",
                                          key="toi_type_text")
            toi_type_text_button = st.button('Done', key="toi_type_text_button")
            if toi_type_text_button:
                server.add_menu_item(toi_type_text, "toi_type")
                toi_type_text = st.empty()
                toi_type_text_button = st.empty()
                toi_type_caption.success("Success")

        SNACKS_TYPE = st.empty()
        snacks_type_button = st.empty()
        snacks_type_caption = st.empty()

        ELX_TYPE = st.empty()
        elx_type_button = st.empty()
        elx_type_caption = st.empty()

    else:
        CUISINE_TYPE = st.empty()
        cuisine_type_button = st.empty()
        cuisine_type_caption = st.empty()

        POP_TYPE = st.empty()
        pop_type_button = st.empty()
        pop_type_caption = st.empty()

        FOODS_TYPE = st.empty()
        foods_type_button = st.empty()
        foods_type_caption = st.empty()

        SNACKS_TYPE = st.empty()
        snacks_type_button = st.empty()
        snacks_type_caption = st.empty()

        ELX_TYPE = st.empty()
        elx_type_button = st.empty()
        elx_type_caption = st.empty()

        TOI_TYPE = st.empty()
        toi_type_button = st.empty()
        toi_type_caption = st.empty()

        success_label = st.empty()

    success_label = st.empty()

