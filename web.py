import streamlit as sl
import numpy as np
import pickle as pkl
import random
sl.header("Recommendation system using machine learning")

model = pkl.load(open('/recSysMaterials/model.pkl', 'rb'))
df_pivot_ids = pkl.load(open('recSysMaterials/df_pivot_ids.pkl', 'rb'))
df_dup = pkl.load(open('recSysMaterials/df_dup.pkl', 'rb'))
df_pivot_table = pkl.load(open('recSysMaterials/df_pivot_table.pkl', 'rb'))
df_data = pkl.load(open('recSysMaterials/df_data.pkl', 'rb'))


selected_name = sl.selectbox("Choose a product and get other recommended products!", df_dup['product_name'].drop_duplicates())
selected_id = df_dup[df_dup['product_name']==selected_name]['product_id'].iloc[0]


def recommend_product(product_id):
    product_index = df_pivot_ids[df_pivot_ids['product_id']==product_id].index[0]
    distance, suggest = model.kneighbors(df_pivot_table.iloc[product_index,:].values.reshape(1,-1), n_neighbors=round(df_pivot_table.index.size*0.01))
    suggest = np.delete(suggest, 0)
    return suggest
    
    
def random_recommend_product(suggest):
    randomX = round(df_pivot_table.index.size*0.01*0.1)
    random_items = random.sample(suggest.tolist(), randomX)
    
    random_items = np.array(random_items)
    
    suggest_id = []
    for i in range(len(random_items)):
        suggest_id.append(df_pivot_table.index[random_items[i]])
    
    recommended = []
    for i in range(len(suggest_id)):
        recommended.append(df_data[df_data['product_id']==suggest_id[i]])
    return recommended

suggestions = recommend_product(selected_id)
print(suggestions)
if sl.button("Show recommendation"):
    random_recommended = random_recommend_product(suggestions)
    col1, col2, col3, col4, col5 = sl.columns(5)
    with col1:
        sl.text(random_recommended[0]['product_name'].values[0])
        sl.image(random_recommended[0]['images'].values[0])
    with col2:
        sl.text(random_recommended[1]['product_name'].values[0])
        sl.image(random_recommended[1]['images'].values[0])
    with col3:
        sl.text(random_recommended[2]['product_name'].values[0])
        sl.image(random_recommended[2]['images'].values[0])
    with col4:
        sl.text(random_recommended[3]['product_name'].values[0])
        sl.image(random_recommended[3]['images'].values[0])
    with col5:
        sl.text(random_recommended[4]['product_name'].values[0])
        sl.image(random_recommended[4]['images'].values[0])
