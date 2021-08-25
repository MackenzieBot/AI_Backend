import numpy as np
from part2 import load_dict, clean_text
from part3 import bag_of_words_category, bag_of_words_response
from part4 import get_model
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)

CATEGORIES = ['password', 'conference', 'security', 'network', 'hardware']
N_OF_CATEGORIES = len(CATEGORIES)

def determine(query, model, cv):
    predictions = model.predict(cv.transform([query]).toarray())
    predicted_index = np.argmax(predictions)
    certainty = predictions.max(1)[0]
    #print(f'Report 1: {predicted_index} at {certainty*100}% of certainty.')
    
    if certainty < 0.6:
        return None
    else:
        return predicted_index


def app():
    data, docs_x, docs_y = load_dict('intents.json')
    
    cv_cat, train_x_cat, train_y_cat = bag_of_words_category(docs_x)
    model_cat = get_model(train_x_cat, train_y_cat, 'category')

    holder = [(), (), (), (), ()]
    for i in range(N_OF_CATEGORIES):
        cv, train_x, train_y, tags = bag_of_words_response(docs_x, docs_y, i)
        model = get_model(train_x, train_y, CATEGORIES[i])
        holder[i] = (model, cv)
    
    while True:
        query = input('You: ')
        clean_query = clean_text(query)
        
        if query.lower() == 'quit':
            break
        
        index = determine(clean_query, model_cat, cv_cat)
        if index is None:
            print("I don't have an answer for that yet. Please be more specific, or try with another question.")
            continue
        
        tag = determine(clean_query, holder[index][0], holder[index][1])
        if tag is None:
            print("I don't have an answer for that yet. Please be more specific, or try with another question.")
            continue
        else:
            # Checking if Query belongs to Hardware category 
            if (index == 4):
                device = input("Are you using a Windows (W) or a Mac (M)?: ")
                if (device.upper() == "W"): 
                    intent = data['intents'][CATEGORIES[index]][tag]['w_answer']
                    source = data['intents'][CATEGORIES[index]][tag]['w_source']
                elif (device.upper() == "M"):
                    intent = data['intents'][CATEGORIES[index]][tag]['m_answer']
                    source = data['intents'][CATEGORIES[index]][tag]['m_source']
            else: 
                intent = data['intents'][CATEGORIES[index]][tag]['answer']
                source = data['intents'][CATEGORIES[index]][tag]['source']
            print("From {} ".format(source))
            print("Answer: {}".format(intent))
        return intent
app()