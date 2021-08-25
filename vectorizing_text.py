from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

def bag_of_words_category(docs_x):
    """
    Create a bag of words representation for all queries, maps each query to
    the CATEGORY.
    """
    train_x = []  # Initialize input list
    train_y = []  # Initialize output list
    
    # Create output list
    for i, doc in enumerate(docs_x):
        train_x.extend(doc)  # Extend main x list with all queries on a category
        train_y.extend([i] * len(doc))  # Extend main y list with a number representing each query's category

    # Initialize Vectorizers
    cv_x = CountVectorizer()
    
    # Fit and transform training lists to its BOW representation
    train_x = cv_x.fit_transform(train_x).toarray()
    train_y = np.array(train_y)

    # Return the x vectorizer and the training lists
    return cv_x, train_x, train_y


def bag_of_words_response(docs_x, docs_y, index):
    """
    Create a bag of words representation for all queries in category given by
    index, maps each query to its TAG.
    """
    # Obtain all unique tags in given category
    tags = sorted(set(docs_y[index]))
        
    train_x = []  # Initialize input list
    train_y = []  # Initialize output list
    
    train_x = docs_x[index]
    train_y = docs_y[index]
    
    # Initialize Vectorizers
    cv_x = CountVectorizer()
    
    # Fit and transform training lists to its BOW representation
    train_x = cv_x.fit_transform(train_x).toarray()
    train_y = np.array(train_y)

    # Return the x vectorizer, the training lists, and the tags
    return cv_x, train_x, train_y, tags