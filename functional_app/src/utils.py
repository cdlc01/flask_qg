import pandas as pd
import pickle

def extract_feature_values(data):
    """ Given a params dict, return the values for feeding into a model"""
    
    # Replace these features with the features for your model. They need to 
    # correspond with the `name` attributes of the <input> tags
    
    with open('src/feature_names.pkl', 'rb') as f:
        expected_features = pickle.load(f)
    
    EXPECTED_FEATURES = expected_features

    # This assumes all inputs will be numeric. If you have categorical features
    # that the user enters as a string, you'll want to rewrite this as a for
    
    # loop that treats different features differently
    
    values = [int(data[feature]) for feature in EXPECTED_FEATURES]
    
    return pd.DataFrame(values, columns=EXPECTED_FEATURES)
