import pandas as pd
import numpy as np

#%% Learn 
def learn(data, class_att, smoothing=0.5):
    model = {}
    
    # Calculating apriori probability 
    apriori = data[class_att].value_counts()
    apriori = apriori / apriori.sum()
    model['_apriori'] = np.log(apriori + smoothing)

    # processing numerical atributes 
    numeric_attributes = data.select_dtypes(include=[np.number]).columns
    categorical_attributes = data.select_dtypes(include=[object]).drop(columns=[class_att]).columns
    
    # processing categorical atributes
    for attribute in categorical_attributes:
        mat_cont = pd.crosstab(data[attribute], data[class_att])
        mat_cont = (mat_cont + smoothing) / (mat_cont.sum(axis=0) + smoothing * len(mat_cont))
        model[attribute] = np.log(mat_cont + smoothing)

    # Gaussian distribution parameters for numerical atributes
    model['_means'] = {}
    model['_vars'] = {}
    for attribute in numeric_attributes:
        means = data.groupby(class_att)[attribute].mean()
        vars_ = data.groupby(class_att)[attribute].var()
        model['_means'][attribute] = means
        model['_vars'][attribute] = vars_
    
    return model

#%% Prediction
def predict(model, new_instance):
    class_probabilities = {}
    for class_value in model['_apriori'].index:
        probability = model['_apriori'][class_value]
        
        for attribute in model:
            if attribute == '_apriori':
                continue
            if attribute in model['_means']:
                # Handling numerical atributes
                mean = model['_means'][attribute][class_value]
                var = model['_vars'][attribute][class_value] + 1e-9  # Add a small number to avoid division by zero
                x = new_instance[attribute]
                prob = -0.5 * np.log(2 * np.pi * var) - ((x - mean) ** 2) / (2 * var)
                probability += prob
            elif attribute in new_instance:
                # Handling caategorical atributes
                if new_instance[attribute] in model[attribute].index:
                    prob = model[attribute][class_value].get(new_instance[attribute], np.log(1e-10))
                else:
                    prob = np.log(1e-10)
                probability += prob
        
        class_probabilities[class_value] = np.exp(probability)

    prediction = max(class_probabilities, key=class_probabilities.get)
    return prediction, class_probabilities


#%% TEST
from sklearn.model_selection import train_test_split
def load_data(file_path, test_size=0.3, random_state=7):
    
    data = pd.read_csv(file_path)
    
    X_train, data_new = train_test_split(data, test_size=test_size, random_state=random_state)
    return X_train, data_new

X_train, data_new = load_data('drug.csv') #example data is drug.csv
class_att = X_train.columns[-1]
model = learn(X_train, class_att, smoothing=0.3)

for index in data_new.index:
    # Extract rows from test_data based on indices
    instance = data_new.loc[index]
    
    # Prediction for the current row
    prediction, confidence = predict(model, instance)
    
    # Update the `data_new` DataFrame with predictions
    data_new.loc[index, 'prediction'] = prediction
    for klasa in confidence:
        data_new.loc[index, 'class=' + klasa] = confidence.get(klasa, 0)  # Use the default value 0 if the key is not present

print(data_new)
