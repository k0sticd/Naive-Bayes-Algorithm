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
                var = model['_vars'][attribute][class_value] + 1e-9  # Dodaj mali broj da izbegne deljenje nulom
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
