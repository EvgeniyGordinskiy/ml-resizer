import pandas as pd
from model import process_user_input, predict_element_role, predict_resizing
import pickle


def resize(user_input):
    with open('model_resizer.pkl', 'rb') as file:
        models = pickle.load(file)
        input_df = process_user_input(user_input)
        role = predict_element_role(models['role']['model'], input_df, models['role']['y'])
        default_data = pd.DataFrame(0, index=input_df.index, columns=models['role']['y'].columns)
        input_df = pd.concat([default_data, input_df], axis=1)
        input_df[role] = 1

        return predict_resizing(models, input_df)
