import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
from xgboost import XGBRegressor

# Read the data
data = pd.read_csv('./elements.csv')

def preprocess_data(data, target_cols, drop_cols):
    # Select target and features
    y = data[target_cols]
    X = data.drop(drop_cols, axis=1)

    return X, y

def select_categorical_columns(X):
    return X.select_dtypes(include='object')


def process_categorical_features(data):
    # Get list of categorical variables
    s = (data.dtypes == 'object')
    object_cols = list(s[s].index)

    OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # Fit and get new column names
    OH_encoder.fit(data[object_cols])
    column_names = OH_encoder.get_feature_names_out(input_features=object_cols)

    # Apply one-hot encoder to each column with categorical data
    OH_cols = pd.DataFrame(OH_encoder.transform(data[object_cols]))

    # Rename generated columns
    OH_cols.columns = column_names

    # One-hot encoding removed index; put it back
    OH_cols.index = data.index

    # Remove categorical columns (will replace with one-hot encoding)
    num_data = data.drop(object_cols, axis=1)

    # Add one-hot encoded columns to numerical features
    OH_data = pd.concat([num_data, OH_cols], axis=1)

    # Ensure all columns have string type
    OH_data.columns = OH_data.columns.astype(str)

    return OH_data, column_names


def process_data_and_fit_model(target_cols, drop_cols, model):
    # Preprocess data and split
    X, y = preprocess_data(data, target_cols, drop_cols)

    X_train, X_valid, y_train, y_valid = train_test_split(X, y)

    X_train, _ = process_categorical_features(X_train);
    X_valid, _ = process_categorical_features(X_valid);
    y_train, _ = process_categorical_features(y_train);
    y_valid, _ = process_categorical_features(y_valid);

    # Fit the model
    model.fit(X_train, y_train)

    return model, X_train, y_train


def predictResizing(modelsDictionary, inputData):
    inputData = inputData[modelsDictionary['width']['x'].columns]
    predictions_width = modelsDictionary['width']['model'].predict(inputData)
    inputData['widthAfter'] = predictions_width[0];
    inputData = inputData[modelsDictionary['height']['x'].columns]
    predictions_height = modelsDictionary['height']['model'].predict(inputData)
    inputData['heightAfter'] = predictions_height[0];
    inputData = inputData[modelsDictionary['top']['x'].columns]
    predictions_top = modelsDictionary['top']['model'].predict(inputData);
    inputData['topAfter'] = predictions_top[0]
    inputData = inputData[modelsDictionary['left']['x'].columns]
    predictions_left = modelsDictionary['left']['model'].predict(inputData);
    inputData['leftAfter'] = predictions_left[0];

    return predictions_width[0], predictions_height[0], predictions_left[0], predictions_top[0]

def process_user_input(userInput):
    dfUserInput = pd.DataFrame(data=userInput)
    dfUserInput, column_names_input = process_categorical_features(dfUserInput)
    X_role, y_role = preprocess_data(data, ['role'], ['role'])
    X_role_oh, column_names = process_categorical_features(X_role)
    for column in column_names:
        if column not in dfUserInput: dfUserInput[column] = 0

    return dfUserInput

def predict_element_role(model, dfUserInput, role_y):
    predictions = model.predict(dfUserInput)

    max_index = np.argmax(predictions)
    return role_y.columns[max_index]

def split_data_and_train_models():
    role_model, role_x, role_y = process_data_and_fit_model(['role'], ['role', 'widthAfter', 'heightAfter', 'leftAfter',
                                                                       'topAfter', 'ratioHeightAfter',
                                                                       'ratioWidthAfter', 'ratioHeightBefore',
                                                                       'ratioWidthBefore'],
                                                            XGBRegressor(n_estimators=1000, learning_rate=0.05))
    model_width, width_x, width_y = process_data_and_fit_model(['widthAfter'],
                                                               ['widthAfter', 'heightAfter', 'leftAfter', 'topAfter',
                                                                'ratioHeightAfter', 'ratioWidthAfter',
                                                                'ratioHeightBefore', 'ratioWidthBefore'],
                                                               XGBRegressor(n_estimators=1000, learning_rate=0.05))
    model_height, height_x, height_y = process_data_and_fit_model(['heightAfter'],
                                                                  ['heightAfter', 'leftAfter', 'topAfter',
                                                                   'ratioHeightAfter', 'ratioWidthAfter',
                                                                   'ratioHeightBefore', 'ratioWidthBefore'],
                                                                  XGBRegressor(n_estimators=1000, learning_rate=0.05))
    model_top, top_x, top_y = process_data_and_fit_model(['topAfter'], ['leftAfter', 'topAfter', 'ratioHeightAfter',
                                                                        'ratioWidthAfter', 'ratioHeightBefore',
                                                                        'ratioWidthBefore'],
                                                         XGBRegressor(n_estimators=1000, learning_rate=0.05))
    model_left, left_x, left_y = process_data_and_fit_model(['leftAfter'],
                                                            ['leftAfter', 'ratioHeightAfter', 'ratioWidthAfter',
                                                             'ratioHeightBefore', 'ratioWidthBefore'],
                                                            XGBRegressor(n_estimators=1000, learning_rate=0.05))
    models = {'width': {'model': model_width, 'x': width_x, 'y': width_y},
              'height': {'model': model_height, 'x': height_x, 'y': height_y},
              'top': {'model': model_top, 'x': top_x, 'y': top_y},
              'left': {'model': model_left, 'x': left_x, 'y': left_y},
              'role': {'model': role_model, 'x': role_x, 'y': role_y}
              }

    with open('model_resizer.pkl', 'wb') as file:
        pickle.dump(models, file)


def predict_resizing(modelsDictionary, inputData):
    inputData = inputData[modelsDictionary['width']['x'].columns]
    predictions_width = modelsDictionary['width']['model'].predict(inputData)
    inputData['widthAfter'] = predictions_width[0];
    inputData = inputData[modelsDictionary['height']['x'].columns]
    predictions_height = modelsDictionary['height']['model'].predict(inputData)
    inputData['heightAfter'] = predictions_height[0];
    inputData = inputData[modelsDictionary['top']['x'].columns]
    predictions_top = modelsDictionary['top']['model'].predict(inputData);
    inputData['topAfter'] = predictions_top[0]
    inputData = inputData[modelsDictionary['left']['x'].columns]
    predictions_left = modelsDictionary['left']['model'].predict(inputData);
    inputData['leftAfter'] = predictions_left[0];

    return {'width': predictions_width[0],
            'height': predictions_height[0],
            'left': predictions_left[0],
            'top': predictions_top[0]
            }
