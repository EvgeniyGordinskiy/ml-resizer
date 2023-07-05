import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# Read the data
data = pd.read_csv('./elements.csv')

# Select target width
y_width = data.widthAfter
X_width = data.drop(['widthAfter', 'heightAfter', 'leftAfter', 'topAfter', 'visibleAfter'], axis=1)

# Separate data into training and validation sets
X_train_width, X_valid_width, y_train_width, y_valid_width = train_test_split(X_width, y_width)

# Select height
y_height = data.heightAfter
X_height = data.drop(['heightAfter', 'leftAfter', 'topAfter', 'visibleAfter'], axis=1)

# Separate data into training and validation sets
X_train_height, X_valid_height, y_train_height, y_valid_height = train_test_split(X_height, y_height)

# Select left
y_left = data.leftAfter
X_left = data.drop(['leftAfter', 'topAfter', 'visibleAfter'], axis=1)

# Separate data into training and validation sets
X_train_left, X_valid_left, y_train_left, y_valid_left = train_test_split(X_left, y_left)

# Select top
y_top = data.topAfter
X_top = data.drop(['topAfter', 'visibleAfter'], axis=1)

# Separate data into training and validation sets
X_train_top, X_valid_top, y_train_top, y_valid_top = train_test_split(X_top, y_top)

# TRAIN

# width
my_model_width = XGBRegressor(n_estimators=1000, learning_rate=0.05)
my_model_width.fit(X_train_width, y_train_width)

# height
my_model_height = XGBRegressor(n_estimators=1000, learning_rate=0.05)
my_model_height.fit(X_train_height, y_train_height)

# left
my_model_left = XGBRegressor(n_estimators=1000, learning_rate=0.05)
my_model_left.fit(X_train_left, y_train_left)

# top
my_model_top = XGBRegressor(n_estimators=1000, learning_rate=0.05)
my_model_top.fit(X_train_top, y_train_top)

models = {
    'width': my_model_width,
    'height': my_model_height,
    'top': my_model_top,
    'left': my_model_left,
}


def resize(dict_values, models=models):
    d = {
        'sizeWidthBefore': [dict_values['sizeWidthBefore']],
        'sizeHeightBefore': [dict_values['sizeHeightBefore']],
        'widthBefore': [dict_values['widthBefore']],
        'heightBefore': [dict_values['heightBefore']],
        'leftBefore': [dict_values['leftBefore']],
        'topBefore': [dict_values['topBefore']],
        'visibleBefore': [True],
        'sizeWidthAfter': [dict_values['sizeWidthAfter']],
        'sizeHeightAfter': [dict_values['sizeHeightAfter']]}
    df = pd.DataFrame(data=d)
    predictions_width = models['width'].predict(df)
    print(predictions_width, 'width')
    df['widthAfter'] = predictions_width
    predictions_height = models['height'].predict(df)
    print(predictions_height, 'height')
    df['heightAfter'] = predictions_height
    predictions_left = models['left'].predict(df)
    print(predictions_left, 'left')
    df['leftAfter'] = predictions_left
    predictions_top = models['top'].predict(df)
    return {'width': predictions_width[0], 'height': predictions_height[0], 'left': predictions_left[0],
            'top': predictions_top[0]}
