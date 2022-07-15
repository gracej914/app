import pickle
import pandas as pd
from lightgbm import LGBMRegressor

df = pd.read_csv('beer_review_export.csv')

df['beer_style'] = df['beer_style'].astype('category')
features = ['beer_style', 'review_aroma','review_appearance',
                'review_palate','review_taste','beer_abv'] 
target = 'review_overall_ave'
X_train = df[features]
y_train = df[target]

model = LGBMRegressor(categorical_feature=0, 
                    n_estimators=700, max_depth=6,
                    learning_rate=0.4, random_state=0,
                    boosting_type='gbdt',n_jobs=-1)  

model.fit(X_train, y_train)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

