# Project using mysql and joblib and slr

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = {
    'name':['vivek','kumar','raj','ajay'],
    'avg_income':[1000,2000,3000,5000],
    'house_age':[10.2,2.5,3.0,4.0],
    'num_rooms':[3,4,5,1],
    'price':[1000,7000,5000,2500]
}

# data frame
df = pd.DataFrame(data)

# handle missing values
df.drop("name",axis=1,inplace=True)

# feature scaling
scaler = StandardScaler()
df[['avg_income','house_age','num_rooms']] = scaler.fit_transform(df[['avg_income','house_age','num_rooms']])
print(df)
 
# split data
x = df[['avg_income','house_age','num_rooms']]
y = df['price']

# train test split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

# model train
model = LinearRegression()
model.fit(x_train,y_train)

# prediction
new_data = pd.DataFrame({
    'avg_income':[900],
    'house_age':[8],
    'num_rooms':[5]
})

# feature scaling
new_data[['avg_income','house_age','num_rooms']] = scaler.fit_transform(new_data[['avg_income','house_age','num_rooms']])
# prediction
predict_data = model.predict(new_data)
print(predict_data[0])