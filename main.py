# Machine learning program using linear regression (correlation)
from sklearn.linear_model import LinearRegression # library for linear models
import random # random number generator

value_set = [] # two empty lists for values (x,y,z)
function_set = [] # and functions (linear)

num_rows = 200 # get number of rows

limit = 2000 # limit values

for i in range(0, num_rows): # for the number of rows
    x = random.randint(0, limit) # x value = random number
    y = random.randint(0, limit) # y value = random number
    z = random.randint(0, limit) # z value = random number

    function = (4*x)+(7*y)+(9*z) # linear function for the target data set

    value_set.append([x,y,z]) # append the values of x,y,z to the value set list
    function_set.append(function) # append the linear function to the function set list

model = LinearRegression() # Linear Regression model
model.fit(value_set, function_set) # fit the value set and function set into the model

predict_set = [[54,0,101]] # test data - x = 54, y = 0, z = 101
prediction = model.predict(predict_set) # predict the data values ( get the sum)

# print the sum of the function and get the coefficients in the linear function
print('Machine Learning Prediction:' + str(prediction) + 'Coefficients in the Function:' + str(model.coef_))