# K6316Project
[K6316 Project] COVID-19 Epidemic Trends
Logistic Regression Model
#Random Guess
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit #Importing curve_fit function
from sklearn.metrics import mean_squared_error #MSE function
from sklearn.model_selection import train_test_split

data = pd.read_csv('E:\\201907-202005 NTU\\GoodGoodStudy\\K6312 Infomation Mining and Analysis\\group project\\PAPER\\data.csv')
x_train, x_test, y_train, y_test = train_test_split(data['date_number'], data['total_cases'], test_size=0.2) 

#Scatter plot of training data
fig=plt.figure(figsize=(16,8))
ax=fig.add_subplot(1, 1, 1)
ax.scatter(x_train,y_train, color="k", label="Training Data")
# Scatter plot of testing data
ax.scatter(x_test,y_test, color="RED",label="Testing Data") 
ax.legend()

#Apply logistic function and fit the curve
def logistic(x_train,K,P0,r): 
    exp_value = np.exp(0.1*(x_train))
    return (K*exp_value*P0)/(K+(exp_value-1)*P0)

coef, pcov = curve_fit(logistic, x_train, y_train)
print(coef) 

#Draw the curve
t = np.arange(0,120,0.01)
y_predict = logistic(t,coef[0], coef[1], coef[2])
ax.plot(t,y_predict,color="blue", linewidth=2, label="Fitting Curve") 

# Calculate RMSE
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
rmse_val = rmse(logistic(x_test, coef[0], coef[1], coef[2]), y_test)
print("rmse is: " + str(rmse_val))

#Grid Search
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit #Importing curve_fit function
from sklearn.metrics import mean_squared_error #MSE function
from sklearn.model_selection import train_test_split

data = pd.read_csv('E:\\201907-202005 NTU\\GoodGoodStudy\\K6312 Infomation Mining and Analysis\\group project\\PAPER\\data.csv')



x_train, x_test, y_train, y_test = train_test_split(data['date_number'], data['total_cases'], test_size=0.2) 


#Scatter plot of training data
fig=plt.figure(figsize=(16,8))
ax=fig.add_subplot(1, 1, 1)
ax.scatter(x_train,y_train, color="k", label="Training Data")
# Scatter plot of testing data
ax.scatter(x_test,y_test, color="RED",label="Testing Data") 
ax.legend()

hyperparameters_r = None
hyperparameters_K = None
#Apply logistic function and fit the curve
def logistic(x_train,P0): 
    r = hyperparameters_r
    K = hyperparameters_K
    exp_value = np.exp(r*(x_train))
    return (K*exp_value*P0)/(K+(exp_value-1)*P0)


def fitting(logistic, x_train, y_train):
    popt = None
    mse = float("inf")
    i = 0
    r = None
    k = None
    k_range = np.arange(1000000, 4000000, 100000)
    r_range = np.arange(0, 1, 0.01)
    for k_ in k_range:
        global hyperparameters_K
        hyperparameters_K = k_
        for r_ in r_range:
            global hyperparameters_r
            hyperparameters_r = r_
            # 用非线性最小二乘法拟合
            popt_, pcov_ = curve_fit(logistic, x_train, y_train, maxfev = 4000)
            # 采用均方误准则选择最优参数
            mse_ = mean_squared_error(y_train, logistic(x_train, *popt_))
            if mse_ <= mse:
                mse = mse_
                popt = popt_
                r = r_
                k = k_
            i = i+1
            print('\r当前进度：{0}{1}%'.format('▉'*int(i*10/len(k_range)/len(r_range)),int(i*100/len(k_range)/len(r_range))), end='')
    print('拟合完成')
    hyperparameters_K = k
    hyperparameters_r = r
    popt, pcov = curve_fit(logistic, x_train, y_train)
    print("K:capacity  P0:initial_value   r:increase_rate")
    print(hyperparameters_K, popt, hyperparameters_r)
    return hyperparameters_K, hyperparameters_r, popt

K, r, popt = fitting(logistic, x_train, y_train)

#Draw the curve
t = np.arange(0,120,0.01)
y_predict = logistic(t,*popt)
ax.plot(t,y_predict,color="blue", linewidth=2, label="Fitting Curve") 


# Calculate RMSE
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
rmse_val = rmse(logistic(x_test, *popt), y_test)
print("rmse is: " + str(rmse_val))
#print(rmse)
