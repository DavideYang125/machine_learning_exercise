import numpy as np
import os



print("hello gradient descent")
print("hello 梯度下降")

current_directory = os.getcwd()
print(current_directory)

# 获取当前文件所在的目录
current_dir = os.path.dirname(__file__)

DATAFILE = 'housing.data'
DATAFULLFILE = os.path.join(current_directory,'ex1_gradient_descent', DATAFILE)

def load_data():
    #从housing data文件中加载数据
    housing_data=np.fromfile(DATAFULLFILE,sep=' ')
    feature_names = ['feature1','feature2','feature3','feature4','feature5','feature6','feature7','feature8','feature9','feature10','feature11','feature12','feature13','feature14']
    feature_nums=len(feature_names)
    housing_data=housing_data.reshape(housing_data.shape[0]//feature_nums,feature_nums)
    X,Y = housing_data[:,0:13], housing_data[:,-1]
    print(X)
    print(Y)
    offseet = 400
    x_train,y_train = X[:offseet],Y[:offseet]
    x_test,y_test = X[offseet:],Y[offseet:]

    return x_train,y_train,x_test,y_test


x_train, y_train, x_test, y_test = load_data()
