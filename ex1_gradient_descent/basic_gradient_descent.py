import numpy as np
import os



print("hello gradient descent")
print("hello 梯度下降")

current_directory = os.getcwd()
#print(current_directory)

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
    #print(X)
    #print(Y)
    offseet = 400
    x_train,y_train = X[:offseet],Y[:offseet]
    x_test,y_test = X[offseet:],Y[offseet:]

    return x_train,y_train,x_test,y_test

# 特征标准化
def normalize_feature(X):
    mu = np.mean(X,axis=0)  #计算数据的平均值（mean）
    sigma = np.std(X,axis=0) #计算数据的标准差（standard deviation）
    return (X-mu)/sigma, mu, sigma #数据标准化：对于每个数据点，将其减去平均值，然后除以标准差。可以使用如下公式进行数据标准化：standardized_data = (data - mean) / std

# 梯度下降算法
def gradient_descent(x,y,w,b,learning_rate,epochs):
    m=len(y)
    cost_history =[]
    for epoch in range(epochs):
        y_hat = np.dot(x,w)+b
        loss = y_hat - y
        cost = np.sum(loss ** 2)/(2*m)
        cost_history.append(cost)

        dw = (1/m)*np.dot(x.T, loss) #.T指的是转置矩阵
        db = (1/m)*np.sum(loss)

        w-=learning_rate*dw
        b-=learning_rate*db
    return w, b, cost_history

# 加载数据
x_train, y_train, x_test, y_test = load_data()

#归一化特征
x_train_norm,mu,sigma = normalize_feature(x_train)

x_test_norm = (x_test-mu)/sigma

# 初始化参数
w = np.zeros(x_train_norm.shape[1]) # 生成全是0的矩阵
b = 0

# 超参数
learning_rate = 0.01
epochs =1000


#运行梯度下降
w, b, cost_history = gradient_descent(x_train_norm,y_train,w,b,learning_rate,epochs)

#测试集上预测
y_pred = np.dot(x_train_norm,w)+b

#测试集上计算均方误差
mse= np.mean((y_pred - y_test) ** 2)
print(f"测试集上均方差为：{mse}")
