import scipy.io
import tensorflow as tf
import math
import numpy as np

def KernelFunctionMatrix(X1,X2,theta0,theta1,theta2):
    insideexp1=tf.multiply(tf.div(theta1,-2.0),np.dot((X1-X2),(X1-X2)))
    insideexp2=theta2
    ret=tf.add(tf.multiply(theta0,tf.exp(insideexp1)),insideexp2)

    return ret

def KernelHyperParameterLearning(learningRate,trainingX,trainingY):

    numDataPoints=len(trainingY)
    numDimension=len(trainingX[0])
    obsX=tf.placeholder(tf.float32,[numDataPoints,numDimension])
    obsY=tf.placeholder(tf.float32,[numDataPoints,1])
    theta0 = tf.Variable(1.0)
    theta1 = tf.Variable(1.0)
    theta2 = tf.Variable(1.0)
    beta=tf.Variable(10.0)

    print(numDataPoints)
    print(numDimension)
    print(obsX)
    print(obsY)
    matCovarianceLinear=[]
    for i in range(numDataPoints):
        for j in range(numDataPoints):
            kernelEvaluationResult=KernelFunctionMatrix(tf.slice(obsX,[i,0],[1,numDimension]),
                                                        tf.slice(obsX,[j,0],[1,numDimension]),
                                                        theta0,theta1,theta2)
            if i != j:
                matCovarianceLinear.append(kernelEvaluationResult)
            if i == j:
                matCovarianceLinear.append((kernelEvaluationResult+tf.div(1.0,beta)))

    matCovarianceCombined=tf.stack(matCovarianceLinear)
    matCovariance=tf.reshape(matCovarianceCombined,[numDataPoints,numDataPoints])
    matCovarianceInv=tf.reciprocal(matCovariance)

    negloglikelihood=0.0
    for i in range(numDataPoints):
        k=tf.Variable(tf.ones([numDataPoints]))
        for j in range(numDataPoints):
            kernelEvaluationResult=KernelFunctionMatrix(tf.slice(obsX,[i,0],[1,numDimension]),
                                                        tf.slice(obsX,[j,0],[1,numDimension]),
                                                        theta0,theta1,theta2)
            indices=tf.constant([j])
            tempTensor=tf.Variable(tf.zeros([1]))
            tempTensor=tf.add(tempTensor,kernelEvaluationResult)
            tf.scatter_update(k,tf.reshape(indices,[1,1]),tempTensor)
        c=tf.Variable(tf.zeros([1,1]))
        kernelEvaluationResult=KernelFunctionMatrix(tf.slice(obsX,[i,0],[1,numDimension]),
                                                    tf.slice(obsX,[i,0],[1,numDimension]),
                                                    theta0,theta1,theta2)
        c=tf.div(tf.add(tf.add(c,kernelEvaluationResult),1),beta)
        k=tf.reshape(k,[1,numDataPoints])

        predictionMu=tf.matmul(k,tf.matmul(matCovarianceInv,obsY))
        predictionVar=tf.subtract(c,tf.matmul(k,tf.matmul(matCovarianceInv,tf.transpose(k))))

        negloglikelihood=tf.add(negloglikelihood,tf.div(tf.pow(tf.subtract(predictionMu,tf.slice(obsY,[i,0],[1,1])),2)
                                                        ,tf.scalar_mul(tf.constant(2.0),predictionVar)))

    training=tf.train.GradientDescentOptimizer(learningRate).minimize(negloglikelihood)

    return training

data_mat = scipy.io.loadmat('emotiv_response_data.mat')
data_yes = data_mat['Yes_data']
data_no = data_mat['No_data']
data_time=data_mat['time']
data_yes = np.float32(data_yes)
data_time = np.float32(data_time)

# X=np.matlib.repmat(data_time,29,1)
X=data_time
y=data_yes[0:,0]
learning_rate=1
iteration=1

param=KernelHyperParameterLearning(learning_rate,X,y)
sess = tf.Session()
init=tf.global_variables_initializer()
sess.run(init)
for step in range(iteration):
    sess.run(param)

path="parameter.txt"
f=open(path,'w')
f.write(param)
f.close()
