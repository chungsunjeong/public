import scipy.io
import tensorflow as tf
import math
import numpy as np

def KernelFunctionMatrix(X1,X2,theta0,theta1,theta2,theta3):
    insideexp1=tf.multiply(tf.div(theta1,2.0),np.dot((X1-X2),(X1-X2)))
    inside1 = tf.multiply(theta0, tf.exp(insideexp1))
    inside2 = theta2
    inside3=tf.multiply(theta3,np.dot(np.transpose(X1),X2))
    ret=tf.add(tf.add(inside1,inside2),inside3)

    return ret

def KernelHyperParameterLearning(iteration,learningRate,trainingX,trainingY):
    numDataPoints=len(trainingY)
    numDimension=len(trainingX[0])

    obsX=tf.placeholder(tf.float32,[numDataPoints,numDimension])
    obsY=tf.placeholder(tf.float32,[numDataPoints,1])
    theta0 = tf.Variable(1.0)
    theta1 = tf.Variable(1.0)
    theta2 = tf.Variable(1.0)
    theta3 = tf.Variable(1.0)
    beta=tf.Variable(10.0)

    matCovarianceLinear=[]
    for i in range(numDataPoints):
        for j in range(numDataPoints):
            kernelEvaluationResult=KernelFunctionMatrix(tf.slice(obsX,[i,0],[1,numDimension]),
                                                        tf.slice(obsX,[j,0],[1,numDimension]),
                                                        theta0,theta1,theta2,theta3)
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
        # k = tf.Variable(tf.range(0,8,1.0),dtype=np.float32)
        for j in range(numDataPoints):
            kernelEvaluationResult=KernelFunctionMatrix(tf.slice(obsX,[i,0],[1,numDimension]),
                                                        tf.slice(obsX,[j,0],[1,numDimension]),
                                                        theta0,theta1,theta2,theta3)
            indices=tf.constant([j])
            tempTensor=tf.Variable(tf.zeros([1]))
            tempTensor=tf.add(tempTensor,kernelEvaluationResult)
            tf.scatter_update(k,tf.reshape(indices,[1,1]),tempTensor)
        c=tf.Variable(tf.zeros([1,1]))
        kernelEvaluationResult=KernelFunctionMatrix(tf.slice(obsX,[i,0],[1,numDimension]),
                                                    tf.slice(obsX,[i,0],[1,numDimension]),
                                                    theta0,theta1,theta2,theta3)
        c=tf.div(tf.add(tf.add(c,kernelEvaluationResult),1),beta)
        k=tf.reshape(k,[1,numDataPoints])

        predictionMu=tf.matmul(k,tf.matmul(matCovarianceInv,obsY))
        predictionVar=tf.subtract(c,tf.matmul(k,tf.matmul(matCovarianceInv,tf.transpose(k))))

        negloglikelihood=tf.add(negloglikelihood,tf.div(tf.pow(tf.subtract(predictionMu,tf.slice(obsY,[i,0],[1,1])),2)
                                                        ,tf.scalar_mul(tf.constant(2.0),predictionVar)))

    training=tf.train.GradientDescentOptimizer(learningRate).minimize(negloglikelihood)
    #
    # sess = tf.Session()
    # init = tf.global_variables_initializer()
    # sess.run(init)
    # for step in range(iteration):
    #     sess.run(training, feed_dict={obsX : trainingX, obsY: trainingY})
    #     print(training)
    #     print(negloglikelihood)

    # sess.close()


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(iteration):
            _, negloglikelihood_ = sess.run([training, negloglikelihood],feed_dict={obsX:trainingX,obsY:trainingY})
            print(sess.run([theta0, theta1, theta2,beta]))
            print(sess.run([negloglikelihood,predictionMu,predictionVar],feed_dict={obsX:trainingX,obsY:trainingY}))
            op_theta0=sess.run(theta0)
            op_theta1 = sess.run(theta1)
            op_theta2 = sess.run(theta2)
            op_beta = sess.run(beta)
            # op_negloglikelihood=sess.run(negloglikelihood)
            # op_predictionMu=sess.run(predictionMu)
            # op_predictionVar = sess.run(predictionVar)
    # op_negloglikelihood, op_predictionMu, op_predictionVar
    return op_theta0,op_theta1,op_theta2,op_beta

data_mat = scipy.io.loadmat('simple_sindata.mat')
data_yes = data_mat['y']
data_time=data_mat['X']
data_yes =  np.asarray(data_yes,dtype=np.float32)
data_time = np.asarray(data_time,dtype=np.float32)

# X=np.matlib.repmat(data_time,29,1)
X=data_time
y=data_yes
# y=data_yes[0:,0]
# y = tf.reshape(tf.constant(y), [-1, 1])
learning_R=0.001
iteration=100

sess=tf.Session()
numDataPoints=8
print(tf.Variable(tf.ones([numDataPoints])))
sess.close()
param=KernelHyperParameterLearning(iteration,learning_R,X,y)
# print(param)
# path="parameter.txt"
# f=open(path,'w')
# f.write(param)
# f.close()
