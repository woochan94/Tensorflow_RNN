import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#MinMaxScaler 정의 -> data를 0부터 1사이의 값으로 변환(normalize)
def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    return numerator / (denominator + 1e-7)

#train Parameter
seq_length = 7 #7일 단위로 학습시키고 8일째를 예측
input_dim = 14 
hidden_dim = 14
output_dim = 1 # 강수량
learning_rate = 0.01
iterations = 5001

#data load
xy = np.loadtxt('2008~2016 일간날씨+배추평균가격(태백).csv', delimiter=',')
xy = MinMaxScaler(xy) #Normalize
x = xy
y = xy[:,[-1]]

#build dataset
#seq_length 만큼을 x, 그다음을 y 반복
dataX = []
dataY = []
for i in range(0, len(y) - seq_length):
    _x = x[i: i+seq_length]
    _y = y[i+seq_length]
    dataX.append(_x)
    dataY.append(_y)

#train/test set 나누기
train_size = int(len(dataY) * 0.9) #train size = 70%
test_size = len(dataY) - train_size
trainX, testX = np.array(dataX[0:train_size]), np.array(dataX[train_size:len(dataX)])
trainY, testY = np.array(dataY[0:train_size]), np.array(dataY[train_size:len(dataY)])

#input placeholders
X = tf.placeholder(tf.float32, [None, seq_length, input_dim])
Y = tf.placeholder(tf.float32, [None, 1])

#build a LSTM network (build Rnn)
cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, state_is_tuple=True,activation=tf.tanh)
outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype = tf.float32)
Y_pred = tf.contrib.layers.fully_connected(outputs[:, -1], output_dim, activation_fn=tf.tanh)

# cost/loss
with tf.name_scope('optimizer'):
    loss = tf.reduce_sum(tf.square(Y_pred - Y))  # sum of the squares
    tf.summary.scalar('loss',loss)
    # optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train = optimizer.minimize(loss)

#RMSE
targets = tf.placeholder(tf.float32, [None, 1])
predictions = tf.placeholder(tf.float32, [None, 1])
rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('./losslog_haenam3',sess.graph)
predictArray=[]

    # Training step
for i in range(iterations):
    _, step_loss, summary = sess.run([train, loss, merged], feed_dict={
                            X: trainX, Y: trainY})
    if i % 100 == 0:
        writer.add_summary(summary,global_step=i)
    if i % 1000 == 0:
        print("[step: {}] loss: {}".format(i, step_loss))

# Test step
for i in range(0,len(testX)):
    testXX = [testX[i]]
    test_predict = sess.run(Y_pred, feed_dict={X: testXX})
    predictArray.append(test_predict[-1])
    if i != len(testX)-1:
        testX[i+1][-1][-1] = test_predict[-1]
        
rmse_val = sess.run(rmse, feed_dict={
                    targets: testY, predictions: predictArray})
print("RMSE: {}".format(rmse_val))

    # Plot predictions
plt.figure(1)
plt.plot(testY, color = "red");
    
plt.plot(predictArray, color = "blue");
plt.xlabel("Time Period")
plt.ylabel("average temperature")
plt.show()
