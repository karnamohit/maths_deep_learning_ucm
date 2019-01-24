import numpy as np
import os
import tensorflow as tf 
import sys
import scipy
import scipy.stats

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)    
    np.random.seed(seed)

f = open('cph_tracking_usabledata_a2_SB' ) # The CSV file KR made. 

N = int(float(sys.argv[1]))

a =np.zeros((2,N+1))
z =np.zeros((2,N+1))

x=0

for i in range(0,N):
    l = str(f.readline())
    l1 = l.split(',')
    x = int(l1[0])

    l2 = l1[1].split('\n')
    y = l2[0]

    a[0][i] = x  
    a[1][i] = y

print("SUBTRACTION STARTS HERE! ")

for i in range(0,N):

    if(i>1):
        z[0][i] = a[0][i] - a[0][i-1]
        z[1][i] = a[1][i] - a[1][i-1]


f.close()


n_steps = 1 # mini-batch size
n_inputs = 2  # x and y coord
n_neurons = 100
n_outputs = 2 # x and y coord
# batch_size = 50
w = 20 # win_size, previously
temp = int((N+1)/w)

#w = 10
q=0

x_win = np.zeros((1,w))
y_win = np.zeros((1,w))
sa_win = np.zeros((1,w))
ve_win = np.zeros((1,w))
pcc = np.zeros((temp,2))

for i in range(0,N):
    if i%w==0:
        i2 = int(i/w)
        # print("ACF: ", np.correlate(x_win[0][:], sa_win[0][:])) # 3rd Arg can be "same" or "full"
        
        #dum5 = scipy.stats.pearsonr(dum1, dum3)
        #print(dum5)
        pcc[i2][0] = np.correlate(x_win[0][:], sa_win[0][:])
        #dum5 = scipy.stats.pearsonr(dum2, dum4)
        pcc[i2][1] = np.correlate(y_win[0][:], ve_win[0][:])
        dum6 = pcc[i2][0]
        dum7 = pcc[i2][1]
        print('%f,%f\n' % (dum6,dum7))
        # print("Variance of new X: ",np.var(x_win))
        sa_win = np.zeros((1,w))  # the sa and ve are the onld values of X and y windows. We save 'em before they disappear.'
        ve_win = np.zeros((1,w))
        
        sa_win = x_win
        ve_win = y_win
        
        x_win = np.zeros((1,w))
        y_win = np.zeros((1,w))
        q=0
    
    x_win[0][q] = z[0][i]
    y_win[0][q] = z[1][i]
    q=q+1



#x_in = np.zeros((temp,n_steps,2))
x_in2 = np.zeros((N+1,n_steps,2)) # Phase 2
y_out = np.zeros((temp,n_steps,2)) # Phase 1
y_out2 = np.zeros((N+1,n_steps,2))
pcc_in = np.zeros((N+1,n_steps,2)) # Phase 1


print(z.shape)
#z1 = np.transpose(z) # transpose is not going to work!
z1 = np.reshape(z,(N+1,2))
print(z1.shape)
z2 = z1[:,np.newaxis,:]
print(z2.shape)
z3 = pcc[:,np.newaxis,:]

i2=0
for i in range(N+1):    # Creating x ie. Inputs
    
    x_in2[i,0,0]=z2[i,0,0]
    x_in2[i,0,1]=z2[i,0,1]
    
    #if i%w == 0:
        #x_in[i2,0,0]=z2[i,0,0]
        #x_in[i2,0,1]=z2[i,0,1]
        
        #x_in3[i2,0,0,0]=z4[i,0,0,0]
        #x_in3[i2,0,1,0]=z4[i,0,1,0]
        #
        #for i3 in (1,w):
            #x_in3[i2,0,0,w-i3]=z4[i-i3,0,0,0]
            #x_in3[i2,0,1,w-i3]=z4[i-i3,0,1,0]
        
        #i2=i2+1


for i in range(N):   # Creating y ie. true labels
      
    if all( [i % w == 0, (i/w) < temp-1 ]):
        y_out[int(i/w),0,0]=x_in2[i+w+1,0,0]
        y_out[int(i/w),0,1]=x_in2[i+w+1,0,1]
        
     #y_out2[i,0,0]=x_in2[i+1,0,0]
     #y_out2[i,1,0]=x_in2[i+1,1,0]

i2=0

for i in range(temp):   # Adding PCC values
    
    if i<temp:
        pcc_in[i2,0,0]=z3[i,0,0]
        pcc_in[i2,0,1]=z3[i,0,1]
        
        #pcc_in2[i,0,0]=z3[i,0,0]
        #pcc_in2[i,0,1]=z3[i,0,1]
        
        for i3 in range(1,w-1):
            pcc_in[i2+i3,0,0]=z3[i,0,0]
            pcc_in[i2+i3,0,1]=z3[i,0,1]
        
        i2=i2+w

#print(pcc_in.shape)
#print(x_in)
#print ("******-******-******")
#print(y_out)





ft = open('cph_tracking_usabledata_b1' ) # The CSV file KR made.
Nt = 399

a =np.zeros((2,Nt+1))
zt =np.zeros((2,Nt+1))
x=0
for i in range(0,Nt):
    l = str(ft.readline())
    l1 = l.split(',')
    x = int(l1[0])
    l2 = l1[1].split('\n')
    y = l2[0]
    a[0][i] = x
    a[1][i] = y
print("SUBTRACTION STARTS HERE! ")
for i in range(0,Nt):
    if(i>1):
        zt[0][i] = a[0][i] - a[0][i-1]
        zt[1][i] = a[1][i] - a[1][i-1]
ft.close()
#w = 10 # win_size, previously
temp = int((Nt+1)/w)
q=0
xt_win = np.zeros((1,w))
yt_win = np.zeros((1,w))
sat_win = np.zeros((1,w))
vet_win = np.zeros((1,w))
pcct = np.zeros((temp,2))
for i in range(0,Nt):
    if i%w==0:
        i2 = int(i/w)
        pcct[i2][0] = np.correlate(xt_win[0][:], sat_win[0][:])
        pcct[i2][1] = np.correlate(yt_win[0][:], vet_win[0][:])
        dum6 = pcct[i2][0]
        dum7 = pcct[i2][1]
        sat_win = np.zeros((1,w))
        vet_win = np.zeros((1,w))
        sat_win = xt_win
        vet_win = yt_win
        xt_win = np.zeros((1,w))
        yt_win = np.zeros((1,w))
        q=0
    xt_win[0][q] = zt[0][i]
    yt_win[0][q] = zt[1][i]
    q=q+1
print(zt.shape)
zt1 = np.reshape(zt,(Nt+1,2))
print(zt1.shape)
zt2 = zt1[:,np.newaxis,:]
print(zt2.shape)
zt3 = pcct[:,np.newaxis,:]

xt_in = np.zeros((Nt+1,n_steps,2))
yt_out = np.zeros((temp,n_steps,2))
pcc_in2 = np.zeros((Nt+1,n_steps,2))

i2=0
for i in range(Nt+1):    # Creating x ie. Inputs

    #if i%w == 0:
    xt_in[i,0,0]=zt2[i,0,0]
    xt_in[i,0,1]=zt2[i,0,1]

        #i2=i2+1

for i in range(temp):   # Creating y ie. true labels

    if all( [i % w == 0, (i/w) < temp-1 ]):
        yt_out[int(i/w),0,0]=xt_in[i+w+1,0,0]
        yt_out[int(i/w),0,1]=xt_in[i+w+1,0,1]


    #if all(i<(temp-1):
        #yt_out[i,0,0]=xt_in[i+1,0,0]
        #yt_out[i,0,1]=xt_in[i+1,0,1]

i2=0

for i in range(temp):   # Adding PCC values

    if i<temp:
        pcc_in2[i2,0,0]=zt3[i,0,0]
        pcc_in2[i2,0,1]=zt3[i,0,1]

        for i3 in range(1,w-1):
            pcc_in2[i2+i3,0,0]=zt3[i,0,0]
            pcc_in2[i2+i3,0,1]=zt3[i,0,1]

        i2=i2+w





n_iteration = 3500 # iteration through a batch
batch_size = 100

## Need to define a batch, ie. input and output data

reset_graph()


X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
#X = tf.placeholder(tf.float32, [None, n_steps, n_inputs*2])
y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])
#X = tf.placeholder(tf.float32, [None, n_steps, n_inputs*2, w+1])
#y = tf.placeholder(tf.float32, [None, n_steps, n_outputs, 1])

#wrapping output
cell = tf.contrib.rnn.OutputProjectionWrapper(
        tf.contrib.rnn.BasicRNNCell(num_units = n_neurons, activation = tf.nn.relu),output_size= n_outputs)

outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
print("outputs shape: ",outputs.shape)

op_losscalc = tf.strided_slice(outputs, [0, 0, 0], [N, 1, 2], [w, 1, 1])
print("shape of stride-sliced outputs tensor: ",op_losscalc.shape)

learning_rate = 0.001

loss = tf.reduce_mean(tf.square(op_losscalc-y))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
training_op = optimizer.minimize(loss)

saver = tf.train.Saver()

init = tf.global_variables_initializer()

#print("Phase 1 w/o PCC")
#print("Phase 1 w/ PCC")
#print("Phase 2 w/o PCC")
print("Phase 2 w/ ACF")

with tf.Session() as sess:
    init.run()
    
    for iteration in range(n_iteration):
        #X_batch = np.append(x_in, pcc_in, axis=2)
        X_batch = x_in2
        #X_batch = np.append(x_in2, pcc_in, axis=2)
        #print("X_batch shape: ",X_batch.shape)
        y_batch = y_out
        #print("y_batch shape: ",y_batch.shape)
        #tf.reshape(X_batch, [40,1,2,10])
        sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        #print("session has started...")
        if(iteration % 100 == 0):
            mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
            print(iteration, "MSE: ",mse)
        #print("session is almost done...")
    print(iteration, "MSE: ",mse)
    saver.save(sess,"ph2_rnn")
    print("Model saved.")

#*****************************************************************************************
#
#                     TESTING CODE BELOW
#
#*****************************************************************************************


Xt_batch = xt_in #np.append(xt_in, pcc_in2, axis=2) # for test
yt_batch = yt_out


with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    print("TEST BEGINS HERE...")
    
    mse_t = loss.eval(feed_dict={X:Xt_batch, y:yt_batch})
    print(mse_t)
