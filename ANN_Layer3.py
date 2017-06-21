"""Simple tutorial following the TensorFlow example of a Convolutional Network.
Parag K. Mital, Jan. 2016"""
# %% Imports
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import sys
#from libs.utils import *
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from numpy import linalg
import os
from sklearn import metrics

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def weight_variable(shape, name=None):
    '''Helper function to create a weight variable initialized with
    a normal distribution
    Parameters
    ----------
    shape : list
        Size of weight variable
    '''
    initial = tf.random_normal(shape, name=name, mean=0.0, stddev=0.01)
    return tf.Variable(initial)


# %%
def bias_variable(shape, name=None):
    '''Helper function to create a bias variable initialized with
    a constant value.
    Parameters
    ----------
    shape : list
        Size of weight variable
    '''
    initial = tf.random_normal(shape, name=name, mean=0.0, stddev=0.01)
    return tf.Variable(initial)

def open_fits(directory, filename):
    '''Helper function to open fits file and return Truth and Data tables
    '''
    #print(os.path.join(directory, filename))
    hdulist = fits.open(os.path.join(directory, filename))
    data = hdulist[1].data

    ##Extracting Truth Table
    Truth_int = data.field('BAL_FLAG_VI')

    ##Extracting data
    Data_int = np.append(data.field('LAMBDA  ') / linalg.norm(data.field('LAMBDA  ')),
                         data.field('FLUX    ') / linalg.norm(data.field('FLUX    ')), axis=1)
    Data_int = np.append(Data_int, data.field('IVAR_FLUX') / linalg.norm(data.field('IVAR_FLUX')), axis=1)

    hdulist.close()
    return [Truth_int,Data_int]


def main():
    ###Input section###

    directory = "./TestSet"
    
    print(directory)

    for filename in os.listdir(directory):
        if filename.endswith(".fits"):
            print(os.path.join(directory, filename))
            hdulist = fits.open(os.path.join(directory, filename))
            data = hdulist[1].data

            ##Extracting Truth Table
            Truth_int = data.field('BAL_FLAG_VI')

            ##Extracting data
            Data_int = np.append(data.field('LAMBDA  ') / linalg.norm(data.field('LAMBDA  ')),
                                 data.field('FLUX    ') / linalg.norm(data.field('FLUX    ')), axis=1)
            Data_int = np.append(Data_int, data.field('IVAR_FLUX') / linalg.norm(data.field('IVAR_FLUX')), axis=1)

            ##Concatenating
            try:
                Truth_val, Data_val
            except NameError:
                Truth_val = Truth_int
                Data_val = Data_int

            else:
                Truth_val = np.append(Truth_val, Truth_int)
                Data_val = np.append(Data_val, Data_int, axis=0)

            hdulist.close()

            continue
        else:
            continue
            
    print(filename)

    ##Checking for number of 0 and 1 in truth table
    #unique, counts = np.unique(Truth, return_counts=True)
    #print(dict(zip(unique, counts)))


    ###Architecture sections###

    ##Custom variables
    #nb_epoch = tf.Variable(..., name="nb_epoch")

    #Architecture

    n_in = 13800
    N =1 
    n_out = 1
    x = tf.placeholder(tf.float32, [None, n_in])
    y = tf.placeholder(tf.float32, [None, n_out])

    # %% Create a fully-connected layer:
    n_fc = 256*N
    W_fc1 = weight_variable([n_in, n_fc], 'w1')
    b_fc1 = bias_variable([n_fc], 'b1')
    h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)

    # %% We can add dropout for regularizing and to reduce overfitting like so:
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # %% Second fully-connected layer:
    n_fc2 = 128*N
    W_fc2 = weight_variable([n_fc, n_fc2], 'w2')
    b_fc2 = bias_variable([n_fc2], 'b2')
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    keep_prob2 = tf.placeholder(tf.float32)
    h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob2)

    # %% Third fully-connected layer:
    n_fc3 = 64*N
    W_fc3 = weight_variable([n_fc2, n_fc3], 'w3')
    b_fc3 = bias_variable([n_fc3], 'b3')
    h_fc3 = tf.nn.relu(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)

    h_fc3_drop = tf.nn.dropout(h_fc3, keep_prob2)

    # %% And finally our softmax layer:
    W_fc4 = weight_variable([n_fc3, n_out], 'w4')
    b_fc4 = bias_variable([n_out], 'b4')
    y_pred = tf.matmul(h_fc3_drop, W_fc4) + b_fc4

    ##Parameter section##

    # %% Define loss/eval/training functions
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred, labels=y)
    #cross_entropy = -tf.reduce_sum(y * tf.log(y_pred))

    learningrate = tf.placeholder(tf.float32)
    optimizer = tf.train.AdamOptimizer(learningrate).minimize(cross_entropy)

    # %% Monitor accuracy


    #correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    error_prediction = tf.reduce_mean(tf.cast(cross_entropy, 'float'))
    #accuracy = tf.reduce_mean(tf.nn.sigmoid(tf.reshape(y_pred, [-1])))
    prediction = tf.nn.sigmoid(tf.reshape(y_pred, [-1]))

    ##Preparing Saving##
    # Add an op to initialize the variables.
    init_op = tf.global_variables_initializer()

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()
    sess = tf.Session()

    if(len(sys.argv)<=1):
        # %% We now create a new session to actually perform the initialization the
        # variables:
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
    else:
        # Later, launch the model, use the saver to restore variables from disk, and
        # do some work with the model.

        saver.restore(sess, sys.argv[1])
        print("Model restored.")

        # Do some work with the model



    # %% We'll train in minibatches and report accuracy:
    batch_size = 100
    n_epochs = 300
    error_old = 100000.0
    learnrate = 0.001

    L_epoch = []
    T_error = []
    V_error = []
    AUC=[]

    for epoch_i in range(n_epochs):
        print("epoch ", epoch_i)
        ##Variable learning rate
        if epoch_i % 100 == 0:
            #learnrate = learnrate / 10.
            print("Reducing learning rate to ", learnrate)




        directory = "./TrainingSet"
        for filename in os.listdir(directory):
            if filename.endswith(".fits"):
                ##Import Data
                #print(filename)
                [Truth,Data] = open_fits(directory,filename)


                l = len(Data)
                n = len(Data) // batch_size
                for ndx in range(0, l - l % n, n):
                    batch_xs = Data[ndx:min(ndx + n, l)]
                    batch_xs = batch_xs.reshape(n, n_in)
                    # (Data[ndx:min(ndx + n, l)][0])
                    # print(batch_xs[0])
                    batch_ys = Truth[ndx:min(ndx + n, l)]
                    batch_ys = batch_ys.reshape(n, 1)
                    # print(Truth[ndx:min(ndx + n, l)][0])
                    # print(batch_ys[0])
                    sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.5, keep_prob2: 0.5, learningrate: learnrate})
                    # print(acc_temp)

                continue
            else:
                continue

        #training_acc = sess.run(accuracy, feed_dict={x: Data.reshape(len(Data), n_in), keep_prob: 1.0, keep_prob2: 1.0})
        training_error = sess.run(error_prediction, feed_dict={x: Data.reshape(len(Data), n_in),
                                                           y: Truth.reshape(len(Truth), 1), keep_prob: 1.0,
                                                           keep_prob2: 1.0})
        error_cont = sess.run(error_prediction, feed_dict={x: Data_val.reshape(len(Data_val),n_in),
                                                           y: Truth_val.reshape(len(Truth_val),1), keep_prob: 1.0, keep_prob2: 1.0})
        pred = sess.run(prediction, feed_dict={x: Data_val.reshape(len(Data_val), n_in), keep_prob: 1.0, keep_prob2: 1.0})

        print("Positive pred ", pred[5030], "  ", Truth_val[5030])
        print(error_cont)
        #print(pred)
        #print(Truth)

        threshold = 0.3

        error = 0
        #print("Positive acc ",acc[5015])
        TP,FP,TN,FN = 0,0,0,0
        for i in range(len(pred)):
            pred_int= 0
            if pred[i] > threshold:
                pred_int = 1
            if (pred_int != Truth_val[i]) and (pred_int == 0):
                FN += 1
            elif (pred_int != Truth_val[i]) and (pred_int == 1):
                FP += 1
            elif (pred_int == Truth_val[i]) and (pred_int == 0):
                TN += 1
            elif (pred_int == Truth_val[i]) and (pred_int == 1):
                TP += 1

            error += abs(pred_int-Truth_val[i])
            #error_cont += abs(acc[i]-Truth_val[i])
        print ("With tr at 0.5: Error = ",  error, "TP = ",  TP, "FP = ",  FP, "FN = ",  FN, ", Validation error = ", error_cont,", Training error = ",training_error,", AUC val = ", metrics.roc_auc_score(Truth_val, pred))
        T_error.append(training_error)
        V_error.append(error_cont)
        L_epoch.append(epoch_i)
        AUC.append(metrics.roc_auc_score(Truth_val, pred))

        #print(Truth_val)
        if error_old > training_error:
			error_old = training_error
			output_name = "./Models/L3N"+str(n_fc)+"_"+str(n_fc2)+"_"+str(n_fc3)+"TMax"
			save_path = saver.save(sess, output_name+".ckpt")
			print("Model saved in file: %s" % save_path)
			##Plot ROCcurve

			fpr, tpr, thresholds = metrics.roc_curve(Truth_val, pred)

			f = open(output_name + "Pred.txt", "w")
			f.write("\n".join(map(lambda x: str(x), AUC)))
			f.close()

    ##Training information saved as fig##
    plt.figure(1)
    plt.plot(np.log(fpr),tpr)
    plt.savefig(output_name + "_ROC.png")
    plt.figure(2)
    plt.plot(L_epoch,T_error,'r')
    plt.plot(L_epoch,V_error,'b')
    plt.savefig(output_name+"_Error.png")
    plt.show()




if __name__ == "__main__":
    main()
