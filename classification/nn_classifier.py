
import tensorflow as tf
import random

from classification.classifiers import *
import numpy as np


def fully_conn_net(_X, _weights, _biases, _dropout):

    """ IF you want two layers use this

    dense1 = tf.nn.relu(tf.add(tf.matmul(_X, _weights['wd1']), _biases['bd1'])) # Relu activation
    dense1 = tf.nn.dropout(dense1, _dropout) # Apply Dropout

    # NEW LAYER
    dense2 = tf.nn.relu(tf.add(tf.matmul(dense1, _weights['wd2']), _biases['bd2'])) # Relu activation
    dense2 = tf.nn.dropout(dense2, _dropout) # Apply Dropout

    # Output, class prediction
    out = tf.add(tf.matmul(dense2, _weights['out']), _biases['out'])
    #out = tf.tanh(tf.add(tf.matmul(dense1, _weights['out']), _biases['out']))
    """
    dense1 = tf.nn.relu(tf.add(tf.matmul(_X, _weights['wd1']), _biases['bd1'])) # Relu activation
    dense1 = tf.nn.dropout(dense1, _dropout) # Apply Dropout

    # Output, class prediction
    out = tf.add(tf.matmul(dense1, _weights['out']), _biases['out'])
    #out = tf.tanh(tf.add(tf.matmul(dense1, _weights['out']), _biases['out']))

    return out



def create_mini_batch(X,Y,batch_size):
    # For now it takes randomly
    batch = random.sample(range(0,len(X)), batch_size)
    #X_b = [ X[i] for i in batch ]
    #Y_b = [ Y[i] for i in batch ]
    # numpy implementation
    X_b = X.take(batch,axis=0)
    Y_b = Y.take(batch,axis=0)

    return X_b,Y_b


def round_values(X):
    #X = tf.tanh(X)
    return [[1 if elem > 0.5 else 0 for elem in line] for line in X]


def sigmoid_cross_entropy_with_logits_and_cost_weights(logits, targets,fp_adjust,fn_adjust, name=None):
    """Computes sigmoid cross entropy given `logits`.

    Measures the probability error in discrete classification tasks in which each
    class is independent and not mutually exclusive.  For instance, one could
    perform multilabel classification where a picture can contain both an elephant
    and a dog at the same time.

    For brevity, let `x = logits`, `z = targets`.  The logistic loss is

        z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
      = z * -log(1 / (1 + exp(-x))) + (1 - z) * -log(exp(-x) / (1 + exp(-x)))
      = z * log(1 + exp(-x)) + (1 - z) * (-log(exp(-x)) + log(1 + exp(-x)))
      = z * log(1 + exp(-x)) + (1 - z) * (x + log(1 + exp(-x))
      = (1 - z) * x + log(1 + exp(-x))
      = x - x * z + log(1 + exp(-x))

    To ensure stability and avoid overflow, the implementation uses

      max(x, 0) - x * z + log(1 + exp(-abs(x)))

    `logits` and `targets` must have the same type and shape.

    Args:
    logits: A `Tensor` of type `float32` or `float64`.
    targets: A `Tensor` of the same type and shape as `logits`.
    name: A name for the operation (optional).

    Returns:
    A `Tensor` of the same shape as `logits` with the componentwise
    logistic losses.
    """
    logits = tf.convert_to_tensor(logits, name="logits")
    targets = tf.convert_to_tensor(targets, name="targets")
    # The logistic loss formula from above is
    #   x - x * z + log(1 + exp(-x))
    # For x < 0, a more numerically stable formula is
    #   -x * z + log(1 + exp(x))
    # To avoid branching, we use the combined version
    #   max(x, 0) - x * z + log(1 + exp(-abs(x)))
    return tf.add(fn_adjust * tf.mul(targets, -tf.log(tf.sigmoid(logits) + 1e-5)),
                         fp_adjust * tf.mul((1. - targets) , -tf.log(1. - tf.sigmoid(logits) + 1e-5)))


def train_nn(X,Y,net_path,evaluation_cost):
    with tf.Graph().as_default(): # Needed because we want all the time different Graphs
        # Parameters
        learning_rate = 0.01#0.001
        training_iters = 300 * len(X)
        batch_size = 100
        display_step = 100000000
        #display_step = 100

        # Network Parameters
        n_input = len(X[0])
        n_classes = len(Y[0])
        dropout = 0.75 # Dropout, probability to keep units

        # tf Graph input
        x = tf.placeholder(tf.float32, [None, n_input])
        y = tf.placeholder(tf.float32, [None, n_classes])
        keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

        #OLD VERSION
        l1_increase = 1.5

        # Store layers weight & bias
        weights = {
            'wd1': tf.Variable(tf.random_normal([n_input, int(n_input*l1_increase)]),name='wd1'), # fully connected,
            'out': tf.Variable(tf.random_normal([int(n_input*l1_increase), n_classes]),name='wout') # class prediction
        }

        biases = {
            'bd1': tf.Variable(tf.random_normal([int(n_input*l1_increase)]),name='bd1'),
            'out': tf.Variable(tf.random_normal([n_classes]),name='bout')
        }

        """
        # NEW VERSION
        l1_increase = 1.5
        l2_decrease = 0.7
        min_possible_size_before_output = int(n_classes*1.2)

        l1_size = int(n_input*l1_increase)
        l2_size = max(int(l1_size*l2_decrease),min_possible_size_before_output)

        # Store layers weight & bias
        weights = {
            'wd1': tf.Variable(tf.random_normal([n_input, l1_size]),name='wd1'), # fully connected,
            'wd2': tf.Variable(tf.random_normal([l1_size, l2_size]),name='wd1'),
            'out': tf.Variable(tf.random_normal([l2_size, n_classes]),name='wout') # class prediction
        }

        biases = {
            'bd1': tf.Variable(tf.random_normal([l1_size]),name='bd1'),
            'bd2': tf.Variable(tf.random_normal([l2_size]),name='bd1'),
            'out': tf.Variable(tf.random_normal([n_classes]),name='bout')
        }
        """
        # We need to have a numpy array of floats:
        X = np.array(X).astype('float32')
        Y = np.array(Y).astype('float32')
        # the network
        pred = fully_conn_net(x,weights, biases,keep_prob)

        # Define loss and optimizer
        #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))

        # We want a cost sensitive learning! mispredicting positives is BAD!
        fp_cost = evaluation_cost[0]
        fn_cost = evaluation_cost[1]
        evaluation_function = cost_ev(fp_cost,fn_cost)

        max_mispred_cost = max(fp_cost,fn_cost)
        fp_adjust = fp_cost/max_mispred_cost
        fn_adjust = fn_cost/max_mispred_cost
        # The cost function takes those weights into account!
        #cost = y * -tf.log(tf.sigmoid(pred)) + (1 - y) * -tf.log(1 - tf.sigmoid(pred))
        cost = sigmoid_cross_entropy_with_logits_and_cost_weights(pred,y,fp_adjust,fn_adjust)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


        # Evaluate model
        correct_pred = tf.equal(tf.round(pred), y)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # Initializing the variables
        init = tf.initialize_all_variables()

        # Initializing the saver
        saver = tf.train.Saver()


        # Launch the graph
        with tf.Session() as sess:
            sess.run(init)
            step = 1
            # Keep training until reach max iterations
            while step * batch_size < training_iters:
                batch_xs, batch_ys = create_mini_batch(X,Y,batch_size)
                # Fit training using batch data

                sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})
                if step % display_step == 0:
                    # Calculate batch accuracy
                    #acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})

                    # Calculate batch loss
                    #loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
                    #print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + "%.6f"%(loss) + ", Training Accuracy= " + "%.5f"%(acc))

                    act_res =sess.run(pred,feed_dict={x: batch_xs, keep_prob: 1.})
                    #print(act_res)
                    # The result is Not Exact! we need to round it!
                    act_res = round_values(act_res)
                    act_conf = compute_confusion_matrix(act_res,batch_ys)
                    print("Step %d, train of %d. Cost function: %f\nEvals: acc: %f, prec: %f, reca: %f, f1:%f" % (step,step*batch_size,evaluation_function(act_conf[0][0],act_conf[1][1],act_conf[1][0],act_conf[0][1]),
                                                                    accuracy_ev(act_conf[0][0],act_conf[1][1],act_conf[1][0],act_conf[0][1])
                                                                   ,precision_ev(act_conf[0][0],act_conf[1][1],act_conf[1][0],act_conf[0][1]),
                                                                   recall_ev(act_conf[0][0],act_conf[1][1],act_conf[1][0],act_conf[0][1]),
                                                                   f1_ev(act_conf[0][0],act_conf[1][1],act_conf[1][0],act_conf[0][1])))


                    # Saves the network
                    save_path = saver.save(sess, net_path)
                step += 1
            # Saves the network
            save_path = saver.save(sess, net_path)
            print("Optimization Finished!")

        return saver


def predict_nn(X,n_classes,net_path):
    with tf.Graph().as_default():
        # Network Parameters
        n_input = len(X[0])
        dropout = 0.75 # Dropout, probability to keep units

        # tf Graph input
        x = tf.placeholder(tf.float32, [None, n_input])
        y = tf.placeholder(tf.float32, [None, n_classes])
        keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

        #OLD VERSION
        l1_increase = 1.5
        # Store layers weight & bias
        weights = {
            'wd1': tf.Variable(tf.random_normal([n_input, int(n_input*l1_increase)]),name='wd1'), # fully connected,
            'out': tf.Variable(tf.random_normal([int(n_input*l1_increase), n_classes]),name='wout') # class prediction
        }

        biases = {
            'bd1': tf.Variable(tf.random_normal([int(n_input*l1_increase)]),name='bd1'),
            'out': tf.Variable(tf.random_normal([n_classes]),name='bout')
        }
        """
        # NEW VERSION
        l1_increase = 1.5
        l2_decrease = 0.7
        min_possible_size_before_output = int(n_classes*1.2)

        l1_size = int(n_input*l1_increase)
        l2_size = max(int(l1_size*l2_decrease),min_possible_size_before_output)

        # Store layers weight & bias
        weights = {
            'wd1': tf.Variable(tf.random_normal([n_input, l1_size]),name='wd1'), # fully connected,
            'wd2': tf.Variable(tf.random_normal([l1_size, l2_size]),name='wd1'),
            'out': tf.Variable(tf.random_normal([l2_size, n_classes]),name='wout') # class prediction
        }

        biases = {
            'bd1': tf.Variable(tf.random_normal([l1_size]),name='bd1'),
            'bd2': tf.Variable(tf.random_normal([l2_size]),name='bd1'),
            'out': tf.Variable(tf.random_normal([n_classes]),name='bout')
        }
        """

        X = np.array(X).astype('float32')
        # the network
        pred = fully_conn_net(x,weights, biases,keep_prob)
        saver = tf.train.Saver()

        with tf.Session() as sess:
            saver.restore(sess, net_path)

            return round_values(sess.run(pred,feed_dict={x: X, keep_prob: 1.}))


def test_nn(X,Y,net_path):
    """

    tests the models with obs and truths and returns a confusion matrix
    """

    predictions = predict_nn(X,len(Y[0]),net_path)
    tp,tn,fp,fn = 0,0,0,0
    for i, ro in enumerate(predictions):
        for j, p in enumerate(ro):
            if p == 1:
                if Y[i][j] == 1: tp += 1
                else: fp += 1
            else:
                if Y[i][j] == 0: tn += 1
                else: fn += 1
    conf = [[tp,fn],[fp,tn]]
    return conf



