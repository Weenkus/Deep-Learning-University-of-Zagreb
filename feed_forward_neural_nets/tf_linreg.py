import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def main():
    ## 1. definicija računskog grafa
    # podatci i parametri
    X  = tf.placeholder(tf.float32, [None])
    Y_ = tf.placeholder(tf.float32, [None])
    a = tf.Variable(0.0)
    b = tf.Variable(0.0)

    # afini regresijski model
    Y = a * X + b

    # kvadratni gubitak
    loss = (Y-Y_)**2

    # optimizacijski postupak: gradijentni spust
    trainer = tf.train.GradientDescentOptimizer(0.1)
    train_op = trainer.minimize(loss)

    ## 2. inicijalizacija parametara
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    ## 3. učenje
    # neka igre počnu!
    for i in range(100):
        val_loss, _, val_a,val_b = sess.run([loss, train_op, a,b], feed_dict={X: [1,2], Y_: [3,5]})
        print(i,val_loss, val_a,val_b)

if __name__ == '__main__':
    main()
