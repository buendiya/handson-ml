# -*-coding:utf-8-*-

import tensorflow as tf
import numpy as np

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
y_test = y_test.astype(np.int32)

# saver = tf.train.Saver()
saver = tf.train.import_meta_graph('./my_model_final.ckpt.meta')

# n_inputs = 28*28  # MNIST
# X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")

init_op = tf.initialize_all_variables()

with tf.Session() as sess:
    init_op.run()

    saver.restore(sess, "./my_model_final.ckpt") # or better, use save_path
    graph = tf.get_default_graph()
    for op in sess.graph.get_operations():
        print(op.name)
    X = graph.get_tensor_by_name("X:0")
    logits = graph.get_tensor_by_name("dnn/outputs/add:0")

    X_new_scaled = X_test[:20]
    Z = logits.eval(feed_dict={X: X_new_scaled})
    y_pred = np.argmax(Z, axis=1)


print("Predicted classes:", y_pred)
print("Actual classes:   ", y_test[:20])

