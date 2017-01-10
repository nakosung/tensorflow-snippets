import tensorflow as tf
import os
from tensorflow.python.tools.freeze_graph import freeze_graph

X = tf.placeholder(dtype=tf.float32,shape=(None,1),name='X')
Y = tf.placeholder(dtype=tf.float32,shape=(None,1))
W = tf.get_variable(name='W',shape=(1,1))
b = tf.get_variable(name='b',shape=(1,1))
O = tf.add(tf.matmul(X,W),b,name='Out')
dist = O - Y
loss = tf.reduce_sum(dist * dist) 
opt = tf.train.AdamOptimizer()
train = opt.minimize(loss)

x_batch = [[1.],[2.],[3.]]
y_batch = [[2.],[4.],[6.]]

init = tf.global_variables_initializer()

with tf.Session() as sess:
    os.system("rm -rf ./graph")
    tf.train.write_graph(sess.graph_def,'./graph','graph.pb',as_text=False)
        
    sess.run(init)
    
    for i in range(1000):
        _, l = sess.run([train,loss],feed_dict={X:x_batch,Y:y_batch})
        print(i,l)

    # save trainable var's weight
    saver = tf.train.Saver()    
    saver.save(sess,'./graph/test')
    
    # trainable var + graph_def ==> frozen graph
    freeze_graph(
        './graph/graph.pb',
        '',
        True,        
        './graph/test',
        'Out',
        'save/restore_all',
        'save/Const:0',
        './graph/frozen.pb',
        False,
        None
    )

# INFERENCE
from tensorflow.python.platform import gfile

# create a placeholder
X = tf.placeholder(tf.float32,[None,1])

# import frozen graph
with gfile.FastGFile('./graph/frozen.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    
    tf.import_graph_def(graph_def,input_map={'X':X})

graph = tf.get_default_graph()
Out = graph.get_tensor_by_name("import/Out:0")
with tf.Session() as sess:        
    out = sess.run(Out,feed_dict={X:[[1.],[2.]]})
    print(out)