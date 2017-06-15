
# coding: utf-8

# ## Imports

# In[ ]:

import numpy as np
import tensorflow as tf, sys
from PIL import Image
import io
get_ipython().magic(u'matplotlib inline')


# ## Classifier

# In[ ]:

def classifier(image_data, label_path, retrained_path):
    # Loads label file, strips off carriage return
    label_lines = [line.rstrip() for line 
                       in tf.gfile.GFile(label_path)]

    # Unpersists graph from file
    with tf.gfile.FastGFile(retrained_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:
        # Feed the image_data as input to the graph and get first prediction
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

        predictions = sess.run(softmax_tensor,                  {'DecodeJpeg/contents:0': image_data})

        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            print('%s (score = %.2f)' % (human_string, score))
            #return label_lines[node_id], predictions[0][node_id]


# ## Define path

# In[ ]:

# label
label_path = "/Users/justinwu/tf_files/retrained_labels.txt"
# retrained
retrained_path = "/Users/justinwu/tf_files/retrained_graph.pb"


# ## Image Recognition

# In[ ]:

# load image
img = Image.open('/Users/justinwu/Desktop/test.jpg', mode='r')
img


# In[ ]:

# Convert image to byte array
imgByteArray = io.BytesIO()
img.save(imgByteArray, format='JPEG')
imgByteArray = imgByteArray.getvalue()

# Classify
classifier(imgByteArray,label_path,retrained_path)


# ## Object detection

# In[ ]:

# change this as you see fit
image_path = '/Users/justinwu/Desktop/test2.jpg'

# Convert image to np.array
image = Image.open(image_path, mode='r')
image_array = np.array(image)

# Sliding window
scale_x = 7
scale_y = 5
y_len,x_len,_ = image_array.shape

for y in range(scale_y):
    for x in range(scale_x):
        print('(%s,%s)' % (x+1, y+1))
        cropped_image = Image.fromarray(image_array[(y*y_len)/scale_y:((y+1)*y_len)/scale_y,
                                      (x*x_len)/scale_x:((x+1)*x_len)/scale_x,:])
        imgByteArray = io.BytesIO()
        cropped_image.save(imgByteArray, format='JPEG')
        imgByteArray = imgByteArray.getvalue()

        # Classify
        classifier(imgByteArray,label_path,retrained_path)


# ## Check image

# In[ ]:

# change this as you see fit
image_path = '/Users/justinwu/Desktop/test2.jpg'

# Convert image to np.array
image = Image.open(image_path, mode='r')
image_array = np.array(image)

# Sliding window
scale_x = 7
scale_y = 5
y_len,x_len,_ = image_array.shape

for y in range(scale_y):
    for x in range(scale_x):
        cropped_image = Image.fromarray(image_array[(y*y_len)/scale_y:((y+1)*y_len)/scale_y,
                                      (x*x_len)/scale_x:((x+1)*x_len)/scale_x,:])
        plt.figure()
        plt.imshow(cropped_image)

