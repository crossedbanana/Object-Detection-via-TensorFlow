# Object-Detection-via-TensorFlow
This repository shows a simple way to implement sliding window object detection in images via TensorFlow. 

## Prerequisites
* Python
* TensorFlow

#### Set up TensorFlow

Experienced users may prefer to [install TensorFlow manually](https://www.tensorflow.org/install/), and skip this section.
This repository recommends using Docker (see below).

##### Setup Docker

If you don't have docker installed already you can [download the installer here](https://www.docker.com/community-edition).

##### Test your Docker installation
To test your Docker installation try running the following command in the terminal :
```
docker run hello-world
```
This should output some text starting with:
```
Hello from Docker!
This message shows that your installation appears to be working correctly.
...
```
##### Run and test the TensorFlow image
Now that you've confirmed that Docker is working, test out the TensorFlow image:
```
docker run -it tensorflow/tensorflow:1.1.0 bash
```
After downloading your prompt should change to root@xxxxxxx:/notebooks#.

Next check to confirm that your TensorFlow installation works by invoking Python from the container's command line:
```
# Your prompt should be "root@xxxxxxx:/notebooks" 
python
```
Once you have a python prompt,``` >>>```, run the following code:
```
# python

import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session() # It will print some warnings here.
print(sess.run(hello))
```
This should print ```Hello TensorFlow!``` (and a couple of warnings after the tf.Session line).

##### Exit Docker
Now press ```Ctrl-d```, on a blank line, once to exit python, and a second time to exit the docker image.

### Relaunch Docker

Now create the working directory:
```
mkdir tf_files
```
Then relaunch Docker with that directory shared as your working directory, and port number 6006 published for TensorBoard:
```
docker run -it \
  --publish 6006:6006 \
  --volume ${HOME}/tf_files:/tf_files \
  --workdir /tf_files \
  tensorflow/tensorflow:1.1.0 bash
```
Your prompt will change to ```root@xxxxxxxxx:/tf_files#```

## Retrieve training images
In order to train the TensorFlow model, we need to gather some images of different categories. I already gathered some sample images that you can download from this repository. Go ahead to download the folder ```train_images```. 

You may also gather your own training images. Make sure you place them in folders which labeled with corresponding categories.

## Retrain Inception

The retrain script is part of the tensorflow repo, but it is not installed as part of the pip package. So you need to download it manually, to the current directory:
```
curl -O https://raw.githubusercontent.com/tensorflow/tensorflow/r1.1/tensorflow/examples/image_retraining/retrain.py
```
curl -O https://raw.githubusercontent.com/tensorflow/tensorflow/r1.1/tensorflow/examples/image_retraining/retrain.py
At this point, we have a trainer, we have data, so let's train! We will train the Inception v3 network.

As noted in the introduction, Inception is a huge image classification model with millions of parameters that can differentiate a large number of kinds of images. We're only training the final layer of that network, so training will end in a reasonable amount of time.

Start your image retraining with one big command (note the --summaries_dir option, sending training progress reports to the directory that tensorboard is monitoring) :
```
python retrain.py \
  --bottleneck_dir=bottlenecks \
  --how_many_training_steps=500 \
  --model_dir=inception \
  --summaries_dir=training_summaries/basic \
  --output_graph=retrained_graph.pb \
  --output_labels=retrained_labels.txt \
  --image_dir=flower_photos
```

This script downloads the pre-trained Inception v3 model, adds a new final layer, and trains that layer on the sample photos you've downloaded.

The above example iterates only 500 times. If you skipped the step where we deleted most of the training data and are training on the full dataset you can very likely get improved results (i.e. higher accuracy) by training for longer. To get this improvement, remove the parameter --how_many_training_steps to use the default 4,000 iterations.

```
python retrain.py \
  --bottleneck_dir=bottlenecks \
  --model_dir=inception \
  --summaries_dir=training_summaries/long \
  --output_graph=retrained_graph.pb \
  --output_labels=retrained_labels.txt \
  --image_dir=flower_photos
```
More detailed steps and explanation about retraining images can be found [here](https://www.tensorflow.org/tutorials/image_retraining).



