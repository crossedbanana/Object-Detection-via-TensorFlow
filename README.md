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

#### Relaunch Docker

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

