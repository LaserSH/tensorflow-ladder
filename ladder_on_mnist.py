import random
import input_data_mnist
import ladder_network
import tensorflow as tf

unlabeled_size = 0
labeled_size = 50000
mnist_size = 60000

print "Loading MNIST data"
mnist = input_data_mnist.read_data_sets(
    "/zfsauton/home/hanqis/data/mnist",
    one_hot=True,
    flatten=True,
    labeled_size=labeled_size,
    validation_size=mnist_size - unlabeled_size - labeled_size)

print mnist.train_unlabeled.num_examples, "unlabeled training examples"
print mnist.train_labeled.num_examples, "labeled training examples"
print mnist.validation.num_examples, "validation examples"
print mnist.test.num_examples, "test examples"


hyperparameters = {
  "learning_rate": 0.001,
  "noise_level": 0.2,
  "input_layer_size": [784],
  "class_count": 10,
  "is_cnn": False,
  "encoder_layer_definitions": [
    (1000, tf.nn.relu), # first hidden layer
    (500, tf.nn.relu), 
    (250, tf.nn.relu), 
    (250, tf.nn.relu), 
    (250, tf.nn.relu), 
    (10, tf.nn.softmax) # output layer
  ],
  "denoising_cost_multipliers": [
    1000, # input layer
    1,
    0.01, 
    0.01,
    0.01,
    0.01,
    0.01 # output layer
  ]
}

graph = ladder_network.Graph(**hyperparameters)

with ladder_network.Session(graph) as session:
  for step in xrange(10000):
    if step % 5 >= 0:
      images, labels = mnist.train_labeled.next_batch(100)
      session.train_supervised_batch(images, labels, step)
    elif unlabeled_size > 0:
      images, _ = mnist.train_unlabeled.next_batch(100)
      session.train_unsupervised_batch(images, step)

    if step % 200 == 0:
      save_path = session.save()
      accuracy = session.test(
        mnist.validation.images, mnist.validation.labels, step)
      print
      print "Model saved in file: %s" % save_path
      print "Accuracy: %f" % accuracy
