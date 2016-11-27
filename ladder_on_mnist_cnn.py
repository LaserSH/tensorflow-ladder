import random
import input_data_mnist
import ladder_network
import tensorflow as tf

unlabeled_size = 50000 
labeled_size = 5000
mnist_size = 60000

print "Loading MNIST data"
mnist = input_data_mnist.read_data_sets(
    "/zfsauton/home/hanqis/data/mnist",
    one_hot=True,
    flatten=False,
    labeled_size=labeled_size,
    validation_size=mnist_size - unlabeled_size - labeled_size)

print mnist.train_unlabeled.num_examples, "unlabeled training examples"
print mnist.train_labeled.num_examples, "labeled training examples"
print mnist.validation.num_examples, "validation examples"
print mnist.test.num_examples, "test examples"


hyperparameters = {
  "learning_rate": 0.001,
  "noise_level": 0.2,
  "input_layer_size": [28,28,3],
  "class_count": 10,
  "is_cnn": True,
  "encoder_layer_definitions": [
    # kernel_field_size, conv_stride_size, activation, 
    # normalization, max_poolk, max_poolk_stride, group
    # Unlike Alexnet, the group is set to one to avoid 
    # complexity in deconv and denoise
    ([11,11,3,96], [1,2,2,1], tf.nn.relu, True, [1,3,3,1], [1,2,2,1], 1),
    #([11,11,3,96], [1,4,4,1], tf.nn.relu, True, [1,3,3,1], [1,2,2,1], 1),
    ([3,3,96,256], [1,1,1,1], tf.nn.relu, True, [1,3,3,1], [1,2,2,1], 1),
    ([3,3,256,384],[1,1,1,1], tf.nn.relu, False, None, None, 1),
    ([2,2,384,384],[1,1,1,1], tf.nn.relu, False, None, None, 1),
    ([3,3,384,256],[1,1,1,1], tf.nn.relu, False, [1,2,2,1], [1,2,2,1], 1),
    # Flag, fully_connected_kernel_size, activation, 
    # local_response_normalization, _, _, dropout_keep_prob
    (None, [256, 128], tf.nn.relu, False, None, None, 0.5),
    (None, [128, 128], tf.nn.relu, False, None, None, 0.5),
    (None, [128, 10], tf.nn.softmax, False, None, None, 1),
  ],
  "denoising_cost_multipliers": [
    1000, # input layer
    1,
    0.01, 
    0.01,
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
    if step % 5 == 0:
      images, labels = mnist.train_labeled.next_batch(100)
      session.train_supervised_batch(images, labels, step)
    else:
      images, _ = mnist.train_unlabeled.next_batch(100)
      session.train_unsupervised_batch(images, step)

    if step % 200 == 0:
      save_path = session.save()
      accuracy = session.test(
        mnist.validation.images, mnist.validation.labels, step)
      print
      print "Model saved in file: %s" % save_path
      print "Accuracy: %f" % accuracy
