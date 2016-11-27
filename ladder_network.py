import tensorflow as tf
import numpy
from tensorflow.python.ops import control_flow_ops
from utils import *
import ladder_alexnet_2d
from time import strftime


class Session:
  def __init__(self, graph):
    self.session = tf.Session()
    self.graph = graph
    self.writer = tf.train.SummaryWriter(
        logdir = strftime("logs/%Y-%m-%d_%H-%M-%S"),
        graph = self.session.graph)

  def __enter__(self):
    self.session.run(tf.initialize_all_variables())
    return self

  def __exit__(self, type, value, traceback):
    self.session.close()

  def train_supervised_batch(self, inputs, labels, step_number):
    return self._run(self.graph.supervised_train_step,
        summary_action = self.graph.supervised_summaries,
        step_number = step_number,
        inputs = inputs,
        labels = labels,
        is_training_phase = True)

  def train_unsupervised_batch(self, inputs, step_number):
    return self._run(self.graph.unsupervised_train_step,
        summary_action = self.graph.unsupervised_summaries,
        step_number = step_number,
        inputs = inputs,
        is_training_phase = True)

  def test(self, inputs, labels, step_number):
    result = self._run(self.graph.accuracy_measure,
        summary_action = self.graph.test_summaries,
        step_number = step_number,
        inputs = inputs,
        labels = labels,
        is_training_phase = False)
    self.writer.flush()
    return result

  def save(self):
    return self.graph.saver.save(self.session, "checkpoints")

  def _run(self, action, summary_action, step_number, inputs, labels = None, is_training_phase = True):
    variable_placements = self.graph.placeholders.placements(
        inputs, labels, is_training_phase)
    action_result, summary = self.session.run(
        [action, summary_action], variable_placements)
    self.writer.add_summary(summary, step_number)
    return action_result


class Graph:
  def __init__(self,
      learning_rate,
      noise_level,
      input_layer_size,
      class_count,
      is_cnn,
      encoder_layer_definitions,
      denoising_cost_multipliers):
    # assert class_count == encoder_layer_definitions[-1][0]

    self.learning_rate = learning_rate
    self.denoising_cost_multipliers = denoising_cost_multipliers

    self.placeholders = _Placeholders(input_layer_size, class_count)

    if is_cnn:
      self.output = ladder_alexnet_2d.ForwardAlexnet(
          self.placeholders,
          noise_level=noise_level,
          encoder_layer_definitions=encoder_layer_definitions)
    else:
      self.output = _ForwardPass(self.placeholders,
        noise_level=noise_level,
        encoder_layer_definitions=encoder_layer_definitions)

    self.accuracy_measure = self._accuracy_measure(
        self.placeholders, self.output)
    self.supervised_train_step = self._supervised_train_step(
        self.placeholders, self.output)
    self.unsupervised_train_step = self._unsupervised_train_step(
        self.placeholders, self.output)

    self.unsupervised_summaries = tf.merge_all_summaries("unsupervised")
    self.supervised_summaries = tf.merge_all_summaries("supervised")
    self.test_summaries = tf.merge_all_summaries("test")

    self.saver = tf.train.Saver()

  def _accuracy_measure(self, placeholders, output):
    with tf.name_scope("accuracy_measure") as scope:
      actual_labels = tf.argmax(output.clean_label_probabilities, 1)
      expected_labels = tf.argmax(placeholders.labels, 1)

      correct_prediction = tf.equal(actual_labels, expected_labels)
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
      tf.histogram_summary("class distribution", actual_labels, ["test"])
      tf.scalar_summary("test accuracy", accuracy, ["test"])
      return accuracy

  def _supervised_train_step(self, placeholders, output):
    with tf.name_scope("supervised_training") as scope:
      total_cost = self._total_cost(placeholders, output)
      return self._optimizer(self.learning_rate, total_cost, ["supervised"])

  def _unsupervised_train_step(self, placeholders, output):
    with tf.name_scope("unsupervised_training") as scope:
      summary_tags = ["unsupervised"]
      total_denoising_cost, layer_denoising_costs = self._total_denoising_cost(
        placeholders, output)
      tf.scalar_summary("total denoising cost", total_denoising_cost, summary_tags)
      for index, layer_cost in enumerate(layer_denoising_costs):
        tf.scalar_summary("layer %i denoising cost" % index, layer_cost, summary_tags)
      return self._optimizer(
          self.learning_rate, total_denoising_cost, summary_tags)

  def _optimizer(self, learning_rate, cost_function, summary_tags):
    with tf.name_scope("optimizer") as scope:
      optimizer = tf.train.AdamOptimizer(learning_rate)
      gradients_and_vars = optimizer.compute_gradients(cost_function)
      for (gradient, var) in gradients_and_vars:
        if gradient is not None:
          tf.histogram_summary("gradient for %s" % var.name, gradient, summary_tags)
      return optimizer.apply_gradients(gradients_and_vars)

  def _total_cost(self, placeholders, output):
    with tf.name_scope("total_cost") as scope:
      cross_entropy = self._cross_entropy(placeholders, output)
      total_denoising_cost, layer_denoising_costs = self._total_denoising_cost(
          placeholders, output)
      total_cost = cross_entropy + total_denoising_cost

      self._log_all_costs(total_cost, cross_entropy, total_denoising_cost,
        layer_denoising_costs, ["supervised"])

      return total_cost

  def _log_all_costs(self, total_cost = None, cross_entropy = None,
        total_denoising_cost = None, layer_denoising_costs = None,
        summary_tags = tf.GraphKeys.SUMMARIES):
      tf.scalar_summary("total cost", total_cost, summary_tags)

      tf.scalar_summary("cross entropy", cross_entropy, summary_tags)
      tf.scalar_summary("cross entropy %", 100 * cross_entropy / total_cost, summary_tags)

      tf.scalar_summary("total denoising cost", total_denoising_cost, summary_tags)
      tf.scalar_summary("total denoising cost %", 100 * total_denoising_cost / total_cost, summary_tags)

      for index, layer_cost in enumerate(layer_denoising_costs):
        tf.scalar_summary("layer %i denoising cost" % index, layer_cost, summary_tags)
        tf.scalar_summary("layer %i denoising cost %%" % index, 100 * layer_cost / total_cost, summary_tags)


  def _cross_entropy(self, placeholders, output):
    with tf.name_scope("cross_entropy_cost") as scope:
      cross_entropy = -tf.reduce_mean(
          placeholders.labels * tf.log(output.corrupted_label_probabilities))
      return cross_entropy

  def _total_denoising_cost(self, placeholders, output):
    with tf.name_scope("denoising_cost") as scope:
      layer_costs = [self._layer_denoising_cost(*params) for params in zip(
          output.clean_encoder_outputs,
          reversed(output.decoder_outputs),
          self.denoising_cost_multipliers)]
      total_denoising_cost = sum(layer_costs)

      return total_denoising_cost, layer_costs

  def _layer_denoising_cost(self, encoder, decoder, cost_multiplier): 
    return cost_multiplier * self._mean_squared_error(
        tf.reshape(encoder.pre_activation, [-1]),
        tf.reshape(decoder.post_2nd_normalization, [-1]))

  def _mean_squared_error(self, expected, actual):
    return tf.reduce_mean(tf.pow(expected - actual, 2))

class _Placeholders:
  def __init__(self, input_layer_size, class_count):
    with tf.name_scope("placeholders") as scope:
      self.inputs = tf.placeholder(tf.float32, 
                                   shape=([None,] + input_layer_size),
                                   name = 'inputs')
      self.labels = tf.placeholder(tf.float32, 
                                   shape=([None,] + [class_count]),
                                   name = 'labels')
      self.is_training_phase = tf.placeholder(tf.bool, name = 'is_training_phase')

  def placements(self, inputs, labels = None, is_training_phase = True):
    if labels is None:
      labels = numpy.zeros([inputs.shape[0], _layer_size(self.labels)])
    return {
      self.inputs: inputs,
      self.labels: labels,
      self.is_training_phase: is_training_phase
    }


class _ForwardPass:
  def __init__(self, placeholders, encoder_layer_definitions, noise_level):
    with tf.name_scope("clean_encoder") as scope:
      clean_encoder_outputs = self._encoder_layers(
          input_layer = placeholders.inputs,
          other_layer_definitions = encoder_layer_definitions,
          is_training_phase = placeholders.is_training_phase)

    with tf.name_scope("corrupted_encoder") as scope:
      corrupted_encoder_outputs = self._encoder_layers(
          input_layer = placeholders.inputs,
          other_layer_definitions = encoder_layer_definitions,
          is_training_phase = placeholders.is_training_phase,
          noise_level = noise_level,
          reuse_variables = clean_encoder_outputs[1:])

    with tf.name_scope("decoder") as scope:
      decoder_outputs = self._decoder_layers(
          clean_encoder_layers = clean_encoder_outputs,
          corrupted_encoder_layers = corrupted_encoder_outputs,
          is_training_phase = placeholders.is_training_phase)

    self.clean_label_probabilities = clean_encoder_outputs[-1].post_activation
    self.corrupted_label_probabilities = corrupted_encoder_outputs[-1].post_activation
    self.autoencoded_inputs = decoder_outputs[-1]
    self.clean_encoder_outputs = clean_encoder_outputs
    self.corrupted_encoder_outputs = corrupted_encoder_outputs
    self.decoder_outputs = decoder_outputs

  def _encoder_layers(self, input_layer, other_layer_definitions,
      noise_level = None, is_training_phase = True, reuse_variables = None):
    first_encoder_layer = _InputLayerWrapper(input_layer)
    if reuse_variables is None:
      reuse_variables = [None for layer in other_layer_definitions]

    layer_accumulator = [first_encoder_layer]
    for ((layer_size, non_linearity), reuse_layer) in zip(
        other_layer_definitions, reuse_variables):
      layer_output = _EncoderLayer(
          inputs = layer_accumulator[-1].post_activation,
          output_size = layer_size,
          non_linearity = non_linearity,
          noise_level = noise_level,
          is_training_phase = is_training_phase,
          reuse_variables = reuse_layer)
      layer_accumulator.append(layer_output)
    return layer_accumulator

  def _decoder_layers(self, clean_encoder_layers, corrupted_encoder_layers,
        is_training_phase):
    encoder_layers = reversed(zip(clean_encoder_layers, corrupted_encoder_layers))
    layer_accumulator = [None]
    for clean_layer, corrupted_layer in encoder_layers:
      layer = _DecoderLayer(
          clean_encoder_layer = clean_layer,
          corrupted_encoder_layer = corrupted_layer,
          previous_decoder_layer = layer_accumulator[-1],
          is_training_phase = is_training_phase)
      layer_accumulator.append(layer)
    return layer_accumulator[1:]


class _InputLayerWrapper:
  def __init__(self, input_layer):
    self.pre_activation = input_layer
    self.post_activation = input_layer
    self.batch_mean = tf.zeros_like(input_layer)
    self.batch_std = tf.ones_like(input_layer)


class _EncoderLayer:
  def __init__(self, inputs, output_size, non_linearity,
      noise_level, is_training_phase, reuse_variables = None):
    with tf.name_scope("encoder_layer") as scope:
      self._create_or_reuse_variables(reuse_variables, _layer_size(inputs), output_size)

      self.pre_normalization = tf.matmul(inputs, self.weights)
      pre_noise, self.batch_mean, self.batch_std = batch_norm(
          self.pre_normalization, is_training_phase = is_training_phase)
      self.pre_activation = self._add_noise(pre_noise, noise_level)
      beta_gamma = self.gamma * (self.pre_activation + self.beta)
      self.post_activation = non_linearity(beta_gamma)

  def _create_or_reuse_variables(self, variables, input_size, output_size):
    if variables is None:
      self.weights = _weight_variable([input_size, output_size], name = 'W')
      self.beta = tf.Variable(tf.constant(0.0, shape = [output_size]), name = 'beta')
      self.gamma = tf.Variable(tf.constant(1.0, shape = [output_size]), name = 'gamma')
    else:
      self.weights = variables.weights
      self.beta = variables.beta
      self.gamma = variables.gamma

  def _add_noise(self, tensor, noise_level):
    if noise_level is None:
      return tensor
    else:
      return tensor + tf.random_normal(
          [_layer_size(tensor)], mean = 0.0, stddev = noise_level)


class _DecoderLayer:
  def __init__(self,
      clean_encoder_layer, corrupted_encoder_layer,
      previous_decoder_layer = None, is_training_phase = True):
    with tf.name_scope("decoder_layer") as scope:
      is_first_decoder_layer = previous_decoder_layer is None
      if is_first_decoder_layer:
        pre_1st_normalization = corrupted_encoder_layer.post_activation
      else:
        input_size = _layer_size(previous_decoder_layer.post_denoising)
        output_size = _layer_size(clean_encoder_layer.post_activation)
        weights = _weight_variable([input_size, output_size], name = 'V')
        pre_1st_normalization = tf.matmul(
          previous_decoder_layer.post_denoising, weights)

      pre_denoising, _, _ = batch_norm(pre_1st_normalization, is_training_phase = is_training_phase)
      post_denoising = self._denoise(
        corrupted_encoder_layer.pre_activation, pre_denoising)
      post_2nd_normalization = \
        (post_denoising - clean_encoder_layer.batch_mean) / clean_encoder_layer.batch_std

      self.post_denoising = post_denoising
      self.post_2nd_normalization = post_2nd_normalization

  def _denoise(self, from_left, from_above):
    with tf.name_scope('mu') as scope:
      mu = self._modulate(from_above)
    with tf.name_scope('v') as scope:
      v = self._modulate(from_above)
    return (from_left - mu) * v + mu

  def _modulate(self, u):
    a = [_weight_variable([_layer_size(u)], name = str(i)) for i in xrange(5)]
    return a[0] * tf.nn.sigmoid(a[1] * u + a[2]) + a[3] * u + a[4]


def _weight_variable(shape, name = 'weight'):
  initial = tf.truncated_normal(shape, stddev = 0.1)
  return tf.Variable(initial, name = name)

def _layer_size(layer_output):
  return layer_output.get_shape()[-1].value


