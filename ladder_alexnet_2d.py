import tensorflow as tf
import numpy
from utils import *

model_path = '../tensorflow_2d/bvlc_alexnet.npy'
layer_names = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', \
               'fc6', 'fc7', 'fc8']
net_data = numpy.load(model_path).item()

class ForwardAlexnet:
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
          other_layer_definitions = encoder_layer_definitions,
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
    index = 0
    for (layer_configuration, reuse_layer) in zip(
        other_layer_definitions, reuse_variables):
      layer_output = _EncoderLayer(
          inputs = layer_accumulator[-1].post_activation,
          layer_config = layer_configuration,
          noise_level = noise_level,
          is_training_phase = is_training_phase,
          reuse_variables = reuse_layer,
          layer_name = layer_names[index])
      layer_accumulator.append(layer_output)
      index += 1
    return layer_accumulator

  def _decoder_layers(self, clean_encoder_layers, 
      corrupted_encoder_layers, other_layer_definitions, 
      is_training_phase):
    encoder_layers = reversed(zip(clean_encoder_layers, corrupted_encoder_layers))
    layer_accumulator = [None]
    index = 0
    other_layer_definitions.reverse()
    dummy_first_conv_layer = list(other_layer_definitions[-1])
    dummy_first_conv_layer[4] = None
    other_layer_definitions.append(tuple(dummy_first_conv_layer))
    prev_clean_layer = None
    for clean_layer, corrupted_layer in encoder_layers:
      layer = _DecoderLayer(
          clean_encoder_layer = clean_layer,
          corrupted_encoder_layer = corrupted_layer,
          previous_clean_layer = prev_clean_layer,
          previous_decoder_layer = layer_accumulator[-1],
          layer_config = other_layer_definitions,
          is_training_phase = is_training_phase,
          index = index)
      layer_accumulator.append(layer)
      prev_clean_layer = clean_layer
      index += 1
    return layer_accumulator[1:]


class _InputLayerWrapper:
  def __init__(self, input_layer):
    self.input_data = input_layer
    self.pre_activation = input_layer
    self.post_activation = input_layer
    self.batch_mean = tf.zeros_like(input_layer)
    self.batch_std = tf.ones_like(input_layer)


class _EncoderLayer:
  def __init__(self, inputs, layer_config, noise_level, 
      is_training_phase, reuse_variables = None, layer_name = None):
    with tf.name_scope("encoder_layer") as scope:
      self.input_data = inputs
      conv_kernel = layer_config[0]
      # fully connected layers
      if conv_kernel is None:
        weight_size = layer_config[1]
        bias_size = [weight_size[-1]]
        non_linearity = layer_config[2]
        keep_prob = layer_config[-1]
        self._create_or_reuse_variables(reuse_variables, weight_size, bias_size)
        if (len(inputs.get_shape().as_list()) > 2):
          inputs = tensor_flatten(inputs)

        self.pre_normalization = tf.matmul(inputs, self.weights)
        print(str(self.pre_normalization.get_shape()))
        pre_noise, self.batch_mean, self.batch_std = batch_norm(
            self.pre_normalization, is_training_phase = is_training_phase)
        self.pre_activation = self._add_noise(pre_noise, noise_level)
        post_activation = non_linearity(self.pre_activation)
        self.post_activation = tf.nn.dropout(post_activation, keep_prob)

      # convolutional layers
      else:
        weight_size = conv_kernel
        bias_size = [weight_size[-1]]
        conv_stride = layer_config[1]
        non_linearity = layer_config[2]
        # local_normalization is ignored
        # as we always do batch_normalization for each layer
        local_normalization = layer_config[3]
        max_poolk = layer_config[4]
        max_pool_stride = layer_config[5]
        group = layer_config[6]
        self._create_or_reuse_variables(reuse_variables, 
            weight_size, bias_size)
        self.pre_normalization = self._conv(inputs, 
            self.weights, self.bias, conv_stride, group=group)
        pre_noise, self.batch_mean, self.batch_std = batch_norm(
            self.pre_normalization, is_training_phase=is_training_phase)
        self.pre_activation = self._add_noise(pre_noise, noise_level)
        post_activation = non_linearity(self.pre_activation)

        if max_poolk is not None:
          self.post_activation = tf.nn.max_pool(post_activation, ksize=max_poolk, strides=max_pool_stride, padding="VALID")
        else:
          self.post_activation = post_activation


  def _conv(self, inputs, kernel, biases, conv_stride, padding="SAME", group=1):
    convolve = lambda i, k: tf.nn.conv2d(i, k, conv_stride, padding=padding)
    if group == 1:
      conv = convolve(inputs, kernel)
    else:
      input_groups = tf.split(3, group, input)
      kernel_groups = tf.split(3, group, kernel)
      output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_gruops)]
      conv = tf.concat(3, output_groups)
    print(str(inputs.get_shape()) + ', ' +str(kernel.get_shape()) + ',' + str(conv.get_shape()))
    return tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:])

  def _create_or_reuse_variables(self, variables, weight_size, bias_size, name=None):
    if variables is None:
      if name is None:
        self.weights = _weight_variable(weight_size, name = 'W')
        self.bias = _weight_variable(bias_size, name = 'b')
      else:
        self.weights = tf.Variable(net_data[name][0])
        self.bias = tf.Variable(net_data[name][1])
    else:
      self.weights = variables.weights
      self.bias = variables.bias

  def _add_noise(self, tensor, noise_level):
    if noise_level is None:
      return tensor
    else:
      return tensor + tf.random_normal(tf.shape(tensor)) * noise_level


class _DecoderLayer:
  def __init__(self,
      layer_config, clean_encoder_layer, corrupted_encoder_layer,
      previous_clean_layer,
      previous_decoder_layer = None, 
      is_training_phase = True, index = None):
    with tf.name_scope("decoder_layer") as scope:
      is_first_decoder_layer = previous_decoder_layer is None
      if is_first_decoder_layer:
        self.pre_1st_normalization = corrupted_encoder_layer.post_activation
      else:
        previous_config = layer_config[index-1]
        current_config = layer_config[index]
        if previous_config[0] is not None:
          # does deconv here
          pre_deconv = previous_decoder_layer.post_denoising
#         if len(pre_deconv.get_shape().as_list()) < 4:
#           pre_deconv = tf.reshape(pre_deconv, 
#               tf.shape(previous_clean_layer.post_activation))
          self.pre_1st_normalization = self._deconv(
              pre_deconv,
              previous_clean_layer.input_data.get_shape().as_list(),
              previous_config[0],
              previous_config[1])
        else:
          # does fully connected here
          weights = _weight_variable(previous_config[1][::-1], 
              name = 'V')
          self.pre_1st_normalization = tf.matmul(
            previous_decoder_layer.post_denoising, weights)

        if current_config[4] is not None:
          # does depool here
          if(len(self.pre_1st_normalization.get_shape().as_list()) < 4):
            self.pre_1st_normalization = tf.reshape(
                self.pre_1st_normalization,
                tf.shape(clean_encoder_layer.post_activation))

          print('sss' + str(corrupted_encoder_layer.pre_activation.get_shape()))
          self.pre_1st_normalization = self._depool(
              self.pre_1st_normalization,
              corrupted_encoder_layer.pre_activation,
              depool_shape = current_config[4],
              depool_stride = current_config[5])
 
      print(str(self.pre_1st_normalization.get_shape()))
      pre_denoising, _, _ = batch_norm(self.pre_1st_normalization, is_training_phase = is_training_phase)
      post_denoising = self._denoise(
        corrupted_encoder_layer.pre_activation, pre_denoising)
      post_2nd_normalization = \
        (post_denoising - clean_encoder_layer.batch_mean) / clean_encoder_layer.batch_std

      self.post_denoising = post_denoising
      self.post_2nd_normalization = post_2nd_normalization

  def _deconv(self, inputs, shape_after, deconv_shape, deconv_stride):
    #deconv_shape[2], deconv_shape[3] = deconv_shape[3], deconv_shape[2]
    print(str(inputs.get_shape()) + 'd,d' + str(deconv_shape) + str(shape_after))
    deconv_weights = _weight_variable(deconv_shape, name = 'V')
    batch_size = tf.shape(inputs)[0]
    shape_after[0] = batch_size
    shape_after = tf.pack(shape_after)
    return tf.nn.conv2d_transpose(inputs, deconv_weights, shape_after,
        deconv_stride, padding='SAME')

  def _depool(self, inputs, desired_layer, depool_shape, depool_stride):
#   old_shape = tf.shape(inputs)
#   new_height = old_shape[1] * depool_stride[1]
#   new_width = old_shape[2] * depool_stride[2]
#   new_size = tf.pack([new_height, new_width])
    new_height = tf.shape(desired_layer)[1]
    new_width = tf.shape(desired_layer)[2]
    new_size = tf.pack([new_height, new_width])
    return tf.image.resize_images(inputs, new_size)

  def _denoise(self, from_left, from_above):
    #print(str(from_left.get_shape()) + ',' + str(from_above.get_shape()))
    above_shape = tf.shape(from_above)
    from_left = tensor_flatten(from_left)
    from_above = tf.reshape(from_above, tf.shape(from_left))
    with tf.name_scope('mu') as scope:
      mu = self._modulate(from_above)
    with tf.name_scope('v') as scope:
      v = self._modulate(from_above)
    post_denoise = (from_left - mu) * v + mu
    return tf.reshape(post_denoise, above_shape)

  def _modulate(self, u):
    a = [_weight_variable([_layer_size(u)], name = str(i)) for i in xrange(5)]
    return a[0] * tf.nn.sigmoid(a[1] * u + a[2]) + a[3] * u + a[4]


def _weight_variable(shape, name = 'weight'):
  if shape is None:
    return None
  initial = tf.truncated_normal(shape, stddev = 0.1)
  return tf.Variable(initial, name = name)

def _layer_size(layer_output):
  return layer_output.get_shape()[-1].value


