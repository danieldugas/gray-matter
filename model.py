import numpy as np
import tensorflow as tf

class ModelParams:
  def __init__(self):
    self.SHAPE = [100]
    self.CONNECTIONS = "fully_connected"
    self.INPUT_SHAPE = [1]
    self.OUTPUT_SHAPE = [1]
    self.FLOAT_TYPE = tf.float64
    # TODO: Variable inits
  def __str__(self):
    return str(self.__dict__)
  def __eq__(self, other): 
    return self.__dict__ == other.__dict__
  def __ne__(self, other):
    return not self.__eq__(other)

class GrayMatter(object):
  def __init__(self, model_params):
    self.MP = model_params

    tf.reset_default_graph()
    self.variables = []
    # Graph input
    with tf.name_scope('Placeholders') as scope:
      self.input_placeholder = tf.placeholder(self.MP.FLOAT_TYPE,
                                              shape=self.MP.INPUT_SHAPE,
                                              name="input")
    # Gray Cells
    if self.CONNECTIONS == "fully_connected":
      self.connection_shape = self.SHAPE + self.SHAPE
      self.input_connection_shape = self.INPUT_SHAPE + self.SHAPE
      self.output_connection_shape = self.SHAPE + self.OUTPUT_SHAPE
    else: raise NotImplementedError
    with tf.variable_scope('CellVariables') as var_scope:
      connections = tf.get_variable("W", dtype=self.MP.FLOAT_TYPE,
                                shape=self.connection_shape,
                                initializer=tf.contrib.layers.xavier_initializer())
      biases      = tf.get_variable("B", dtype=self.MP.FLOAT_TYPE,
                                shape=self.SHAPE,
                                initializer=tf.constant_initializer(0))
      recentness  = tf.get_variable("R", dtype=self.MP.FLOAT_TYPE,
                                   shape=self.connection_shape,
                                   initializer=tf.constant_initializer(0))
      activation  = tf.get_variable("A", dtype=self.MP.FLOAT_TYPE,
                                shape=self.SHAPE,
                                initializer=tf.constant_initializer(0))

      self.variables.append(connections)
      self.variables.append(biases)
      self.variables.append(recentness)
      self.variables.append(activation)
      input_connections = tf.get_variable("WI", dtype=self.MP.FLOAT_TYPE,
                                shape=self.input_connection_shape,
                                initializer=tf.contrib.layers.xavier_initializer())
      output_connections = tf.get_variable("WO", dtype=self.MP.FLOAT_TYPE,
                                shape=self.output_connection_shape,
                                initializer=tf.contrib.layers.xavier_initializer())
      self.variables.append(input_connections)
      self.variables.append(output_connections)
    # Step logic
    with tf.name_scope('FireInputCells')
      activation = activation + tf.matmul(input_placeholder, input_connections)
    with tf.name_scope('FireCells')
      activation = activation + tf.matmul(activation, connections)
    with tf.name_scope('FireOutputCells')
      output = tf.matmul(activation, output_connections)
    # Initialize session
    self.sess = tf.Session()
    tf.initialize_all_variables().run(session=self.sess)
    # Saver
    variable_names = {}
    for var in self.variables:
      variable_names[var.name] = var
    self.saver = tf.train.Saver(variable_names)

  ## Example functions for different ways to call the autoencoder graph.
  def step(self, input_=None):
    if input_ is None:
      input_ = np.zeros(self.MP.INPUT_SHAPE)
    return self.sess.run(self.output,
                         feed_dict={self.input_placeholder: input_})
