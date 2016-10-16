import numpy as np
import tensorflow as tf

class ModelParams:
  def __init__(self):
    self.SHAPE = [100]
    self.CONNECTIONS = "fully_connected"
    self.INPUT_SHAPE = [2]
    self.OUTPUT_SHAPE = [2]
    self.FLOAT_TYPE = tf.float64
    # TODO: Variable inits
  def __str__(self):
    return str(self.__dict__)
  def __eq__(self, other): 
    return self.__dict__ == other.__dict__
  def __ne__(self, other):
    return not self.__eq__(other)

def tf_dot(vector, matrix, v_shape, m_shape):
  reduction_indices = list(range(len(v_shape), len(m_shape)))
  return tf.reduce_sum(tf.mul(vector, matrix), reduction_indices)

def unbounded_sigmoid(x):
  # x/(1+x^2)^(1/3)
  with tf.name_scope('unbounded_sigmoid') as scope:
    return tf.div(x,tf.pow((1+tf.square(x)),(1/3)))

class GrayMatter(object):
  def __init__(self, model_params):
    self.MP = model_params

    tf.reset_default_graph()
    self.variables = []
    # Graph input
    with tf.name_scope('Placeholders') as scope:
      self.input_placeholder  = tf.placeholder(self.MP.FLOAT_TYPE,
                                               shape=self.MP.INPUT_SHAPE,
                                               name="input")
      self.target_placeholder = tf.placeholder(self.MP.FLOAT_TYPE,
                                               shape=self.MP.INPUT_SHAPE,
                                               name="target")
    # Gray Cells
    if self.MP.CONNECTIONS == "fully_connected":
      self.connection_shape = self.MP.SHAPE + self.MP.SHAPE
      self.input_connection_shape = self.MP.SHAPE + self.MP.INPUT_SHAPE
      self.output_connection_shape = self.MP.OUTPUT_SHAPE + self.MP.SHAPE
    else: raise NotImplementedError
    with tf.variable_scope('CellVariables') as var_scope:
      self.connections = tf.get_variable("W", dtype=self.MP.FLOAT_TYPE,
                                         shape=self.connection_shape,
                                         initializer=tf.contrib.layers.xavier_initializer())
      self.biases      = tf.get_variable("B", dtype=self.MP.FLOAT_TYPE,
                                         shape=self.MP.SHAPE,
                                         initializer=tf.constant_initializer(0))
      self.recentness  = tf.get_variable("R", dtype=self.MP.FLOAT_TYPE,
                                         shape=self.connection_shape,
                                         initializer=tf.constant_initializer(0))
      self.activation  = tf.get_variable("A", dtype=self.MP.FLOAT_TYPE,
                                         shape=self.MP.SHAPE,
                                         initializer=tf.constant_initializer(0))
      self.no_self_connections = tf.constant(1-np.eye(np.prod(self.MP.SHAPE)),
                                             dtype=self.MP.FLOAT_TYPE)

      self.variables.append(self.connections)
      self.variables.append(self.biases)
      self.variables.append(self.recentness)
      self.variables.append(self.activation)
      self.input_connections = tf.get_variable("WI", dtype=self.MP.FLOAT_TYPE,
                                shape=self.input_connection_shape,
                                initializer=tf.contrib.layers.xavier_initializer())
      self.output_connections = tf.get_variable("WO", dtype=self.MP.FLOAT_TYPE,
                                shape=self.output_connection_shape,
                                initializer=tf.contrib.layers.xavier_initializer())
      self.variables.append(self.input_connections)
      self.variables.append(self.output_connections)
    # Step - Part 0: taper off activation spikes
    with tf.name_scope('TaperOff'):
      self.a = self.activation * 0.8
    # Step - Part I: propagate activations
    with tf.name_scope('FireInputCells'):
      self.a = tf.nn.relu(self.a + tf_dot(self.input_placeholder,
                                          self.input_connections,
                                          self.MP.INPUT_SHAPE,
                                          self.input_connection_shape))
    with tf.name_scope('FireCells'):
      self.a = tf.nn.relu(self.a + tf_dot(self.a,
                                          self.connections,
                                          self.MP.SHAPE,
                                          self.connection_shape) - self.biases)
      self.a = self.activation.assign(self.a)
    with tf.name_scope('FireOutputCells'):
      self.output = tf.nn.relu(tf_dot(self.a,
                                      self.output_connections,
                                      self.MP.SHAPE,
                                      self.output_connection_shape))
    # Step - Part II: adjust weights
    with tf.name_scope('EvaluateHappiness'):
      self.happiness = tf.get_variable("Happiness", dtype=self.MP.FLOAT_TYPE,
                                       shape=(), initializer=tf.constant_initializer(0))
      self.prev_happiness = self.happiness * 1.0
      self.new_happiness = tf.reduce_sum(-tf.square(self.target_placeholder - 
                                                    self.output)) + (self.happiness*0.01)
      self.new_happiness = self.happiness.assign(self.new_happiness)
      self.happiness_dt = self.new_happiness - self.prev_happiness
      self.happiness_dt = self.happiness_dt * 1.1 # happiness and disctontent increase if nothing improves

      self.recentness = tf.tanh(self.a)
      self.update_connections = self.connections.assign(((1 - self.recentness) + 
                                                         self.recentness *
                                                         tf.nn.softplus(self.happiness_dt)/np.log(2)) *
                                                        self.connections *
                                                        self.no_self_connections)
      # Sometimes spike random neurons
      # Sometimes change a random connection
    # Initialize session
    self.sess = tf.Session()
    tf.initialize_all_variables().run(session=self.sess)
    # Saver
    variable_names = {}
    for var in self.variables:
      variable_names[var.name] = var
    self.saver = tf.train.Saver(variable_names)

  ## Example functions for different ways to call the autoencoder graph.
  def step(self, input_=None, target=None):
    if input_ is None:
      input_ = np.zeros(self.MP.INPUT_SHAPE)
    if target is None:
      target = np.zeros(self.MP.INPUT_SHAPE)
    return self.sess.run((self.output, self.update_connections),
                         feed_dict={self.input_placeholder: input_,
                                    self.target_placeholder: target})[0]
