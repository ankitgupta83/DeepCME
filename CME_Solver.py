import time
import numpy as np
import tensorflow as tf
import data_saving as dts


class CMESolver(object):
    """The fully connected neural network model."""

    def __init__(self, network, config_data):
        self.network = network
        self.config_data = config_data
        lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            config_data['net_config']['lr_boundaries'], config_data['net_config']['lr_values'])
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=1e-8)
        self.total_num_simulated_trajectories = self.config_data['net_config']['valid_size'] \
                                                + self.config_data['net_config']['batch_size']
        # create data for training and validation..unless it is already available
        if config_data['net_config']['training_samples_needed'] == "True":
            start_time = time.time()
            self.training_data = self.network.generate_sampled_rtc_trajectories(
                self.config_data['reaction_network_config']['final_time'],
                self.config_data['reaction_network_config']['num_time_interval'],
                self.config_data['net_config']['batch_size'])
            print("Time needed to generate training trajectories: %3u" % (time.time() - start_time))
            dts.save_sampled_trajectories(config_data['reaction_network_config']['output_folder'] + "/",
                                          self.training_data, sample_type="training")
        else:
            self.training_data = dts.load_save_sampled_trajectories(config_data['reaction_network_config']['output_folder']
                                                                    + "/", sample_type="training")
        if config_data['net_config']['validation_samples_needed'] == "True":
            start_time = time.time()
            self.valid_data = self.network.generate_sampled_rtc_trajectories(
                self.config_data['reaction_network_config']['final_time'],
                self.config_data['reaction_network_config']['num_time_interval'],
                self.config_data['net_config']['valid_size'])
            self.total_num_simulated_trajectories = self.config_data['net_config']['valid_size'] + \
                                                    self.config_data['net_config']['batch_size']
            print("Time needed to generate validation trajectories: %3u" % (time.time() - start_time))
            dts.save_sampled_trajectories(config_data['reaction_network_config']['output_folder'] + "/", self.training_data,
                                          sample_type="validation")
        else:
            self.valid_data = dts.load_save_sampled_trajectories(config_data['reaction_network_config']['output_folder']
                                                                 + "/", sample_type="validation")

        # set initial values for functions
        times, states_trajectories, martingale_trajectories = self.training_data
        yvals = self.network.output_function(states_trajectories[:, -1, :])
        y0 = tf.reduce_mean(yvals, axis=0)
        # set func_clipping_thresholds
        self.delta_clip = np.ones(shape=[self.network.output_function_size], dtype="float64") + \
                          tf.math.reduce_mean(yvals, axis=0) + 2 * tf.math.reduce_std(yvals, axis=0)
        self.model = NonsharedModel(network, config_data, y0, self.delta_clip)
        if config_data['net_config']['use_previous_training_weights'] == "True":
            filename = config_data['reaction_network_config']['output_folder'] + "/" + "trained_weights"
            self.model.load_weights(filename)
        self.y_init = self.model.y_init

    def train(self):
        start_time = time.time()
        training_history = []
        function_value_data = []
        num_iterations = self.config_data['net_config']['num_iterations']
        # num_batch_size = self.config_data['net_config']['batch_size']
        logging_frequency = self.config_data['net_config']['logging_frequency']
        # training_data_reset_frequency = self.config_data['net_config']['training_data_reset_frequency']

        # begin sgd iteration
        for step in range(1, num_iterations + 1):
            self.train_step(self.training_data)
            if step % logging_frequency == 0:
                loss = self.loss_fn(self.valid_data, training=False).numpy()
                y_init = self.y_init.numpy()
                elapsed_time = time.time() - start_time
                training_history.append([step, loss, elapsed_time])
                function_value_data.append(y_init)
                print("step: %5u, loss: %.4e, elapsed time: %3u" % (
                    step, loss, elapsed_time))
                print_array_nicely(y_init, "Estimated Value")
        return np.array(training_history), np.array(function_value_data), self.total_num_simulated_trajectories

    def loss_fn(self, inputs, training):
        times, states_trajectories, martingale_trajectories = inputs
        y_terminal = self.model(inputs, training)
        y_comp = self.network.output_function(states_trajectories[:, -1, :])
        delta = (y_terminal - y_comp) / self.delta_clip
        loss = tf.reduce_mean(tf.where(tf.abs(delta) < 1, tf.square(delta), 2 * tf.abs(delta) - 1), axis=0)
        return tf.reduce_sum(loss)

    def grad(self, inputs, training):
        with tf.GradientTape(persistent=True) as tape:
            loss = self.loss_fn(inputs, training)
        grad = tape.gradient(loss, self.model.trainable_variables)
        del tape
        return grad

    @tf.function
    def train_step(self, train_data):
        grad = self.grad(train_data, training=True)
        self.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))

    def estimate_parameter_sensitivities(self):
        times, states_trajectories, martingale_trajectories = self.training_data
        return self.model.compute_parameter_jacobian(states_trajectories, times, len(self.network.parameter_dict),
                                                     training=False)


class NonsharedModel(tf.keras.Model):

    def __init__(self, network, config_data, y0, delta_clip):
        super(NonsharedModel, self).__init__()
        self.network = network
        self.delta_clip = delta_clip
        self.stop_time = config_data['reaction_network_config']['final_time']
        self.num_exponential_features = config_data['net_config']['num_exponential_features']
        self.num_temporal_dnns = config_data['net_config']['num_temporal_dnns']
        self.num_time_samples = config_data['reaction_network_config']['num_time_interval']
        self.y_init = tf.Variable(y0)
        self.eigval_real = tf.Variable(-np.random.uniform(0, 1, size=[1, self.num_exponential_features]),
                                       dtype="float64")
        self.eigval_imag = tf.Variable(np.zeros([1, self.num_exponential_features], dtype="float64"))
        self.eigval_phase = tf.Variable(np.zeros([1, self.num_exponential_features], dtype="float64"))
        self.subnet = [FeedForwardSubNet(self.network.num_reactions, self.network.output_function_size, config_data)
                       for _ in range(self.num_temporal_dnns)]

    def call(self, inputs, training):
        times, states_trajectories, martingale_trajectories = inputs
        batch_size = tf.shape(martingale_trajectories)[0]
        all_one_vec = tf.ones(shape=tf.stack([batch_size, 1]), dtype="float64")
        y = tf.matmul(all_one_vec, self.y_init[None, :])
        for t in range(0, self.num_time_samples - 1):
            time_left = self.stop_time - times[t]
            temporal_dnn = int(t * self.num_temporal_dnns / self.num_time_samples)
            features_real = tf.reshape(tf.tile(tf.exp(self.eigval_real * time_left), [batch_size, 1]),
                                       [batch_size, self.num_exponential_features])
            features_imag = tf.reshape(
                tf.tile(tf.sin(self.eigval_imag * time_left + self.eigval_phase), [batch_size, 1]),
                [batch_size, self.num_exponential_features])
            inputs = tf.stack(tf.unstack(states_trajectories[:, t, :], axis=-1)
                              + tf.unstack(features_real, axis=-1) + tf.unstack(features_imag, axis=-1), axis=1)
            z = self.subnet[temporal_dnn](inputs, training)
            z = tf.reshape(z, shape=[batch_size, self.network.output_function_size,
                                     self.network.num_reactions])
            martingale_increment = tf.expand_dims(martingale_trajectories[:, t + 1, :]
                                                  - martingale_trajectories[:, t, :], axis=1)
            y = y + tf.reduce_sum(z * martingale_increment, axis=2)
        return y

    def compute_parameter_jacobian(self, states_trajectories, times, num_params, training):
        batch_size = tf.shape(states_trajectories)[0]
        jacobian = tf.zeros(shape=tf.stack([batch_size, num_params, self.network.output_function_size]),
                            dtype="float64")
        for t in range(0, self.num_time_samples - 1):
            time_left = self.stop_time - times[t]
            temporal_dnn = int(t * self.num_temporal_dnns / self.num_time_samples)
            features_real = tf.reshape(tf.tile(tf.exp(self.eigval_real * time_left), [batch_size, 1]),
                                       [batch_size, self.num_exponential_features])
            features_imag = tf.reshape(
                tf.tile(tf.sin(self.eigval_imag * time_left + self.eigval_phase), [batch_size, 1]),
                [batch_size, self.num_exponential_features])

            inputs = tf.stack(tf.unstack(states_trajectories[:, t, :], axis=-1)
                              + tf.unstack(features_real, axis=-1) + tf.unstack(features_imag, axis=-1), axis=1)
            z = self.subnet[temporal_dnn](inputs, training)
            z = tf.reshape(z, shape=[batch_size, self.network.output_function_size,
                                     self.network.num_reactions])
            propensity_jacobian = tf.stack([self.network.propensity_sensitivity_matrix(states_trajectories[i, t, :])
                                            for i in range(states_trajectories[:, t, :].shape[0])], axis=0)
            jacobian = jacobian + tf.matmul(propensity_jacobian, z, transpose_b=True) * (times[t + 1] - times[t])
        return tf.reduce_mean(jacobian, axis=0)


class FeedForwardSubNet(tf.keras.Model):
    def get_config(self):
        pass

    def __init__(self, num_reactions, output_function_size, config_data):
        super(FeedForwardSubNet, self).__init__()
        num_hiddens = config_data['net_config']['num_nodes_per_layer']
        num_layers = config_data['net_config']['num_hidden_layers']
        self.dense_layers = [tf.keras.layers.Dense(num_hiddens,
                                                   use_bias=True,
                                                   activation=None,
                                                   kernel_initializer='zeros',
                                                   bias_initializer='zeros')
                             for _ in range(num_layers)]
        # final output should be a value of dimension num_reactions*output_function_size
        self.dense_layers.append(tf.keras.layers.Dense(num_reactions * output_function_size,
                                                       use_bias=True, activation=None,
                                                       kernel_initializer='zeros',
                                                       bias_initializer='zeros'))

    def call(self, x, training):
        for i in range(len(self.dense_layers) - 1):
            x = self.dense_layers[i](x)
            x = tf.nn.relu(x)
        x = self.dense_layers[-1](x)
        return x


def print_array_nicely(y, name):
    size_input = y.size
    y = y.reshape(size_input, )
    print(name, ":", end=' (')
    for i in range(y.size - 1):
        print("%.3f" % y[i], end=', ')
    print("%.3f" % y[y.size - 1], end=')\n')
