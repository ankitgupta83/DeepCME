import numpy as np
import math
import random


def encode(index, sizes):
    n = np.size(sizes)
    if n == 1:
        return index[0]
    else:
        return index[n - 1] + sizes[n - 1] * encode(index[0:n - 1], sizes[0:n - 1])


def generate_random_initial_state(lambda_array):
    return np.random.poisson(lambda_array)


class ReactionNetworkDefinition(object):
    def __init__(self, num_species, num_reactions, reactant_matrix, product_matrix,
                 parameter_dict, reaction_dict, species_labels, output_species_labels):
        # public attributes:
        self.num_species = num_species
        self.num_reactions = num_reactions
        # reactant matrices rows represent number of molecules of each species consumed in that reaction.
        self.reactant_matrix = reactant_matrix
        # product matrices rows represent number of molecules of species produced in that reaction.
        self.product_matrix = product_matrix
        self.stoichiometry_matrix = product_matrix - reactant_matrix
        # contains information about the parameters
        self.parameter_dict = parameter_dict
        self.reaction_dict = reaction_dict
        self.species_labels = species_labels
        self.output_species_labels = output_species_labels
        self.output_species_indices = [self.species_labels.index(i)
                                       for i in self.output_species_labels]

        self.output_function_size = None

    def mass_action_propensity(self, state, reaction_no, rate_constant_key):
        prop = self.parameter_dict[rate_constant_key]
        for j in range(self.num_species):
            for k in range(self.reactant_matrix[reaction_no][j]):  # check order of indices
                prop *= float(state[j] - k)
            prop = prop / math.factorial(self.reactant_matrix[reaction_no][j])
        return prop

    def mass_action_propensity_derivative(self, state, reaction_no, rate_constant_key, param_names):
        propensity_derivatives = np.zeros([len(param_names)])
        prop = 1
        for j in range(self.num_species):
            for k in range(self.reactant_matrix[reaction_no][j]):  # check order of indices
                prop *= float(state[j] - k)
            prop = prop / math.factorial(self.reactant_matrix[reaction_no][j])
        propensity_derivatives[param_names.index(rate_constant_key)] = prop
        return propensity_derivatives

    def hill_propensity_activation(self, state, species_no, parameter_key1, parameter_key2, parameter_key3,
                                   parameter_key4):
        # implements propensity b + a*x^h/(k + x^h) with x=X[species_no]
        a = self.parameter_dict[parameter_key1]
        k = self.parameter_dict[parameter_key2]
        h = self.parameter_dict[parameter_key3]
        if parameter_key4 is not None:
            b = self.parameter_dict[parameter_key4]
        else:
            b = 0.0
        xp = float(state[species_no])
        return b + a * (xp ** h) / (k + (xp ** h))

    def hill_propensity_repression(self, state, species_no, parameter_key1, parameter_key2, parameter_key3,
                                   parameter_key4):
        # implements propensity b + a/(k + x^h) with x=X[species_no]
        a = self.parameter_dict[parameter_key1]
        k = self.parameter_dict[parameter_key2]
        h = self.parameter_dict[parameter_key3]
        if parameter_key4 is not None:
            b = self.parameter_dict[parameter_key4]
        else:
            b = 0.0
        xp = float(state[species_no])
        return b + a / (k + (xp ** h))

    def hill_propensity_activation_derivative(self, state, species_no, parameter_key1, parameter_key2, parameter_key3,
                                              parameter_key4, param_names):
        a = self.parameter_dict[parameter_key1]
        k = self.parameter_dict[parameter_key2]
        h = self.parameter_dict[parameter_key3]
        xp = float(state[species_no])
        den = k + (xp ** h)
        propensity_derivatives = np.zeros([len(param_names)])
        propensity_derivatives[param_names.index(parameter_key1)] = a * (xp ** h) / den
        propensity_derivatives[param_names.index(parameter_key2)] = -a * (xp ** h) / (den ** 2)
        if parameter_key4 is not None:
            b = self.parameter_dict[parameter_key4]
            propensity_derivatives[param_names.index(parameter_key4)] = 1
        else:
            b = 0.0
        if xp > 0:
            propensity_derivatives[param_names.index(parameter_key3)] = -(a * k * (xp ** h) * math.log(xp)) / (den ** 2)
        return propensity_derivatives

    def hill_propensity_repression_derivative(self, state, species_no, parameter_key1, parameter_key2, parameter_key3,
                                              parameter_key4, param_names):
        a = self.parameter_dict[parameter_key1]
        k = self.parameter_dict[parameter_key2]
        h = self.parameter_dict[parameter_key3]
        xp = float(state[species_no])
        den = k + (xp ** h)
        propensity_derivatives = np.zeros([len(param_names)])
        propensity_derivatives[param_names.index(parameter_key1)] = 1 / den
        propensity_derivatives[param_names.index(parameter_key2)] = -a / (den ** 2)
        if parameter_key4 is not None:
            b = self.parameter_dict[parameter_key4]
            propensity_derivatives[param_names.index(parameter_key4)] = 1
        else:
            b = 0.0
        if xp > 0:
            propensity_derivatives[param_names.index(parameter_key3)] = -(a * (xp ** h) * math.log(xp)) / (den ** 2)
        return propensity_derivatives

    def propensity_vector(self, state):
        raise NotImplementedError

    def set_propensity_vector(self):
        def func(state_current):
            prop = np.zeros(self.num_reactions)
            for k in range(self.num_reactions):
                reaction_type = self.reaction_dict[k][0]
                if reaction_type == 'mass action':
                    prop[k] = self.mass_action_propensity(state_current, k, self.reaction_dict[k][1])
                elif reaction_type == 'Hill_activation':
                    prop[k] = self.hill_propensity_activation(state_current, self.reaction_dict[k][1],
                                                              self.reaction_dict[k][2], self.reaction_dict[k][3],
                                                              self.reaction_dict[k][4], self.reaction_dict[k][5])
                elif reaction_type == 'Hill_repression':
                    prop[k] = self.hill_propensity_repression(state_current, self.reaction_dict[k][1],
                                                              self.reaction_dict[k][2], self.reaction_dict[k][3],
                                                              self.reaction_dict[k][4], self.reaction_dict[k][5])
                else:
                    raise NotImplementedError
            return prop

        self.propensity_vector = func

    def propensity_sensitivity_matrix(self, state):
        raise NotImplementedError

    def set_propensity_sensitivity_matrix(self):
        def func(state_current):
            param_names = list(self.parameter_dict.keys())
            propensity_jacobian = np.zeros([len(param_names), self.num_reactions])
            for k in range(self.num_reactions):
                reaction_type = self.reaction_dict[k][0]
                if reaction_type == 'mass action':
                    propensity_jacobian[:, k] = self.mass_action_propensity_derivative(state_current, k,
                                                                                       self.reaction_dict[k][1],
                                                                                       param_names)
                elif reaction_type == 'Hill_activation':
                    propensity_jacobian[:, k] = self.hill_propensity_activation_derivative(state_current,
                                                                                           self.reaction_dict[k][1],
                                                                                           self.reaction_dict[k][2],
                                                                                           self.reaction_dict[k][3],
                                                                                           self.reaction_dict[k][4],
                                                                                           self.reaction_dict[k][5],
                                                                                           param_names
                                                                                           )
                elif reaction_type == 'Hill_repression':
                    propensity_jacobian[:, k] = self.hill_propensity_repression_derivative(state_current,
                                                                                           self.reaction_dict[k][1],
                                                                                           self.reaction_dict[k][2],
                                                                                           self.reaction_dict[k][3],
                                                                                           self.reaction_dict[k][4],
                                                                                           self.reaction_dict[k][5],
                                                                                           param_names
                                                                                           )
                else:
                    raise NotImplementedError
            return propensity_jacobian

        self.propensity_sensitivity_matrix = func

    def output_function(self, state):
        raise NotImplementedError

    def gillespie_ssa_next_reaction(self, state):
        prop = self.propensity_vector(state)
        sum_prop = np.sum(prop)
        if sum_prop == 0:
            delta_t = math.inf
            next_reaction = -1
        else:
            prop = np.cumsum(np.divide(prop, sum_prop))
            delta_t = -math.log(np.random.uniform(0, 1)) / sum_prop
            next_reaction = sum(prop < np.random.uniform(0, 1))
        return delta_t, next_reaction

    def update_state(self, next_reaction, state):
        if next_reaction != -1:
            state = state + self.stoichiometry_matrix[next_reaction, :]
        return state

    def run_gillespie_ssa(self, initial_state, stop_time):
        """
        Runs Gillespie's SSA without storing any values until stop_time; start time is 0 and
        initial_state is specified
        """
        t = 0
        state_curr = initial_state
        while 1:
            delta_t, next_reaction = self.gillespie_ssa_next_reaction(state_curr)
            t = t + delta_t
            if t > stop_time:
                return state_curr
            else:
                state_curr = self.update_state(next_reaction, state_curr)

    def generate_sampled_ssa_trajectory(self, stop_time, num_time_samples, seed=None):

        """
        Create a uniformly sampled SSA Trajectory.
        """
        if seed is None:
            random.seed(seed)
        sampling_times = np.linspace(0, stop_time, num_time_samples)
        state_curr = self.initial_state
        states_array = np.array([state_curr])
        for j in range(sampling_times.size - 1):
            state_curr = self.run_gillespie_ssa(state_curr, sampling_times[j + 1] - sampling_times[j])
            states_array = np.append(states_array, [state_curr], axis=0)
        return sampling_times, states_array

    def generate_sampled_ssa_trajectories(self, stop_time, num_time_samples, num_trajectories=1, seed=None):

        """
        Create several uniformly sampled SSA Trajectories.
        """
        states_trajectories = np.zeros([num_trajectories, num_time_samples, self.num_species])
        times = np.linspace(0, stop_time, num_time_samples)
        for i in range(num_trajectories):
            times, states_trajectories[i, :, :] = \
                self.generate_sampled_ssa_trajectory(stop_time, num_time_samples, seed)
        return times, states_trajectories

    def run_random_time_change(self, initial_state, stop_time):

        internal_times = np.zeros(self.num_reactions)
        unit_poisson_jump_times = -np.log(np.random.uniform(0, 1, self.num_reactions))
        curr_state = initial_state
        curr_time = 0
        delta_reactions = np.zeros([self.num_reactions])
        while 1:
            # compute delta
            prop = self.propensity_vector(curr_state)
            for k in range(self.num_reactions):
                if prop[k] > 0:
                    delta_reactions[k] = (unit_poisson_jump_times[k] - internal_times[k]) / prop[k]
                else:
                    delta_reactions[k] = np.inf
            next_reaction = np.argmin(delta_reactions, axis=0)
            delta_time = delta_reactions[next_reaction]
            internal_times += prop * delta_time
            curr_time += delta_time
            if curr_time > stop_time:
                return curr_state
            else:
                curr_state = self.update_state(next_reaction, curr_state)
                unit_poisson_jump_times[next_reaction] += -np.log(np.random.uniform(0, 1))

    def BPA_generate_rtc_difference_sample(self, initialstate1, initialstate2, stop_time, seed=None):
        if seed is None:
            random.seed(seed)
        state1 = initialstate1
        state2 = initialstate2

        internal_times = np.zeros([self.num_reactions, 3])
        unit_poisson_jump_times = -np.log(np.random.uniform(0, 1, [self.num_reactions, 3]))
        delta_reactions = np.zeros([self.num_reactions, 3])
        a_matrix = np.zeros([self.num_reactions, 3], dtype="float64")
        curr_time = 0

        while 1:
            if np.all(state1 == state2):
                return np.zeros([self.output_function_size], dtype="float64")
            prop1 = self.propensity_vector(state1)
            prop2 = self.propensity_vector(state2)
            for k in range(self.num_reactions):
                a_matrix[k, 0] = min(prop1[k], prop2[k])
                a_matrix[k, 1] = prop1[k] - a_matrix[k, 0]
                a_matrix[k, 2] = prop2[k] - a_matrix[k, 0]

            for k in range(self.num_reactions):
                for j in range(3):
                    if a_matrix[k, j] > 0:
                        delta_reactions[k, j] = (unit_poisson_jump_times[k, j] - internal_times[k, j]) / a_matrix[k, j]
                    else:
                        delta_reactions[k, j] = np.inf

            index = np.unravel_index(np.argmin(delta_reactions, axis=None), delta_reactions.shape)
            delta_t = delta_reactions[index]
            curr_time += delta_t
            if curr_time > stop_time:
                return self.output_function(np.expand_dims(state2, axis=0)) - self.output_function(
                    np.expand_dims(state1, axis=0))
            else:
                # update state
                if index[1] == 0 or index[1] == 1:
                    state1 = self.update_state(index[0], state1)
                if index[1] == 0 or index[1] == 2:
                    state2 = self.update_state(index[0], state2)

                # update internal times
                for k in range(self.num_reactions):
                    for j in range(3):
                        internal_times[k, j] += a_matrix[k, j] * delta_t

                # update Poisson jump time
                unit_poisson_jump_times[index] += -np.log(np.random.uniform(0, 1))

    def BPA_find_normalisation_parameters(self, stop_time, num_normalisation_paths, seed=None):
        if seed is None:
            random.seed(seed)
        param_names = list(self.parameter_dict.keys())
        rates_sum = np.zeros([len(param_names), self.num_reactions])
        for n in range(num_normalisation_paths):
            t = 0
            state_curr = self.initial_state
            while t < stop_time:
                delta_t, next_reaction = self.gillespie_ssa_next_reaction(state_curr)
                delta_t = min(delta_t, stop_time - t)
                t = t + delta_t
                rates_sum += delta_t * np.abs(self.propensity_sensitivity_matrix(state_curr))
                state_curr = self.update_state(next_reaction, state_curr)
        rates_sum = rates_sum / num_normalisation_paths
        return np.sum(rates_sum)

    def BPA_generate_sensitivity_sample_with_ssa(self, stop_time, normalisation_constant, seed=None):
        if seed is None:
            random.seed(seed)
        param_names = list(self.parameter_dict.keys())
        sample_value = np.zeros([len(param_names), self.output_function_size])
        t = 0
        state_curr = self.initial_state
        while t < stop_time:
            delta_t, next_reaction = self.gillespie_ssa_next_reaction(state_curr)
            delta_t = min(delta_t, stop_time - t)
            t = t + delta_t
            jacobian = self.propensity_sensitivity_matrix(state_curr)
            for i in range(len(param_names)):
                for k in range(self.num_reactions):
                    jac_abs = np.abs(jacobian[i, k])
                    jac_sign = np.sign(jacobian[i, k])
                    if jac_abs > 0:
                        state2 = self.update_state(k, state_curr)
                        rate = jac_abs * delta_t
                        prob = min(rate / normalisation_constant, 1)
                        if np.random.uniform(0, 1) < prob:
                            sample_value[i, :] += (jac_sign * rate / prob) * self.BPA_generate_rtc_difference_sample \
                                (state_curr, state2, stop_time - t + delta_t * np.random.uniform(0, 1))

            state_curr = self.update_state(next_reaction, state_curr)
        return sample_value

    def BPA_generate_sensitivity_sample_with_mNRM(self, stop_time, normalisation_constant, seed=None):
        if seed is None:
            random.seed(seed)
        param_names = list(self.parameter_dict.keys())
        sample_value = np.zeros([len(param_names), self.output_function_size])

        internal_times = np.zeros(self.num_reactions)
        unit_poisson_jump_times = -np.log(np.random.uniform(0, 1, self.num_reactions))
        delta_reactions = np.zeros([self.num_reactions])
        t = 0
        state_curr = self.initial_state
        while t < stop_time:
            prop = self.propensity_vector(state_curr)
            for k in range(self.num_reactions):
                if prop[k] > 0:
                    delta_reactions[k] = (unit_poisson_jump_times[k] - internal_times[k]) / prop[k]
                else:
                    delta_reactions[k] = np.inf
            next_reaction = np.argmin(delta_reactions, axis=0)
            delta_t = delta_reactions[next_reaction]
            internal_times += prop * delta_t
            delta_t = min(delta_t, stop_time - t)
            t = t + delta_t
            jacobian = self.propensity_sensitivity_matrix(state_curr)
            for i in range(len(param_names)):
                for k in range(self.num_reactions):
                    jac_abs = np.abs(jacobian[i, k])
                    jac_sign = np.sign(jacobian[i, k])
                    if jac_abs > 0:
                        state2 = self.update_state(k, state_curr)
                        rate = jac_abs * delta_t
                        prob = min(rate / normalisation_constant, 1)
                        if np.random.uniform(0, 1) < prob:
                            sample_value[i, :] += (jac_sign * rate / prob) * self.BPA_generate_rtc_difference_sample \
                                (state_curr, state2, stop_time - t + delta_t * np.random.uniform(0, 1))
            state_curr = self.update_state(next_reaction, state_curr)
            unit_poisson_jump_times[next_reaction] += -np.log(np.random.uniform(0, 1))
        return sample_value

    def generate_sampled_rtc_trajectory(self, stop_time, num_time_samples, seed=None):

        """
        Create a uniformly sampled RTC Trajectories.
        """
        if seed is None:
            random.seed(seed)
        sampling_times = np.linspace(0, stop_time, num_time_samples)
        states_array = np.zeros([num_time_samples, self.num_species])
        reaction_counts_array = np.zeros([num_time_samples, self.num_reactions])
        compensator_array = np.zeros([num_time_samples, self.num_reactions])
        current_reaction_count = np.zeros(self.num_reactions)
        internal_times = np.zeros(self.num_reactions)
        unit_poisson_jump_times = -np.log(np.random.uniform(0, 1, self.num_reactions))
        curr_state = self.initial_state
        curr_time = 0
        delta_reactions = np.zeros([self.num_reactions])
        counter = 0
        while 1:
            # compute delta
            prop = self.propensity_vector(curr_state)
            for k in range(self.num_reactions):
                if prop[k] > 0:
                    delta_reactions[k] = (unit_poisson_jump_times[k] - internal_times[k]) / prop[k]
                else:
                    delta_reactions[k] = np.inf
            next_reaction = np.argmin(delta_reactions, axis=0)
            delta_time = delta_reactions[next_reaction]
            internal_times += prop * delta_time
            # update the arrays
            while counter < num_time_samples and curr_time <= sampling_times[counter] < (curr_time + delta_time):
                states_array[counter, :] = curr_state
                reaction_counts_array[counter, :] = current_reaction_count
                if counter > 0:
                    compensator_array[counter, :] = compensator_array[counter - 1, :] \
                                                    + prop * (sampling_times[counter] - sampling_times[counter - 1])
                counter += 1
            curr_time += delta_time
            if curr_time > stop_time:
                return sampling_times, states_array, reaction_counts_array, compensator_array
            else:
                curr_state = self.update_state(next_reaction, curr_state)
                current_reaction_count[next_reaction] += 1
                unit_poisson_jump_times[next_reaction] += -np.log(np.random.uniform(0, 1))

    def generate_sampled_rtc_trajectories(self, stop_time, num_time_samples, num_trajectories=1, seed=None):

        """
        Create several uniformly sampled RTC Trajectories.
        """
        if seed is None:
            random.seed(seed)

        states_trajectories = np.zeros([num_trajectories, num_time_samples, self.num_species])
        martingale_trajectories = np.zeros([num_trajectories, num_time_samples, self.num_reactions])
        times = np.linspace(0, stop_time, num_time_samples)
        for i in range(num_trajectories):
            times, states_array, reaction_counts_array, compensator_array \
                = self.generate_sampled_rtc_trajectory(stop_time, num_time_samples, seed)
            states_trajectories[i, :, :] = states_array
            martingale_trajectories[i, :, :] = reaction_counts_array - compensator_array
        return times, states_trajectories, martingale_trajectories

    # def generate_sampled_rtc_trajectories_random_initial_state(self, stop_time, num_time_samples, num_trajectories=1,
    #                                                            lambda_array=None, seed=None):
    #
    #     """
    #     Create several uniformly sampled RTC Trajectories.
    #     """
    #     if seed is None:
    #         random.seed(seed)
    #     if lambda_array is None:
    #         lambda_array = np.ones(self.num_species) * 10
    #
    #     states_trajectories = np.zeros([num_trajectories, num_time_samples, self.num_species])
    #     martingale_trajectories = np.zeros([num_trajectories, num_time_samples, self.num_reactions])
    #     times = np.linspace(0, stop_time, num_time_samples)
    #     for i in range(num_trajectories):
    #         self.initial_state = generate_random_initial_state(lambda_array)
    #         times, states_array, reaction_counts_array, compensator_array \
    #             = self.generate_sampled_rtc_trajectory(stop_time, num_time_samples, seed)
    #         states_trajectories[i, :, :] = states_array
    #         martingale_trajectories[i, :, :] = reaction_counts_array - compensator_array
    #     return times, states_trajectories, martingale_trajectories

    # # noinspection PyAttributeOutsideInit
    # def set_hist_output_function(self, levels_dict):
    #     output_function_sizes = [len(levels_dict[key]) + 1 for key in levels_dict.keys()]
    #     self.output_function_size = np.prod(output_function_sizes)
    #
    #     def func(state):
    #         index_list = []
    #         for i in range(len(self.output_species_indices)):
    #             index_list.append(sum([int(state[self.output_species_indices[i]] > k)
    #                                    for k in levels_dict[self.output_species_labels[i]]]))
    #
    #         if tf.executing_eagerly():
    #             index = np.stack(index_list, axis=0)
    #             code = np.ravel_multi_index(index, output_function_sizes)
    #         else:
    #             T = tf.stack(index_list, axis=0)
    #             code = encode(T, output_function_sizes)
    #         return tf.one_hot(code, self.output_function_size, dtype="float64")
    #
    #     self.output_function = func
    #
    # def set_first_moments_output_function(self):
    #     self.output_function_size = len(self.output_species_indices)
    #
    #     def func(state):
    #         return tf.stack([state[i] for i in self.output_species_indices])
    #
    #     self.output_function = func
    #
    # def set_first_and_second_moments_output_function(self):
    #     n = len(self.output_species_indices)
    #     self.output_function_size = int(n * (n + 3) / 2)
    #
    #     def func(state):
    #         output_list = [state[i] for i in self.output_species_indices]
    #         output_list_second_moment = [state[i] ** 2 for i in self.output_species_indices]
    #         output_list_cross_moments = [state[subset[0]] * state[subset[1]] for subset
    #                                      in itertools.combinations(self.output_species_indices, 2)]
    #         for elem in output_list_second_moment + output_list_cross_moments:
    #             output_list.append(elem)
    #
    #         return tf.stack(output_list, axis=0)
    #
    #     self.output_function = func
