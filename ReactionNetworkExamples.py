import numpy as np
import ReactionNetworkClass as rxn
import tensorflow as tf
import itertools
from scipy.integrate import odeint


class independent_birth_death(rxn.ReactionNetworkDefinition):
    """independent birth death network"""

    def __init__(self, num_species):
        num_reactions = 2 * num_species
        species_labels = ["X%d" % i for i in range(num_species)]
        output_species_labels = [species_labels[-1]]
        reactant_matrix = np.zeros([num_reactions, num_species], dtype=int)
        product_matrix = np.zeros([num_reactions, num_species], dtype=int)
        # 1. Birth of all the species
        for i in np.arange(num_species):
            product_matrix[i, i] = 1
        # 2. degradation of all the species
        for i in np.arange(num_species):
            reactant_matrix[num_species + i, i] = 1

        # define parameters
        parameter_dict = {'production rate': 10, 'degradation rate': 1}
        reaction_dict = {}
        for i in np.arange(num_species):
            reaction_dict[i] = ['mass action', 'production rate']
        for i in np.arange(num_species):
            reaction_dict[i + num_species] = ['mass action', 'degradation rate']

        super(independent_birth_death, self).__init__(num_species, num_reactions, reactant_matrix,
                                                      product_matrix, parameter_dict, reaction_dict,
                                                      species_labels, output_species_labels)
        self.set_propensity_vector()
        self.set_propensity_sensitivity_matrix()
        self.output_function_size = 2
        self.initial_state = np.zeros(self.num_species)

    # define output function
    def output_function(self, state):
        output_list = [state[:, i] for i in self.output_species_indices]
        output_list_second_moment = [state[:, i] ** 2 for i in self.output_species_indices]
        output_list_cross_moments = [state[:, subset[0]] * state[:, subset[1]] for subset
                                     in itertools.combinations(self.output_species_indices, 2)]
        for elem in output_list_second_moment + output_list_cross_moments:
            output_list.append(elem)

        return tf.stack(output_list, axis=1)

    # here we compute the exact outputs and their sensitivities for this example
    def moment_eqn_sens(self, y, t):
        dydt = np.zeros(np.shape(y))
        k = self.parameter_dict['production rate']
        g = self.parameter_dict['degradation rate']
        dydt[0] = k - g * y[0]
        dydt[1] = -2 * g * y[1] + (2 * k + g) * y[0] + k
        dydt_sens = np.zeros([len(self.parameter_dict.keys()), self.output_function_size])
        y_sens = np.reshape(y[self.output_function_size:], np.shape(dydt_sens), order='C')
        dydt_sens[0, 0] = 1 - g * y_sens[0, 0]
        dydt_sens[1, 0] = - y[0] - g * y_sens[1, 0]
        dydt_sens[0, 1] = - 2 * g * y_sens[0, 1] + 2 * y[0] + 2 * k * y_sens[0, 0] + 1
        dydt_sens[1, 1] = -2 * y[1] - 2 * g * y_sens[1, 1] + y[0] + (2 * k + g) * y_sens[1, 0]
        dydt[self.output_function_size:] = np.ndarray.flatten(dydt_sens, order='C')
        return dydt

    def exact_values(self, finaltime):
        y0 = np.zeros([self.output_function_size + self.output_function_size * len(self.parameter_dict.keys())])
        t = np.linspace(0, finaltime, 101)
        # solve the moment equations
        sol = odeint(self.moment_eqn_sens, y0, t)
        exact_sens = sol[-1, :]
        exact_function_vals = exact_sens[:self.output_function_size]
        exact_sens_vals = np.reshape(exact_sens[self.output_function_size:], [len(self.parameter_dict.keys()),
                                                                              self.output_function_size])
        return exact_function_vals, exact_sens_vals


class linear_signalling_cascade(rxn.ReactionNetworkDefinition):
    """linear signalling cascade network"""

    def __init__(self, num_species):
        num_reactions = 2 * num_species
        species_labels = ["X%d" % i for i in range(num_species)]
        output_species_labels = [species_labels[-1]]
        reactant_matrix = np.zeros([num_reactions, num_species], dtype=int)
        product_matrix = np.zeros([num_reactions, num_species], dtype=int)
        # 1. Constitutive production of the first species
        product_matrix[0, 0] = 1
        # 2. Catalytic production of the other species
        for i in np.arange(num_species - 1):
            reactant_matrix[i + 1, i] = 1
            product_matrix[i + 1, i] = 1
            product_matrix[i + 1, i + 1] = 1
        # 3. Dilution of all the species
        for i in np.arange(num_species):
            reactant_matrix[num_species + i, i] = 1

        # define parameters
        parameter_dict = {'base production rate': 10.0, 'translation rate': 5.0, 'dilution rate': 1.0}
        reaction_dict = {0: ['mass action', 'base production rate']}
        for i in np.arange(num_species - 1):
            reaction_dict[i + 1] = ['mass action', 'translation rate']
        for i in np.arange(num_species):
            reaction_dict[i + num_species] = ['mass action', 'dilution rate']

        super(linear_signalling_cascade, self).__init__(num_species, num_reactions, reactant_matrix,
                                                        product_matrix, parameter_dict, reaction_dict,
                                                        species_labels, output_species_labels)
        self.initial_state = np.zeros(self.num_species)
        self.set_propensity_vector()
        self.set_propensity_sensitivity_matrix()
        self.output_function_size = 2

    # define output function
    def output_function(self, state):
        output_list = [state[:, i] for i in self.output_species_indices]
        output_list_second_moment = [state[:, i] ** 2 for i in self.output_species_indices]
        output_list_cross_moments = [state[:, subset[0]] * state[:, subset[1]] for subset
                                     in itertools.combinations(self.output_species_indices, 2)]
        for elem in output_list_second_moment + output_list_cross_moments:
            output_list.append(elem)

        return tf.stack(output_list, axis=1)

    # here we compute the exact outputs and their sensitivities for this example
    def moment_eqn_sens(self, y, t):
        dydt = np.zeros(np.shape(y))
        beta = self.parameter_dict['base production rate']
        k = self.parameter_dict['translation rate']
        g = self.parameter_dict['dilution rate']
        n = self.num_species
        num_params = 3
        W = np.zeros([2 * n, n], dtype=float)
        w_0 = np.zeros(2 * n, dtype=float)
        w_0[0] = beta
        W[0:n, :] = k * np.diag(np.ones(n - 1), -1)
        W[n: 2 * n, :] = g * np.diag(np.ones(n))
        A = np.matmul(np.transpose(self.stoichiometry_matrix), W)
        b = np.matmul(np.transpose(self.stoichiometry_matrix), w_0)
        dydt[0:n] = np.matmul(A, y[0:n]) + b
        Sigma = np.reshape(y[n:n * (n + 1)], [n, n], order='C')
        dsigma_dt = np.matmul(A, Sigma) + np.matmul(Sigma, np.transpose(A))
        dsigma_dt += np.matmul(np.matmul(np.transpose(self.stoichiometry_matrix), np.diag(np.matmul(W, y[0:n]) + w_0)),
                               self.stoichiometry_matrix)
        dydt[n:n * (n + 1)] = np.ndarray.flatten(dsigma_dt, order='C')

        W_sens = np.zeros([num_params, 2 * n, n], dtype=float)
        A_sens = np.zeros([num_params, n, n], dtype=float)
        w_0_sens = np.zeros([num_params, 2 * n], dtype=float)
        b_sens = np.zeros([num_params, n], dtype=float)
        temp_dydt = np.zeros([num_params, n], dtype=float)
        temp2_dydt = np.zeros([num_params, n, n], dtype=float)
        # der w.r.t. beta
        w_0_sens[0, 0] = 1
        # der w.r.t. k
        W_sens[1, 0:n, :] = np.diag(np.ones(n - 1), -1)
        # der w.r.t. gamma
        W_sens[2, n:2 * n, :] = np.diag(np.ones(n))
        y_sens = np.reshape(y[n * (n + 1):n * (n + 1) + num_params * n], [num_params, n], order='C')
        Sigma_sens = np.reshape(y[n * (n + 1) + num_params * n:], [num_params, n, n], order='C')
        for i in np.arange(num_params):
            A_sens[i, :, :] = np.matmul(np.transpose(self.stoichiometry_matrix), W_sens[i, :, :])
            b_sens[i, :] = np.matmul(np.transpose(self.stoichiometry_matrix), w_0_sens[i, :])
            temp_dydt[i, :] = np.matmul(A_sens[i, :, :], y[0:n]) + np.matmul(A, y_sens[i, :]) + b_sens[i, :]
            temp2_dydt[i, :, :] = np.matmul(A_sens[i, :, :], Sigma) + np.matmul(A, Sigma_sens[i, :, :]) \
                                  + np.matmul(Sigma, np.transpose(A_sens[i, :, :])) + np.matmul(Sigma_sens[i, :, :],
                                                                                                np.transpose(A))
            temp2_dydt[i, :, :] += np.matmul(np.matmul(np.transpose(self.stoichiometry_matrix),
                                                       np.diag(np.matmul(W_sens[i, :, :], y[0: n])
                                                               + np.matmul(W, y_sens[i, :]) + w_0_sens[i, :])),
                                             self.stoichiometry_matrix)

        dydt[n * (n + 1):n * (n + 1) + num_params * n] = np.ndarray.flatten(temp_dydt, order='C')
        dydt[n * (n + 1) + num_params * n:] = np.ndarray.flatten(temp2_dydt, order='C')
        return dydt

    def exact_values(self, finaltime):
        n = self.num_species
        num_params = 3
        y0 = np.zeros([n * (n + 1) + num_params * n * (n + 1)])
        t = np.linspace(0, finaltime, 1001)
        # solve the moment equations
        sol = odeint(self.moment_eqn_sens, y0, t)
        exact_vals = sol[-1, :]
        Sigma = np.reshape(exact_vals[n:n * (n + 1)], [n, n], order='C')
        y_sens = np.reshape(exact_vals[n * (n + 1):n * (n + 1) + num_params * n], [num_params, n], order='C')
        Sigma_sens = np.reshape(exact_vals[n * (n + 1) + num_params * n:], [num_params, n, n], order='C')
        exact_function_vals = np.array([exact_vals[n - 1], Sigma[n - 1, n - 1] + exact_vals[n - 1] ** 2])
        exact_sens_vals = np.zeros([num_params, self.output_function_size])
        exact_sens_vals[:, 0] = y_sens[:, n - 1]
        exact_sens_vals[:, 1] = Sigma_sens[:, n - 1, n - 1] + 2 * exact_vals[n - 1] * exact_sens_vals[:, 0]
        return exact_function_vals, exact_sens_vals


class nonlinear_signalling_cascade(rxn.ReactionNetworkDefinition):
    """nonlinear_signalling_cascade network"""

    def __init__(self, num_species):
        num_reactions = 2 * num_species
        species_labels = ["X%d" % i for i in range(num_species)]
        output_species_labels = [species_labels[-1]]
        reactant_matrix = np.zeros([num_reactions, num_species], dtype=int)
        product_matrix = np.zeros([num_reactions, num_species], dtype=int)
        # 1. Constitutive production of the first species
        product_matrix[0, 0] = 1
        # 2. Enzymatic production of the other species
        for i in np.arange(num_species - 1):
            reactant_matrix[i + 1, i] = 1
            product_matrix[i + 1, i] = 1
            product_matrix[i + 1, i + 1] = 1
        # 3. Dilution of all the species
        for i in np.arange(num_species):
            reactant_matrix[num_species + i, i] = 1

        # define parameters
        parameter_dict = {'base production rate': 10.0, 'max translation rate': 100.0, 'Hill constant den': 10.0,
                          'Hill coefficient': 1.0, 'dilution rate': 1.0, 'basal rate': 1.0}
        reaction_dict = {0: ['mass action', 'base production rate']}
        for i in np.arange(num_species - 1):
            reaction_dict[i + 1] = ['Hill_activation', i, 'max translation rate', 'Hill constant den',
                                    'Hill coefficient', 'basal rate']
        for i in np.arange(num_species):
            reaction_dict[i + num_species] = ['mass action', 'dilution rate']

        super(nonlinear_signalling_cascade, self).__init__(num_species, num_reactions,
                                                                                 reactant_matrix,
                                                                                 product_matrix, parameter_dict,
                                                                                 reaction_dict,
                                                                                 species_labels, output_species_labels)
        self.initial_state = np.zeros(self.num_species)
        self.set_propensity_vector()
        self.set_propensity_sensitivity_matrix()
        self.output_function_size = 2

    # define output function
    def output_function(self, state):
        output_list = [state[:, i] for i in self.output_species_indices]
        output_list_second_moment = [state[:, i] ** 2 for i in self.output_species_indices]
        output_list_cross_moments = [state[:, subset[0]] * state[:, subset[1]] for subset
                                     in itertools.combinations(self.output_species_indices, 2)]
        for elem in output_list_second_moment + output_list_cross_moments:
            output_list.append(elem)

        return tf.stack(output_list, axis=1)


class linear_signalling_cascade_with_feedback(rxn.ReactionNetworkDefinition):
    """linear signalling cascade network with feedback"""

    def __init__(self, num_species):
        num_reactions = 2 * num_species
        species_labels = ["X%d" % i for i in range(num_species)]
        output_species_labels = [species_labels[-1]]
        reactant_matrix = np.zeros([num_reactions, num_species], dtype=int)
        product_matrix = np.zeros([num_reactions, num_species], dtype=int)
        # 1. Constitutive production of the first species
        product_matrix[0, 0] = 1
        # 2. Enzymatic production of the other species
        for i in np.arange(num_species - 1):
            reactant_matrix[i + 1, i] = 1
            product_matrix[i + 1, i] = 1
            product_matrix[i + 1, i + 1] = 1
        # 3. Dilution of all the species
        for i in np.arange(num_species):
            reactant_matrix[num_species + i, i] = 1

        # define parameters
        parameter_dict = {'max translation rate': 100.0, 'Hill constant den': 10.0, 'Hill coefficient': 1.0,
                          'translation rate': 5.0, 'dilution rate': 1.0, 'basal rate': 1.0}
        reaction_dict = {0: ['Hill_repression', num_species - 1, 'max translation rate', 'Hill constant den',
                             'Hill coefficient', 'basal rate']}
        for i in np.arange(num_species - 1):
            reaction_dict[i + 1] = ['mass action', 'translation rate']
        for i in np.arange(num_species):
            reaction_dict[i + num_species] = ['mass action', 'dilution rate']

        super(linear_signalling_cascade_with_feedback, self).__init__(num_species, num_reactions,
                                                                                reactant_matrix,
                                                                                product_matrix, parameter_dict,
                                                                                reaction_dict,
                                                                                species_labels, output_species_labels)
        self.initial_state = np.zeros(self.num_species)
        self.set_propensity_vector()
        self.set_propensity_sensitivity_matrix()
        self.output_function_size = 2

    # define output function
    def output_function(self, state):
        output_list = [state[:, i] for i in self.output_species_indices]
        output_list_second_moment = [state[:, i] ** 2 for i in self.output_species_indices]
        output_list_cross_moments = [state[:, subset[0]] * state[:, subset[1]] for subset
                                     in itertools.combinations(self.output_species_indices, 2)]
        for elem in output_list_second_moment + output_list_cross_moments:
            output_list.append(elem)

        return tf.stack(output_list, axis=1)
