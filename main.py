import ReactionNetworkExamples as rxn_examples
import tensorflow as tf
import CME_Solver as cme
import simulation_validation as sim
import plotting as plt_file
import matplotlib.pyplot as plt
import json
import os
import sys
import data_saving as dts


def main(argv):
    tf.keras.backend.set_floatx("float64")
    config_filename = "./Configs/" + argv[1]
    print("The configuration file is: " + config_filename)
    config_file = open(config_filename, )
    config_data = json.load(config_file)
    config_file.close()
    # get the network
    network = getattr(rxn_examples, config_data['reaction_network_config']['reaction_network_name'])(config_data['reaction_network_config']['num_species'])
    # set up output folder
    results_folder_path = config_data['reaction_network_config']['output_folder'] + "/"
    if not os.path.exists(results_folder_path):
        os.makedirs(results_folder_path)
    if config_data['net_config']['dnn_training_needed'] == "True":
        cme_solver = cme.CMESolver(network, config_data)
        print('Begin to solve %s ' % config_data['reaction_network_config']['reaction_network_name'])
        training_history, function_value_data, total_num_simulated_trajectories = cme_solver.train()
        # save the training data
        dts.save_training_data(results_folder_path, training_history, function_value_data)
        # compute the sensitivity matrix from the trained model
        dnn_sens_matrix = cme_solver.estimate_parameter_sensitivities()
        # save the sensitivity matrix in a file
        dts.save_sensitivity_data(results_folder_path, dnn_sens_matrix, list(network.parameter_dict.keys()))
        # save the model weights
        cme_solver.model.save_weights(results_folder_path + "trained_weights")

    # simulation based validation
    if config_data['simulation_validation']['simulation_based_validation_needed'] == "True":
        num_ssa_simulations = config_data["simulation_validation"]["num_trajectories"]
        number_of_auxiliary_paths_for_BPA = config_data["simulation_validation"]["number_of_auxiliary_paths_for_BPA"]
        sim.validate_with_simulation(network, config_data['reaction_network_config']['final_time'], num_ssa_simulations,
                                     results_folder_path)
        sim.generate_sensitivity_estimate(network, config_data['reaction_network_config']['final_time'], num_ssa_simulations,
                                          results_folder_path, number_of_auxiliary_paths_for_BPA)
    func_names = config_data["reaction_network_config"]["func_names"]
    save_pdf = True
    # plot loss function and function trajectories (optional)
    plt_file.plotLossFunction(results_folder_path, save_pdf)
    # compare with simulation based results (optional)
    if config_data["reaction_network_config"]["exact_values_computable"] == "True":
        exact_function_vals, exact_sens_vals = network.exact_values(config_data['reaction_network_config']['final_time'])
    else:
        exact_function_vals = None
        exact_sens_vals = None
    plt_file.plot_validation_charts_function_separate(results_folder_path,
                                             func_names, exact_function_vals, save_pdf)
    plt_file.plot_validation_charts_sensitivity(results_folder_path, func_names,
                                                config_data["plotting"]["sensitivity_parameters"],
                                                config_data["plotting"]["parameter_labels"],  exact_sens_vals,
                                                save_pdf)
    plt.show()


if __name__ == '__main__':
    main(sys.argv)
