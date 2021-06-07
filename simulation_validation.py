import numpy as np
import time
import CME_Solver as cme


def ssa_validation_routine(network, final_time, num_ssa_samples):
    state_list = []
    for i in range(num_ssa_samples):
        print("Generating trajectory %d" %i)
        state_list.append(network.run_gillespie_ssa(network.initial_state, final_time))
    return network.output_function(np.stack(state_list, axis=0))


def validate_with_simulation(network, final_time, num_ssa_samples, results_folder_path):
    validation_start_time = time.time()
    outfile = open((results_folder_path + "SimulationValidation.txt"), "w+")
    str1 = "The true values are estimated with %5u SSA samples: \n" % num_ssa_samples
    print("\n" + str1)
    str2 = "The function value estimates and standard deviations are given below\n"
    print(str2)
    func_values = ssa_validation_routine(network, final_time, num_ssa_samples)
    ssa_estimated_func_values = np.mean(func_values, axis=0)
    standard_deviations = np.std(func_values, axis=0) / np.sqrt(num_ssa_samples - 1)
    cme.print_array_nicely(ssa_estimated_func_values, "SSA Estimated function values")
    cme.print_array_nicely(standard_deviations, "Standard Deviations")
    str3 = "Total time needed for validation: %3u\n" % (time.time() - validation_start_time)
    print(str3)
    outfile.write(str1)
    outfile.write(str3)
    outfile.write(str2)
    outfile.write("Estimates")
    for i in range(ssa_estimated_func_values.size):
        outfile.write(format(",%.3f " % ssa_estimated_func_values[i]))
    outfile.write("\n")
    outfile.write("std.")
    for i in range(standard_deviations.size):
        outfile.write(format(",%.3f " % standard_deviations[i]))
    outfile.close()


def generate_sensitivity_estimate(network, final_time, num_ssa_samples, results_folder_path, number_of_auxiliary_paths):
    validation_start_time = time.time()
    normalisation_constant = network.BPA_find_normalisation_parameters(final_time, 100)
    normalisation_constant = normalisation_constant / number_of_auxiliary_paths
    param_names = list(network.parameter_dict.keys())
    sensitivity_matrix = np.zeros([num_ssa_samples, len(param_names), network.output_function_size])
    for i in range(num_ssa_samples):
        print("Generating sample %d" %i)
        sensitivity_matrix[i, :, :] = network.BPA_generate_sensitivity_sample(final_time, normalisation_constant)
    BPA_estimated_sensitivity_values = np.mean(sensitivity_matrix, axis=0)
    sensitivity_standard_deviations = np.std(sensitivity_matrix, axis=0) / np.sqrt(num_ssa_samples - 1)

    outfile = open((results_folder_path + "BPA_Sens_Values.txt"), "w+")
    str1 = "The true sensitivity values are estimated with %u SSA samples: \n" % num_ssa_samples
    outfile.write(str1)
    print(str1)
    str_time = "Total time needed for sensitivity validation: %3u\n" % (time.time() - validation_start_time)
    outfile.write(str_time)
    print(str_time )
    outfile.write("Parameter ")
    for i in range(np.shape(BPA_estimated_sensitivity_values)[1]):
        outfile.write(format(",func.%u" % (i + 1)))
    outfile.write("\n")
    param_names = list(network.parameter_dict.keys())
    for j in range(np.shape(BPA_estimated_sensitivity_values)[0]):
        outfile.write(param_names[j])
        for i in range(np.shape(BPA_estimated_sensitivity_values)[1]):
            outfile.write(format(",%.3f" % BPA_estimated_sensitivity_values[j, i]))
        outfile.write("\n")

    for j in range(np.shape(sensitivity_standard_deviations)[0]):
        outfile.write(param_names[j] + "(std.)")
        for i in range(np.shape(sensitivity_standard_deviations)[1]):
            outfile.write(format(",%.3f" % sensitivity_standard_deviations[j, i]))
        outfile.write("\n")
    outfile.close()
