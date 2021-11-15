import numpy as np


def save_cpu_time(results_folder_path, elapsed_time, training=True):
    if training:
        np.savez(results_folder_path + "training_data_cpu.npz", elapsed_time=elapsed_time)
    else:
        np.savez(results_folder_path + "validation_data_cpu.npz", elapsed_time=elapsed_time)


def load_cpu_time(results_folder_path, training=True):
    if training:
        return np.load(results_folder_path + "training_data_cpu.npz")['elapsed_time']
    else:
        return np.load(results_folder_path + "validation_data_cpu.npz")['elapsed_time']


def save_training_data(results_folder_path, training_history, function_value_data=None):
    np.savetxt('{}training_history.csv'.format(results_folder_path),
               training_history,
               fmt=['%d', '%.5f', '%d'],
               delimiter=",",
               header='step,loss_function,elapsed_time',
               comments='')
    if function_value_data is not None:
        np.savetxt('{}function_value_data.csv'.format(results_folder_path),
                   function_value_data,
                   delimiter=",",
                   comments='')


def save_sensitivity_data(results_folder_path, dnn_sens_matrix, param_names):
    outfile = open((results_folder_path + "DNN_Sens_Values.txt"), "w+")
    outfile.write("Parameter ")
    for i in range(np.shape(dnn_sens_matrix)[1]):
        outfile.write(format(" func.%u" % (i + 1)))
    outfile.write("\n")
    for j in range(np.shape(dnn_sens_matrix)[0]):
        outfile.write(param_names[j])
        for i in range(np.shape(dnn_sens_matrix)[1]):
            outfile.write(format(",%.3f" % dnn_sens_matrix[j, i]))
        outfile.write("\n")
    outfile.close()


def save_sampled_trajectories(results_folder_path, data, sample_type):
    times, states_trajectories, martingale_trajectories = data
    np.savez(results_folder_path + sample_type + ".npz", times=times, states_trajectories=states_trajectories,
             martingale_trajectories=martingale_trajectories)


def load_save_sampled_trajectories(results_folder_path, sample_type):
    npzfile = np.load(results_folder_path + sample_type + ".npz")
    times = npzfile['times']
    states_trajectories = npzfile['states_trajectories']
    martingale_trajectories = npzfile['martingale_trajectories']
    return times, states_trajectories, martingale_trajectories
