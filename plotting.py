import numpy as np
import seaborn as sns
import pandas as pd
import math
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"],
    "text.latex.preamble": [r'\usepackage{amsfonts}'],
    'font.size': 15,
    "xtick.labelsize": 15,  # large tick labels
    "ytick.labelsize": 15,  # large tick labels
    'figure.figsize': [9, 6]}  # default: 6.4 and 4.8
)


def barplot_err(x, y, yerr=None, legend_loc=0, data=None, ax=None, **kwargs):
    _data = []
    for _i in data.index:
        _row = data.loc[_i]
        if _row[yerr] is not None:
            _data_i = pd.concat([data.loc[_i:_i]] * 3, ignore_index=True, sort=False)
            _data_i[y] = [_row[y] - _row[yerr], _row[y], _row[y] + _row[yerr]]
        else:
            _data_i = pd.concat([data.loc[_i:_i]], ignore_index=True, sort=False)
            _data_i[y] = _row[y]
        _data.append(_data_i)
    _data = pd.concat(_data, ignore_index=True, sort=False)
    _ax = sns.barplot(x=x, y=y, data=_data, ci='sd', ax=ax, **kwargs)
    _ax.legend(loc=legend_loc, fontsize=12)
    # _ax.set_yscale("log")
    return _ax


def plotLossFunction(results_folder_path, save_pdf=False):
    plt.figure("Loss function")
    filename2 = results_folder_path + "training_history.csv"
    training_history = np.loadtxt(filename2, delimiter=",", skiprows=1)
    steps = training_history[:, 0].astype(int)
    loss_trj = training_history[:, 1]
    cpu_time = training_history[-1, 2]
    print("Training time %d seconds" % cpu_time)

    # plot the loss function
    plt.plot(steps, loss_trj, color='g', linewidth=2)
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    if save_pdf:
        plt.savefig(results_folder_path + "loss_function.pdf", bbox_inches='tight', transparent="False", pad_inches=0)


#
# def plotFunctionTrajectory(results_folder_path, func_names):
#     plt.figure("Function Trajectory")
#     filename = results_folder_path + "function_value_data.csv"
#     filename2 = results_folder_path + "training_history.csv"
#     training_history = np.loadtxt(filename2, delimiter=",", skiprows=1)
#     steps = training_history[:, 0].astype(int)
#     loss_trj = training_history[:, 1]
#     cpu_time = training_history[-1, 2]
#     del training_history
#     print("Training time %d seconds" % cpu_time)
#
#     function_value_data = pd.read_csv(filename, delimiter=",",
#                                       names=func_names)
#
#     function_value_data.insert(loc=0, column="Steps", value=steps)
#     function_value_data.set_index("Steps")
#     sns.set(rc={'figure.figsize': (10, 7)})
#     for f in func_names:
#         ax = sns.lineplot(x="Steps", y=f, data=function_value_data)
#     ax.set(ylabel="function values")
#     plt.legend(labels=func_names)


def plot_validation_charts_function(results_folder_path, func_names, exact_values, save_pdf=False):
    filename = results_folder_path + "function_value_data.csv"
    function_value_data = pd.read_csv(filename, delimiter=",",
                                      names=func_names)
    dnn_func_values = function_value_data.tail(1).to_numpy()[0, :]
    del function_value_data
    # we first compare the mean function estimates
    filename = results_folder_path + "SimulationValidation.txt"
    function_value_data = pd.read_csv(filename, delimiter=",",
                                      names=func_names, skiprows=3)
    ssa_func_values_mean = function_value_data.values[0, :]
    ssa_func_values_std = function_value_data.values[1, :]
    dict1 = {"Function": func_names, "Estimate": ssa_func_values_mean,
             "Error": 1.96 * ssa_func_values_std, "Estimator": "SSA"}
    dict2 = {"Function": func_names, "Estimate": dnn_func_values,
             "Error": None, "Estimator": "DeepCME"}
    df1 = pd.DataFrame(dict1)
    df2 = pd.DataFrame(dict2)
    if exact_values is not None:
        dict3 = {"Function": func_names, "Estimate": np.array(exact_values),
                 "Error": None, "Estimator": "Exact"}
        df3 = pd.DataFrame(dict3)
        df = pd.concat([df2, df1, df3])

    else:
        filename = results_folder_path + "SimulationValidation_exact.txt"
        function_value_data = pd.read_csv(filename, delimiter=",",
                                          names=func_names, skiprows=3)
        sim_est_func_values_mean2 = function_value_data.values[0, :]
        sim_est_values_std2 = function_value_data.values[1, :]
        dict3 = {"Function": func_names, "Estimate": sim_est_func_values_mean2,
                 "Error": 1.96 * sim_est_values_std2, "Estimator": "mNRM ($10^4$ samples)"}
        df3 = pd.DataFrame(dict3)
        df = pd.concat([df2, df1, df3])

    df.set_index(np.arange(0, 3 * len(func_names)), inplace=True)
    plt.figure("Estimated function values")
    barplot_err(x="Function", y="Estimate", legend_loc=3, yerr="Error", hue="Estimator",
                capsize=.2, data=df)
    if save_pdf:
        plt.savefig(results_folder_path + "func_estimates.pdf", bbox_inches='tight', transparent="False", pad_inches=0)


def plot_validation_charts_function_separate(results_folder_path, func_names, exact_values, save_pdf=False):
    filename = results_folder_path + "function_value_data.csv"
    function_value_data = pd.read_csv(filename, delimiter=",",
                                      names=func_names)
    dnn_func_values = function_value_data.tail(1).to_numpy()[0, :]
    print(dnn_func_values)
    del function_value_data
    # we first compare the mean function estimates
    filename = results_folder_path + "SimulationValidation.txt"
    function_value_data = pd.read_csv(filename, delimiter=",",
                                      names=func_names, skiprows=3)
    sim_est_func_values_mean = function_value_data.values[0, :]
    sim_est_values_std = function_value_data.values[1, :]

    dict1 = {"Function": func_names, "Estimate": sim_est_func_values_mean,
             "Error": 1.96 * sim_est_values_std, "Estimator": "mNRM ($10^3$ samples)"}
    dict2 = {"Function": func_names, "Estimate": dnn_func_values,
             "Error": None, "Estimator": "DeepCME"}
    df1 = pd.DataFrame(dict1)
    df2 = pd.DataFrame(dict2)
    if exact_values is not None:
        dict3 = {"Function": func_names, "Estimate": np.array(exact_values),
                 "Error": None, "Estimator": "Exact"}
        df3 = pd.DataFrame(dict3)
        df = pd.concat([df2, df1, df3])
    else:
        filename = results_folder_path + "SimulationValidation_exact.txt"
        function_value_data = pd.read_csv(filename, delimiter=",",
                                          names=func_names, skiprows=3)
        sim_est_func_values_mean2 = function_value_data.values[0, :]
        sim_est_values_std2 = function_value_data.values[1, :]
        dict3 = {"Function": func_names, "Estimate": sim_est_func_values_mean2,
                 "Error": 1.96 * sim_est_values_std2, "Estimator": "mNRM ($10^4$ samples)"}
        df3 = pd.DataFrame(dict3)
        df = pd.concat([df2, df1, df3])

    df.set_index(np.arange(0, 3 * len(func_names)), inplace=True)
    fig, axs = plt.subplots(1, len(func_names), num="Estimated function values")
    for i in range(len(func_names)):
        barplot_err(x="Function", y="Estimate", legend_loc=3, yerr="Error", hue="Estimator",
                    capsize=.2, data=df.loc[df["Function"] == func_names[i]], ax=axs[i])
        if i == 0:
            axs[i].set_ylabel("Function Estimate")
        else:
            axs[i].set_ylabel("")
        axs[i].set_title(func_names[i])
        axs[i].set_xticklabels("")
        axs[i].set_xlabel("")
    if save_pdf:
        plt.savefig(results_folder_path + "func_estimates.pdf", bbox_inches='tight', transparent="False", pad_inches=0)


def plot_validation_charts_sensitivity(results_folder_path, func_names, parameter_list, parameter_labels,
                                       exact_sens_estimates,
                                       save_pdf=False):
    filename1 = results_folder_path + "BPA_Sens_Values.txt"
    filename2 = results_folder_path + "DNN_Sens_Values.txt"
    filename3 = results_folder_path + "BPA_Sens_Values_exact.txt"
    if exact_sens_estimates is not None:
        sens_exact2 = pd.DataFrame(exact_sens_estimates, columns=func_names)
        func_names.insert(0, "Parameter")
    else:
        func_names.insert(0, "Parameter")
        sens_exact2 = pd.read_csv(filename3, delimiter=",", names=func_names, skiprows=3)
    sens_bpa2 = pd.read_csv(filename1, delimiter=",", names=func_names, skiprows=3)
    sens_dnn2 = pd.read_csv(filename2, delimiter=",", names=func_names, skiprows=1)

    if exact_sens_estimates is not None:
        sens_exact2.insert(0, "Parameter", sens_dnn2["Parameter"])
    if parameter_list:
        _extended = []
        for param in parameter_list:
            _extended.append(param + "(std.)")
        for param in _extended:
            parameter_list.append(param)
        sens_bpa = sens_bpa2[sens_bpa2["Parameter"].isin(parameter_list)]
        sens_dnn = sens_dnn2[sens_dnn2["Parameter"].isin(parameter_list)]
        sens_exact = sens_exact2[sens_exact2["Parameter"].isin(parameter_list)]
    else:
        sens_bpa = sens_bpa2
        sens_dnn = sens_dnn2
        sens_exact = sens_exact2
    addl_col1 = ["BPA ($10^3$ samples)" for _ in range(len(sens_bpa.index))]
    addl_col2 = ["DeepCME" for _ in range(len(sens_dnn.index))]
    if exact_sens_estimates is not None:
        addl_col3 = ["Exact" for _ in range(len(sens_exact.index))]
    else:
        addl_col3 = ["BPA ($10^4$ samples)" for _ in range(len(sens_exact.index))]
    sens_bpa.insert(3, "Estimator", addl_col1)
    sens_dnn.insert(3, "Estimator", addl_col2)
    sens_exact.insert(3, "Estimator", addl_col3)

    sens_bpa_mean = sens_bpa.head(int(sens_bpa.shape[0] / 2))
    sens_bpa_mean_std = sens_bpa.tail(int(sens_bpa.shape[0] / 2))
    if exact_sens_estimates is None:
        sens_exact_mean = sens_exact.head(int(sens_exact.shape[0] / 2))
        sens_exact_mean_std = sens_exact.tail(int(sens_exact.shape[0] / 2))

    fig, axs = plt.subplots(1, len(func_names) - 1, num='Estimated Parameter Sensitivities for output functions')
    for i in range(len(func_names) - 1):
        # sns.barplot(data=df, x="Parameter", y=func_names[i + 1], hue="Estimator")
        f1 = sens_bpa_mean[["Parameter", func_names[i + 1], "Estimator"]]
        stds = sens_bpa_mean_std[func_names[i + 1]].to_numpy()
        f1.insert(2, "Error", 1.96 * stds)
        f2 = sens_dnn[["Parameter", func_names[i + 1], "Estimator"]]
        f2.insert(2, "Error", None)
        if exact_sens_estimates is not None:
            f3 = sens_exact[["Parameter", func_names[i + 1], "Estimator"]]
            f3.insert(2, "Error", None)
            df = pd.concat([f2, f1, f3])
        else:
            f3 = sens_exact_mean[["Parameter", func_names[i + 1], "Estimator"]]
            stds = sens_exact_mean_std[func_names[i + 1]].to_numpy()
            f3.insert(2, "Error", 1.96 * stds)
            df = pd.concat([f2, f1, f3])
        df.set_index(np.arange(df.shape[0]), inplace=True)
        # sns.set(rc={'figure.figsize': (9, 6)})
        barplot_err(x="Parameter", y=func_names[i + 1], yerr="Error", legend_loc=3, hue="Estimator",
                    capsize=.2, data=df, ax=axs[i])
        if i == 0:
            axs[i].set_ylabel("Parameter Sensitivity Estimate")
        else:
            axs[i].set_ylabel("")
        axs[i].set_title(func_names[i + 1])
        if parameter_list:
            axs[i].set_xticklabels(parameter_labels, fontsize=20)
    if save_pdf:
        plt.savefig(results_folder_path + "Param_Sens.pdf", bbox_inches='tight', transparent="False", pad_inches=0)


def plotAllLossFunctions(result_folder_path, species_list, save_pdf=False):
    plt.rcParams.update({'figure.figsize': [6.4, 4.8]})
    plt.figure("Loss functions")
    loss_profile_dict = {}
    result_folder_path_root = result_folder_path.rstrip('0123456789/')
    for species in species_list:
        filename = result_folder_path_root + str(species) + "/training_history.csv"
        training_history = np.loadtxt(filename, delimiter=",", skiprows=1)
        steps = training_history[:, 0].astype(int)
        if training_history[0, 1] != 0:
            loss_trj = training_history[:, 1] / training_history[0, 1]
        else:
            loss_trj = training_history[:, 1]
        loss_profile_dict[species] = [steps, loss_trj]

    # plot the loss functions
    sns.set_style("ticks")
    sns.set_context("paper", font_scale=1.4)
    for species in species_list:
        ax = sns.lineplot(x=loss_profile_dict[species][0], y=loss_profile_dict[species][1], linewidth=2,
                          label="\# species = " + str(species))
    ax.legend(loc=0, fontsize=20)
    ax.grid(b=True, which='major', linewidth=1.0)
    ax.grid(b=True, which='minor', linewidth=0.5)
    # ax.set(ylabel='Loss')
    # ax.set(xlabel="steps")
    # ax.set(ylabel="")
    ax.set(xlabel="")

    if save_pdf:
        print("File saved: " + result_folder_path_root + str(species_list[0]) + "/all_loss_functions.pdf")
        plt.savefig(result_folder_path_root + str(species_list[0]) + "/all_loss_functions.pdf", bbox_inches='tight',
                    transparent="False", pad_inches=0)


def plotAllCPUTimes(result_folder_path, species_list, save_pdf=False):
    plt.rcParams.update({'figure.figsize': [9, 6]})
    plt.figure("CPU Times")
    num_species = []
    method = []
    cpu_time = []
    result_folder_path_root = result_folder_path.rstrip('0123456789/')
    for species in species_list:
        temp = np.loadtxt(result_folder_path_root + str(species) + "/training_history.csv", delimiter=",", skiprows=1)
        cpu_time.append(temp[-1, 2])
        method.append("DeepCME")
        num_species.append(species)
        temp = np.loadtxt(result_folder_path_root + str(species) + "/SimulationValidation.txt", delimiter=":",
                          skiprows=1, max_rows=1, dtype=str)
        sim_time = float(temp[1])
        temp = np.loadtxt(result_folder_path_root + str(species) + "/BPA_Sens_Values.txt", delimiter=":",
                          skiprows=1, max_rows=1, dtype=str)
        sim_time += float(temp[1])
        method.append("Simulation")
        cpu_time.append(sim_time)
        num_species.append(species)
    d = {"\# species": num_species, "CPU Time (seconds)": cpu_time, "Method": method}
    df = pd.DataFrame.from_dict(d)
    # sns.set_theme()
    sns.set_style("ticks")
    sns.set_context("paper", font_scale=1.4)

    ax = sns.barplot(x="\# species", y="CPU Time (seconds)", hue="Method", data=df)
    ax.set_yscale('log')
    plt.grid()
    ax.legend(loc=0, fontsize=20)
    ax.grid(b=True, which='major', linewidth=1.0)
    ax.grid(b=True, which='minor', linewidth=0.5)
    ax.set(ylabel="")
    ax.set(xlabel="")
    ax.get_xaxis().set_ticks([])
    # ax.get_yaxis().set_ticks([])
    if save_pdf:
        print("File saved: " + result_folder_path_root + str(species_list[0]) + "/all_cpu_times.pdf")
        plt.savefig(result_folder_path_root + str(species_list[0]) + "/all_cpu_times.pdf", bbox_inches='tight',
                    transparent="False", pad_inches=0)


def plot_validation_charts_function_separate_comparison(results_folder_path, results_folder_path2, func_names,
                                                        exact_values, save_pdf=False):
    filename = results_folder_path + "function_value_data.csv"
    function_value_data = pd.read_csv(filename, delimiter=",", names=func_names)
    dnn_func_values = function_value_data.tail(1).to_numpy()[0, :]
    filename2 = results_folder_path2 + "function_value_data.csv"
    function_value_data2 = pd.read_csv(filename2, delimiter=",", names=func_names)
    dnn_func_values2 = function_value_data2.tail(1).to_numpy()[0, :]
    print(dnn_func_values)
    print(dnn_func_values2)
    del function_value_data
    del function_value_data2
    dict1 = {"Function": func_names, "Estimate": dnn_func_values2, "Error": None, "Estimator": "DeepCME ($N_H$ = 4)"}
    dict2 = {"Function": func_names, "Estimate": dnn_func_values, "Error": None, "Estimator": "DeepCME ($N_H$ = 8)"}
    df1 = pd.DataFrame(dict1)
    df2 = pd.DataFrame(dict2)
    if exact_values is not None:
        dict3 = {"Function": func_names, "Estimate": np.array(exact_values),
                 "Error": None, "Estimator": "Exact"}
        df3 = pd.DataFrame(dict3)
        df = pd.concat([df1, df2, df3])
    else:
        filename = results_folder_path + "SimulationValidation_exact.txt"
        function_value_data = pd.read_csv(filename, delimiter=",",
                                          names=func_names, skiprows=3)
        sim_est_func_values_mean2 = function_value_data.values[0, :]
        sim_est_values_std2 = function_value_data.values[1, :]
        dict3 = {"Function": func_names, "Estimate": sim_est_func_values_mean2,
                 "Error": 1.96 * sim_est_values_std2, "Estimator": "mNRM ($10^4$ samples)"}
        df3 = pd.DataFrame(dict3)
        df = pd.concat([df1, df2, df3])

    df.set_index(np.arange(0, 3 * len(func_names)), inplace=True)
    fig, axs = plt.subplots(1, len(func_names), num="Estimated function values")
    for i in range(len(func_names)):
        barplot_err(x="Function", y="Estimate", legend_loc=3, yerr="Error", hue="Estimator",
                    capsize=.2, data=df.loc[df["Function"] == func_names[i]], ax=axs[i])
        if i == 0:
            axs[i].set_ylabel("Function Estimate")
        else:
            axs[i].set_ylabel("")
        axs[i].set_title(func_names[i])
        axs[i].set_xticklabels("")
        axs[i].set_xlabel("")
    if save_pdf:
        plt.savefig(results_folder_path + "func_estimates.pdf", bbox_inches='tight', transparent="False", pad_inches=0)


def plot_validation_charts_sensitivity_comparison(results_folder_path, results_folder_path2, func_names, parameter_list,
                                       parameter_labels,
                                       exact_sens_estimates,
                                       save_pdf=False):
    filename1 = results_folder_path + "DNN_Sens_Values.txt"
    filename2 = results_folder_path2 + "DNN_Sens_Values.txt"
    filename3 = results_folder_path + "BPA_Sens_Values_exact.txt"
    if exact_sens_estimates is not None:
        sens_exact2 = pd.DataFrame(exact_sens_estimates, columns=func_names)
        func_names.insert(0, "Parameter")
    else:
        func_names.insert(0, "Parameter")
        sens_exact2 = pd.read_csv(filename3, delimiter=",", names=func_names, skiprows=3)
    sens_dnn2 = pd.read_csv(filename1, delimiter=",", names=func_names, skiprows=1)
    sens_dnn2_2 = pd.read_csv(filename2, delimiter=",", names=func_names, skiprows=1)

    if exact_sens_estimates is not None:
        sens_exact2.insert(0, "Parameter", sens_dnn2["Parameter"])
    if parameter_list:
        _extended = []
        for param in parameter_list:
            _extended.append(param + "(std.)")
        for param in _extended:
            parameter_list.append(param)
        sens_dnn = sens_dnn2[sens_dnn2["Parameter"].isin(parameter_list)]
        sens_dnn_2 = sens_dnn2_2[sens_dnn2_2["Parameter"].isin(parameter_list)]
        sens_exact = sens_exact2[sens_exact2["Parameter"].isin(parameter_list)]
    else:
        sens_dnn = sens_dnn2
        sens_dnn_2 = sens_dnn2_2
        sens_exact = sens_exact2
    addl_col1 = ["DeepCME ($N_H$ = 8)" for _ in range(len(sens_dnn.index))]
    addl_col2 = ["DeepCME ($N_H$ = 4)" for _ in range(len(sens_dnn_2.index))]
    if exact_sens_estimates is not None:
        addl_col3 = ["Exact" for _ in range(len(sens_exact.index))]
    else:
        addl_col3 = ["BPA ($10^4$ samples)" for _ in range(len(sens_exact.index))]
    sens_dnn.insert(3, "Estimator", addl_col1)
    sens_dnn_2.insert(3, "Estimator", addl_col2)
    sens_exact.insert(3, "Estimator", addl_col3)
    if exact_sens_estimates is None:
        sens_exact_mean = sens_exact.head(int(sens_exact.shape[0] / 2))
        sens_exact_mean_std = sens_exact.tail(int(sens_exact.shape[0] / 2))

    fig, axs = plt.subplots(1, len(func_names) - 1, num='Estimated Parameter Sensitivities for output functions')
    for i in range(len(func_names) - 1):
        # sns.barplot(data=df, x="Parameter", y=func_names[i + 1], hue="Estimator")
        f1 = sens_dnn[["Parameter", func_names[i + 1], "Estimator"]]
        f1.insert(2, "Error", None)
        f2 = sens_dnn_2[["Parameter", func_names[i + 1], "Estimator"]]
        f2.insert(2, "Error", None)
        if exact_sens_estimates is not None:
            f3 = sens_exact[["Parameter", func_names[i + 1], "Estimator"]]
            f3.insert(2, "Error", None)
            df = pd.concat([f2, f1, f3])
        else:
            f3 = sens_exact_mean[["Parameter", func_names[i + 1], "Estimator"]]
            stds = sens_exact_mean_std[func_names[i + 1]].to_numpy()
            f3.insert(2, "Error", 1.96 * stds)
            df = pd.concat([f2, f1, f3])
        df.set_index(np.arange(df.shape[0]), inplace=True)
        # sns.set(rc={'figure.figsize': (9, 6)})
        barplot_err(x="Parameter", y=func_names[i + 1], yerr="Error", legend_loc=3, hue="Estimator",
                    capsize=.2, data=df, ax=axs[i])
        if i == 0:
            axs[i].set_ylabel("Parameter Sensitivity Estimate")
        else:
            axs[i].set_ylabel("")
        axs[i].set_title(func_names[i + 1])
        if parameter_list:
            axs[i].set_xticklabels(parameter_labels, fontsize=20)
    if save_pdf:
        plt.savefig(results_folder_path + "Param_Sens.pdf", bbox_inches='tight', transparent="False", pad_inches=0)