import numpy as np
import seaborn as sns
import pandas as pd
import math
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"],
    'font.size': 15,
    'figure.figsize': [9, 6]} # default: 6.4 and 4.8
)


def barplot_err(x, y, yerr=None, data=None, ax=None, **kwargs):
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
    _ax.legend(loc=1)
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
        plt.savefig(results_folder_path + "loss_function.pdf")

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
        m = 3
    else:
        df = pd.concat([df2, df1])
        m = 2

    df.set_index(np.arange(0, m * len(func_names)), inplace=True)
    plt.figure("Estimated function values")
    barplot_err(x="Function", y="Estimate", yerr="Error", hue="Estimator",
                capsize=.2, data=df)
    if save_pdf:
        plt.savefig(results_folder_path + "func_estimates.pdf")


def plot_validation_charts_function_separate(results_folder_path, func_names, exact_values, save_pdf=False):
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
        m = 3
    else:
        df = pd.concat([df2, df1])
        m = 2

    df.set_index(np.arange(0, m * len(func_names)), inplace=True)

    fig, axs = plt.subplots(1, len(func_names), num="Estimated function values")
    for i in range(len(func_names)):
        barplot_err(x="Function", y="Estimate", yerr="Error", hue="Estimator",
                    capsize=.2, data=df.loc[df["Function"] == func_names[i]], ax=axs[i])
        if i == 0:
            axs[i].set_ylabel("Function Estimate")
        else:
            axs[i].set_ylabel("")
        axs[i].set_title(func_names[i])
        axs[i].set_xticklabels("")
        axs[i].set_xlabel("")
    if save_pdf:
        plt.savefig(results_folder_path + "func_estimates.pdf")


def plot_validation_charts_sensitivity(results_folder_path, func_names, parameter_list, parameter_labels,
                                       exact_sens_estimates,
                                       save_pdf=False):
    filename1 = results_folder_path + "BPA_Sens_Values.txt"
    filename2 = results_folder_path + "DNN_Sens_Values.txt"
    if exact_sens_estimates is not None:
        sens_exact2 = pd.DataFrame(exact_sens_estimates, columns=func_names)
    func_names.insert(0, "Parameter")
    sens_bpa2 = pd.read_csv(filename1, delimiter=",", names=func_names, skiprows=3)
    sens_dnn2 = pd.read_csv(filename2, delimiter=",", names=func_names, skiprows=1)
    if exact_sens_estimates is not None:
        sens_exact2.insert(0, "Parameter", sens_bpa2["Parameter"])
    if parameter_list:
        _extended = []
        for param in parameter_list:
            _extended.append(param + "(std.)")
        for param in _extended:
            parameter_list.append(param)
        sens_bpa = sens_bpa2[sens_bpa2["Parameter"].isin(parameter_list)]
        sens_dnn = sens_dnn2[sens_dnn2["Parameter"].isin(parameter_list)]
        if exact_sens_estimates is not None:
            sens_exact = sens_exact2[sens_exact2["Parameter"].isin(parameter_list)]
        else:
            sens_exact = None
    else:
        sens_bpa = sens_bpa2
        sens_dnn = sens_dnn2
        if exact_sens_estimates is not None:
            sens_exact = sens_exact2
        else:
            sens_exact = None
    addl_col1 = ["BPA" for _ in range(len(sens_bpa.index))]
    addl_col2 = ["DeepCME" for _ in range(len(sens_dnn.index))]
    if sens_exact is not None:
        addl_col3 = ["Exact" for _ in range(len(sens_exact.index))]
    sens_bpa.insert(3, "Estimator", addl_col1)
    sens_dnn.insert(3, "Estimator", addl_col2)
    if sens_exact is not None:
        sens_exact.insert(3, "Estimator", addl_col3)
    sens_bpa_mean = sens_bpa.head(int(sens_bpa.shape[0] / 2))
    sens_bpa_mean_std = sens_bpa.tail(int(sens_bpa.shape[0] / 2))
    fig, axs = plt.subplots(1, len(func_names) - 1, num='Estimated Parameter Sensitivities for output functions')
    for i in range(len(func_names) - 1):
        # sns.barplot(data=df, x="Parameter", y=func_names[i + 1], hue="Estimator")
        f1 = sens_bpa_mean[["Parameter", func_names[i + 1], "Estimator"]]
        stds = sens_bpa_mean_std[func_names[i + 1]].to_numpy()
        f1.insert(2, "Error", 1.96 * stds)
        f2 = sens_dnn[["Parameter", func_names[i + 1], "Estimator"]]
        f2.insert(2, "Error", None)
        if sens_exact is not None:
            f3 = sens_exact[["Parameter", func_names[i + 1], "Estimator"]]
            f3.insert(2, "Error", None)
            df = pd.concat([f2, f1, f3])
        else:
            df = pd.concat([f2, f1])
        df.set_index(np.arange(df.shape[0]), inplace=True)
        #sns.set(rc={'figure.figsize': (9, 6)})

        barplot_err(x="Parameter", y=func_names[i + 1], yerr="Error", hue="Estimator",
                    capsize=.2, data=df, ax=axs[i])
        if i == 0:
            axs[i].set_ylabel("Parameter Sensitivity Estimate")
        else:
            axs[i].set_ylabel("")
        axs[i].set_title(func_names[i + 1])
        if parameter_list:
            axs[i].set_xticklabels(parameter_labels, fontsize=20)
    if save_pdf:
        plt.savefig(results_folder_path + "Param_Sens.pdf")
