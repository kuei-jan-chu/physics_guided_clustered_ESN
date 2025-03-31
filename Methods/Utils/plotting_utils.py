#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python
import math
import numpy as np
import socket

# Plotting parameters
import matplotlib
hostname = socket.gethostname()
# print("PLOTTING HOSTNAME: {:}".format(hostname))
CLUSTER = True if ((hostname[:2]=='eu')  or (hostname[:5]=='daint') or (hostname[:3]=='nid')) else False
if CLUSTER: matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import rc


from matplotlib import colors
import six
color_dict = dict(six.iteritems(colors.cnames))

font = {'size'   : 16, }
matplotlib.rc('font', **font)


def plotTrainingLosses(model, loss_train, loss_val, min_val_error,additional_str=""):
    if (len(loss_train) != 0) and (len(loss_val) != 0):
        min_val_epoch = np.argmin(np.abs(np.array(loss_val)-min_val_error))
        fig_path = model.saving_path + model.fig_dir + model.model_name + "/Loss_total"+ additional_str + ".pdf"
        fig, ax = plt.subplots()
        plt.title("Validation error {:.10f}".format(min_val_error))
        plt.plot(np.arange(np.shape(loss_train)[0]), loss_train, color=color_dict['green'], label="Train RMSE")
        plt.plot(np.arange(np.shape(loss_val)[0]), loss_val, color=color_dict['blue'], label="Validation RMSE")
        plt.plot(min_val_epoch, min_val_error, "o", color=color_dict['red'], label="optimal")
        ax.set_xlabel(r"Epoch")
        ax.set_ylabel(r"Loss")
        plt.legend()
        plt.savefig(fig_path)
        plt.close()

        fig_path = model.saving_path + model.fig_dir + model.model_name + "/Loss_total_log"+ additional_str + ".pdf"
        fig, ax = plt.subplots()
        plt.title("Validation error {:.10f}".format(min_val_error))
        plt.plot(np.arange(np.shape(loss_train)[0]), np.log(loss_train), color=color_dict['green'], label="Train RMSE")
        plt.plot(np.arange(np.shape(loss_val)[0]), np.log(loss_val), color=color_dict['blue'], label="Validation RMSE")
        plt.plot(min_val_epoch, np.log(min_val_error), "o", color=color_dict['red'], label="optimal")
        ax.set_xlabel(r"Epoch")
        ax.set_ylabel(r"Log-Loss")
        plt.legend()
        plt.savefig(fig_path)
        plt.close()

    else:
        print("## Empty losses. Not printing... ##")



def plotLatentDynamics(model, set_name, latent_states, ic_idx):
    # print(np.shape(latent_states))
    if np.shape(latent_states)[1] >= 2:
        fig, ax = plt.subplots()
        plt.title("Latent dynamics in {:}".format(set_name))
        X = latent_states[:, 0]
        Y = latent_states[:, 1]
        # epsilon = 1e-7
        # for i in range(len(X)-1):
        #     if np.abs(X[i+1]-X[i]) > epsilon and np.abs(Y[i+1]-Y[i]) > epsilon:
        #         # plt.arrow(X[i], Y[i], X[i+1]-X[i], Y[i+1]-Y[i], color='red', head_width=.05, shape='full', lw=0, length_includes_head=True, zorder=2, linestyle='')
        #         plt.arrow(X[i], Y[i], X[i+1]-X[i], Y[i+1]-Y[i], color='blue', head_width=.05, shape='full', length_includes_head=True, zorder=2)
        plt.tight_layout()
        plt.plot(X, Y, 'b', linewidth = 1, label='prediction', zorder=1)
        plt.legend(loc="lower right")
        plt.legend(fontsize=14)
        plt.xlabel('dim 1')
        plt.ylabel('dim 2')
        plt.autoscale(enable=True, axis='both')
        fig_path = model.saving_path + model.fig_dir + model.model_name + "/lattent_dynamics_{:}_{:}.pdf".format(set_name, ic_idx)
        plt.savefig(fig_path, dpi=300)
        plt.close()
    else:
        fig, ax = plt.subplots()
        plt.tight_layout()
        plt.legend(fontsize=14)
        plt.legend(loc="lower right")
        plt.title("Latent dynamics in {:}".format(set_name))
        plt.plot(latent_states[:-1, 0], latent_states[1:, 0], 'b', linewidth = 2.0, label='prediction')
        fig_path = model.saving_path + model.fig_dir + model.model_name + "/lattent_dynamics_{:}_{:}.pdf".format(set_name, ic_idx)
        plt.savefig(fig_path, dpi=300)
        plt.close()

def plotAttractor(model, set_name, targets, predictions, ic_idx):
    # print(np.shape(latent_states))
    if np.shape(predictions)[1] >= 2:
        fig, ax = plt.subplots()
        plt.title("Attractor reconstruction by " + labelNameForModels(model.class_name))
        X_predict = predictions[:, 0]
        Y_predict = predictions[:, 1]
        X_target = targets[:, 0]
        Y_target = targets[:, 1]
        # epsilon = 1e-7
        # for i in range(len(X_predict)-1):
        #     if np.abs(X_predict[i+1]-X_predict[i]) > epsilon and np.abs(Y_predict[i+1]-Y_predict[i]) > epsilon:
        #         # plt.arrow(X[i], Y[i], X[i+1]-X[i], Y[i+1]-Y[i], color='red', head_width=.05, shape='full', lw=0, length_includes_head=True, zorder=2, linestyle='')
        #         plt.arrow(X_predict[i], Y_predict[i], X_predict[i+1]-X_predict[i], Y_predict[i+1]-Y_predict[i], color='blue', head_width=.05, shape='full', length_includes_head=True, zorder=2)
        #         plt.arrow(X_target[i], Y_target[i], X_target[i+1]-X_target[i], Y_target[i+1]-Y_target[i], color='red', head_width=.05, shape='full', length_includes_head=True, zorder=2)
        plt.plot(X_predict, Y_predict, 'b', linewidth = 1, label='prediction', zorder=1)
        plt.plot(X_target, Y_target, 'r', linewidth = 1, label='target', zorder=1)
        plt.legend(loc="lower right")
        plt.legend(fontsize=14)
        plt.xlabel('dim 1')
        plt.ylabel('dim 2')
        plt.tight_layout()
        # plt.margins(0.05)
        # plt.autoscale(enable=True, axis='both')
        fig_path = model.saving_path + model.fig_dir + model.model_name + "/attractor_{:}_{:}.pdf".format(set_name, ic_idx)
        plt.savefig(fig_path, dpi=300)
        plt.close()
    else:
        fig, ax = plt.subplots()
        plt.title("Attractor reconstruction by " + labelNameForModels(model.class_name) + " in {:}".format(set_name))
        plt.plot(predictions[:-1, 0], predictions[1:, 0], 'b', linewidth = 2.0, label='prediction')
        plt.tight_layout()
        plt.legend(loc="lower right")
        plt.legend(fontsize=14)
        plt.xlabel('dim 1')
        plt.ylabel('dim 2')
        fig_path = model.saving_path + model.fig_dir + model.model_name + "/attractor_{:}_{:}.pdf".format(set_name, ic_idx)
        plt.savefig(fig_path, dpi=300)
        plt.close()



def plotIterativePrediction(model, set_name, target, prediction, error, nerror, ic_idx, dt, data_mle, data_std, truth_augment=None, prediction_augment=None, warm_up=None, latent_states=None):
    if latent_states is not None:
        plotLatentDynamics(model, set_name, latent_states, ic_idx)
    
    plotAttractor(model, set_name, target, prediction, ic_idx)
    
    dim = np.shape(target)[1]

    if ((truth_augment is not None) and (prediction_augment is not None)):
        for i in range(5):
            if i < dim:
                fig_path = model.saving_path + model.fig_dir + model.model_name + "/prediction_augmend_{:}_{:}_dim{:}.pdf".format(set_name, ic_idx, i)
                plt.figure(figsize=(15, 5))  # Width is twice the height
                # plt.plot(np.arange(np.shape(prediction_augment)[0]), prediction_augment[:,i], 'b', linewidth = 2.0, label='prediction')
                # plt.plot(np.arange(np.shape(truth_augment)[0]), truth_augment[:,i], 'r', linewidth = 2.0, label='target')
                # len = int(np.shape(prediction_augment)[0]*0.67)
                len = int(np.shape(prediction_augment)[0])
                plt.title("Time series prediction by " + labelNameForModels(model.class_name))
                plt.plot(np.arange(len), prediction_augment[:len,i], 'b', linewidth = 2.0, label='prediction')
                plt.plot(np.arange(len), truth_augment[:len,i], 'r--', linewidth = 2.0, label='target')
                plt.plot(np.ones((100,1))*warm_up, np.linspace(np.min([np.min(truth_augment[:len,i]), np.min(prediction_augment[:len,i])]), np.max([np.max(truth_augment[:len,i]), np.max(prediction_augment[:len,i])]), 100), 'g--', linewidth = 2.0, label='warm-up')
                # plt.ylim(-6, 6)
                plt.xlabel('time')
                plt.ylabel('dim 1')
                plt.tight_layout()
                plt.legend(loc="lower right")
                plt.legend(fontsize=14)
                plt.savefig(fig_path, bbox_inches="tight", dpi=300)
                plt.close()

    for i in range(5):
        if i < dim:
            plt.title("Time series prediction by " + labelNameForModels(model.class_name))
            fig_path = model.saving_path + model.fig_dir + model.model_name + "/prediction_{:}_{:}_dim{:}.pdf".format(set_name, ic_idx, i)
            plt.figure(figsize=(10, 5))  # Width is twice the height
            plt.plot(prediction[:,i], 'b', label='prediction')
            plt.plot(target[:,i], 'r', label='target')
            plt.tight_layout()
            plt.legend(loc="lower right")
            plt.legend(fontsize=14)
            plt.savefig(fig_path, bbox_inches="tight", dpi=300)
            plt.close()

    fig_path = model.saving_path + model.fig_dir + model.model_name + "/prediction_{:}_{:}_error.pdf".format(set_name, ic_idx)
    plt.plot(error, label='error')
    plt.legend(loc="lower right")
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(fig_path, bbox_inches="tight", dpi=300)
    plt.close()

    fig_path = model.saving_path + model.fig_dir + model.model_name + "/prediction_{:}_{:}_log_error.pdf".format(set_name, ic_idx)
    plt.plot(np.log(np.arange(np.shape(error)[0])), np.log(error), label='log(error)')
    plt.legend(loc="lower right")
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(fig_path, bbox_inches="tight", dpi=300)
    plt.close()

    fig_path = model.saving_path + model.fig_dir + model.model_name + "/prediction_{:}_{:}_nerror.pdf".format(set_name, ic_idx)
    plt.plot(nerror, label='nerror')
    plt.legend(loc="lower right")
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(fig_path, bbox_inches="tight", dpi=300)
    plt.close()

    if model.input_dim >=3: createTestingContours(model, target, prediction, dt, data_mle, ic_idx, data_std, set_name)

def plotFirstThreePredictions(model, num_test_ICS, targets_all, prediction_all, rmse_all, rmnse_all, test_ic_indexes, dt, data_mle, target_augment_all, prediction_augment_all, hidden_state_all, dynamics_length, data_std, set_name):
        if data_mle < 0:
            data_mle = 1
        for ic_num in range(num_test_ICS):  
            if ic_num < 3: plotIterativePrediction(model, set_name, targets_all[ic_num], prediction_all[ic_num], rmse_all[ic_num], rmnse_all[ic_num], test_ic_indexes[ic_num], dt, data_mle, data_std, target_augment_all[ic_num], prediction_augment_all[ic_num], dynamics_length, hidden_state_all[ic_num])
               
def labelNameForModels(model):
    if model == "standard_esn":
          return "Std-ESN"
    elif model == "pgclustered_esn":
          return "PGC-ESN"
    elif model == "asym_pgclustered_esn":
          return "PGC-ESN"
    elif model == "randomly_clustered_esn":
         return "RandC-ESN"
    elif model == "paralleled_esn":
         return "Paral-ESN"
    elif model == "partially_pgclustered_esn":
        return "PartPGC-ESN"
    elif model == "moved_pgclustered_esn":
        return "MovedPGC-ESN"
    else: 
        return model

def createTestingContours(model, target, output, dt, data_mle, ic_idx, data_std, set_name):
    fontsize = 24
    target = target[:700]
    output = output[:700]
    error = np.sqrt(np.square(target-output)/np.square(data_std))
    # error =  np.abs(target-output)
    # vmin = np.array([target.min(), output.min()]).min()
    # vmax = np.array([target.max(), output.max()]).max()
    vmin = int(target.min())
    vmax = int(target.max())
    vmin_error = 0.0
    # vmax_error = error.max()
    vmax_error = 5

    print("IC_IDX: {:}, \nVMIN: {:} \nVMAX: {:} \n".format(ic_idx, vmin, vmax))

    # Plotting the contour plot
    fig, axes = plt.subplots(nrows=1, ncols=3,figsize=(14, 6), sharey=True)
    fig.subplots_adjust(hspace=0.4, wspace = 0.4)
    axes[0].set_ylabel(r"$t/T^{\lambda}$")
    # axes[0].set_ylabel(r"t $/T$", fontsize=fontsize)
    time_end = np.ceil(target.shape[0]*dt*data_mle)
    # time_end = np.ceil(target.shape[0]*dt/data_mle)
    axes[0].set_yticks(np.arange(0, time_end + 1, np.ceil(time_end/6)))  # Adjust step size (e.g., 4) as needed

    createContour_(fig, axes[0], target, "Target", fontsize, vmin, vmax, plt.get_cmap("seismic"), dt, data_mle, np.ceil((vmax-vmin)/6))
    # createContour_(fig, axes[1], output, labelNameForModels(model.class_name) + " Prediction", fontsize, vmin, vmax, plt.get_cmap("seismic"), dt, data_mle, np.ceil((vmax-vmin)/6))
    createContour_(fig, axes[1], output, labelNameForModels(model.class_name) , fontsize, vmin, vmax, plt.get_cmap("seismic"), dt, data_mle, np.ceil((vmax-vmin)/6))
    createContour_(fig, axes[2], error, labelNameForModels(model.class_name) + " NRSE", fontsize, vmin_error, vmax_error, plt.get_cmap("Reds"), dt, data_mle, np.ceil(vmax_error/6))
    plt.tight_layout()
    fig_path = model.saving_path + model.fig_dir + model.model_name + "/prediction_{:}_{:}_contour.pdf".format(set_name, ic_idx)
    plt.savefig(fig_path, bbox_inches="tight", dpi=50)
    plt.close()

def createContour_(fig, ax, data, title, fontsize, vmin, vmax, cmap, dt, data_mle, cbar_step=1):
    if cbar_step == 0:
        cbar_step = 1
    ax.set_title(title)
    # np.meshgrid: create grid arrays for plotting time and state indices as the x and y coordinates for the contour plot.
    t, s = np.meshgrid(np.arange(data.shape[0])*dt*data_mle, np.arange(1, data.shape[1]+1))
    # specifying 15 contours and setting color scales with vmin and vmax.
    # level: divides the data range into 60 equal levels for smooth color transitions.
    # cmap: The "seismic" colormap highlights positive and negative deviations in target and output, while "Reds" is effective for emphasizing error magnitudes in error.
    mp = ax.contourf(s, t, np.transpose(data), 15, cmap=cmap, levels=np.linspace(vmin, vmax, 60), extend="both")
    cbar = fig.colorbar(mp, ax=ax)
    # Set integer labels for the colorbar
    cbar.set_ticks(np.arange(int(vmin), int(vmax) + 1, cbar_step))  # Adjust step size (e.g., 1) as needed
    # cbar.set_ticklabels([str(i) for i in range(int(vmin), int(vmax) + 1, 1)])  # Ensure they are displayed as integers
    ax.set_xlabel(r"$State$")
    # Ensure integer ticks on both axes
    # ax.set_xticks(np.arange(0, data.shape[1], data.shape[1]-1))  # Adjust step size (e.g., 10) as needed
    ax.set_xticks(np.arange(1, data.shape[1]+1, data.shape[1]-1))  # Adjust step size (e.g., 10) as needed

    return mp

def plotSpectrum(model, sp_true, sp_pred, freq_true, freq_pred, set_name):
    fig_path = model.saving_path + model.fig_dir + model.model_name + "/frequencies_{:}.pdf".format(set_name)
    plt.plot(freq_pred, sp_pred, 'b--', label="prediction")
    plt.plot(freq_true, sp_true, 'r--', label="target")
    plt.title("Power spectrum reconstruction by " + labelNameForModels(model.class_name))
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Power Spectrum [dB]')
    plt.legend(loc="lower right")
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(fig_path, bbox_inches="tight", dpi=300)
    plt.close()

def labelNameForMetrics(metric):
    if metric == "rmnse":
          return "NRMSE"
    elif metric == "num_accurate_pred_05":
          return "VPT(0.5)"
    elif metric == "num_accurate_pred_1":
          return "VPT(1)"
    elif metric == "error_freq":
         return "PSE"
    elif metric == "d_temp":
         return "D_temp"
    elif metric == "d_geom":
         return "D_geom"


def plotEvaluationResultOverAllXValues(all_models_results, params_dict, fig_dir_path, evaluation_metric):
    # Calculate the global min and max for y-axis
    y_min, y_max = float('inf'), float('-inf')
    # Create separate figures for TRAIN and TEST
    fig_train, ax_train = plt.subplots()
    fig_test, ax_test = plt.subplots()

    # Loop through all models' results
    for model_data in all_models_results:
        model_name = model_data["model_name"]
        model_results = model_data["model_results"]

        # Extract reservoir sizes, train, and test metrics
        # reservoir_sizes = [res[0] for res in model_results]
        # train_metrics = [res[1] for res in model_results]
        # test_metrics = [res[2] for res in model_results]
        reservoir_sizes = [res[params_dict["x_value"]] for res in model_results]  
        metric_results_avg_train = [res["metric_result_avg_train"] for res in model_results]      
        metric_results_avg_test = [res["metric_result_avg_test"] for res in model_results]
        metric_results_std_train = [res["metric_result_std_train"] for res in model_results]      
        metric_results_std_test = [res["metric_result_std_test"] for res in model_results]

        # Update global min and max
        # Calculate min and max considering avg ± std
        y_min = min(y_min, *(avg - std for avg, std in zip(metric_results_avg_train, metric_results_std_train)))
        y_min = min(y_min, *(avg - std for avg, std in zip(metric_results_avg_test, metric_results_std_test)))
        y_max = max(y_max, *(avg + std for avg, std in zip(metric_results_avg_train, metric_results_std_train)))
        y_max = max(y_max, *(avg + std for avg, std in zip(metric_results_avg_test, metric_results_std_test)))

        # # Plot TRAIN metrics
        # ax_train.plot(reservoir_sizes, metric_results_avg_train, label=model_name, marker='o')
        # # Plot TEST metrics
        # ax_test.plot(reservoir_sizes, metric_results_avg_test, label=model_name, marker='o')
        # Plot TRAIN metrics with error bars
        ax_train.errorbar(
            reservoir_sizes, 
            metric_results_avg_train, 
            yerr=metric_results_std_train, 
            label=labelNameForModels(model_name), 
            fmt='-o', 
            capsize=5
        )
        
        # Plot TEST metrics with error bars
        ax_test.errorbar(
            reservoir_sizes, 
            metric_results_avg_test, 
            yerr=metric_results_std_test, 
            label=labelNameForModels(model_name), 
            fmt='-o', 
            capsize=5
        )
    if math.isinf(y_min) or math.isinf(y_max):
        return 0
    
    if evaluation_metric == "num_accurate_pred_05" or evaluation_metric == "num_accurate_pred_1":
        y_max = 5

    # Add some padding to the y-axis range
    padding = (y_max - y_min) * 0.05  # 5% padding
    y_min -= padding
    y_max += padding

    metric_label = labelNameForMetrics(evaluation_metric)
    # Customize TRAIN plot
    # ax_train.set_title("Training Metrics")
    ax_train.set_title(metric_label + " over " + params_dict["x_value"])
    ax_train.set_xlabel(params_dict["x_value"])
    ax_train.set_ylabel(f"{metric_label}")
    ax_train.legend()
    # ax_train.grid(True)
    ax_train.set_ylim(y_min, y_max)
    # ax_train.set_yscale('log')
    ax_train.legend(fontsize=14)

    # Customize TEST plot
    ax_test.set_title(metric_label + " over " + params_dict["x_value"])
    ax_test.set_xlabel(params_dict["x_value"])
    ax_test.set_ylabel(f"{metric_label}")
    ax_test.legend()
    # ax_test.grid(True)
    ax_test.set_ylim(y_min, y_max)
    # ax_test.set_yscale('log')
    ax_test.legend(fontsize=14)


    # save the plots
    train_fig_file = fig_dir_path + "training_{:}_plot.pdf".format(evaluation_metric)
    test_fig_file = fig_dir_path + "testing_{:}_plot.pdf".format(evaluation_metric)
    fig_train.savefig(train_fig_file, dpi=300, bbox_inches='tight')  # Save TRAIN plot
    fig_test.savefig(test_fig_file, dpi=300, bbox_inches='tight')  # Save TEST plot

    # Close the figures to free memory
    plt.close(fig_train)
    plt.close(fig_test)


def plotFirstThreeNRMSEForAllModels(models, nrmses_for_all_models, num_test_ICS, testing_ic_indexes, dt, data_mle, fig_dir_path, set_name):
    for ic_num in range(num_test_ICS):  
        if ic_num < 3:
            nrmse_for_all_models = []
            for i in range(len(models)):
                nrmse_for_all_models.append(nrmses_for_all_models[i][ic_num])
            nrmse_for_all_models = np.vstack(nrmse_for_all_models)
            plotNRMSEForAllModels(models, nrmse_for_all_models, testing_ic_indexes[ic_num], dt, data_mle, fig_dir_path, set_name)
         
    return 0
     
def plotNRMSEForAllModels(models, nrmse_for_all_models, ic_idx, dt, data_mle, fig_dir_path, set_name):
    plt.figure(figsize=(4, 6))  # Adjust dimensions for a tall, narrow figure
    nrmse_for_all_models = nrmse_for_all_models[:, :700]
    time = np.arange(len(nrmse_for_all_models[0])) * dt * data_mle
    # time = np.arange(len(nrmse_for_all_models[0])) * dt / data_mle
    vmin, vmax = 0, max([max(nrmse_sequence) for nrmse_sequence in nrmse_for_all_models])
    # print(vmax)
    ymin, ymax = 0, time[-1]

    for model_name, nrmse_sequence in zip(models, nrmse_for_all_models):
        nrmse_sequence = nrmse_sequence[:1000]
        plt.plot(np.transpose(nrmse_sequence), time, label=labelNameForModels(model_name), linewidth=1.5)
    
    ax = plt.gca()  # Get the current axis object
    ax.set_xlabel(r"NRMSE")
    ax.set_ylabel(r"$t/T^{\lambda}$")
    ax.set_xlim(vmin, np.ceil(vmax))  # Set x-axis limits
    ax.set_ylim(ymin, ymax)  # Set y-axis limits
    ax.set_xticks(np.arange(vmin, np.ceil(vmax)+1, np.ceil(vmax/2)))  # Adjust x-axis ticks
    ax.set_yticks(np.arange(0, int(ymax) + 1, np.ceil(ymax/6)))  # Adjust y-axis ticks
    # ax.tick_params(axis="both", which="major", labelsize=14)

    plt.title("NRMSE")
    plt.legend(fontsize=14)
    # plt.grid(True, which="both", linestyle="--", linewidth=0.1)
    plt.tight_layout()
    fig_path = fig_dir_path + "/nrmse_evolution_{:}_{:}.pdf".format(set_name, ic_idx)
    plt.savefig(fig_path, bbox_inches="tight", dpi=300)
    return 0

def plotTwoList(list_y, list_x, y_label, x_label, file_path, file_name):
    y_min = min(list_y)
    y_max = max(list_y)
    padding = (y_max - y_min) * 0.05  # 5% padding
    fig = plt.figure()  # Set figure size
    plt.ylim(y_min - padding, y_max + padding)
    plt.plot(list_x, list_y, marker="o", linestyle="-", color="b")  # Plot with markers
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f"{y_label} vs {x_label}")
    fig_path = file_path + "{:}.pdf".format(file_name)
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')  # Save TRAIN plot

    return 0

