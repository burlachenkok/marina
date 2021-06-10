#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import utils
import sys, os
from matplotlib.ticker import FormatStrFormatter

class NNConfiguration: pass

plt.rcParams["lines.markersize"] = 20
plt.rcParams["lines.linewidth"] = 2
plt.rcParams["font.size"] = 32
#===================================================================================================
def get_subset(ctr, indicies):
    return [ctr[ind] for ind in indicies]
#===================================================================================================

main_fig   = plt.figure(figsize=(18, 18))
grad_fig   = plt.figure(figsize=(18, 18))
bits_fig_1 = plt.figure(figsize=(18, 18))
bits_fig_2 = plt.figure(figsize=(18, 18))
main_fig_epochs   = plt.figure(figsize=(18, 18))
grad_fig_epochs  = plt.figure(figsize=(18, 18))

#===================================================================================================
files = sys.argv[1:]
g = -1

for fname in files:

#===================================================================================================
    my = utils.deserialize(fname)
    transfered_bits_by_node = my["transfered_bits_by_node"]
    fi_grad_calcs_by_node   = my["fi_grad_calcs_by_node"]
    train_loss              = my["train_loss"]
    test_loss               = my["test_loss"]
    #train_acc               = my["train_acc"]
    #test_acc                = my["test_acc"]
    fn_train_loss_grad_norm = my["fn_train_loss_grad_norm"]
    fn_test_loss_grad_norm  = my["fn_test_loss_grad_norm"]
    nn_config               = my["nn_config"]
    current_data_and_time   = my["current_data_and_time"]
    experiment_description  = my["experiment_description"]

    if experiment_description == "Training resnet18@CIFAR100":
        experiment_description = "Training ResNet-18 @ CIFAR100"

    compressor              = my["compressors"]

    nn_config               = my["nn_config"]
    algo_name               = my["algo_name"]

    if "compressors_rand_K" in my:
        compressors_rand_K      = my["compressors_rand_K"]
        algo_name               = algo_name + f" (K$\\approx${(compressors_rand_K/nn_config.D):.3f}d)"
    elif fname == "experiment_vr_marina_no_compr_custom_p.bin":
        compressors_rand_K      = " (no_compress)"
        algo_name               = algo_name + f" (no compr.$p\\approx{nn_config.p:.3f}$)" 
    else:
        compressors_rand_K  = "no_compress"
        algo_name               = algo_name + f" (no compr.)" 

    freq = 10

    train_loss = [train_loss[i] for i in range(len(train_loss)) if i % freq == 0]
    test_loss  = [test_loss[i]  for i in range(len(test_loss))  if i % freq == 0]
    #train_acc  = [train_acc[i]  for i in range(len(train_acc))  if i % freq == 0]
    #test_acc   = [test_acc[i]   for i in range(len(test_acc))   if i % freq == 0]
    fn_train_loss_grad_norm  = [fn_train_loss_grad_norm[i]  for i in range(len(fn_train_loss_grad_norm))  if i % freq == 0]
    fn_test_loss_grad_norm   = [fn_test_loss_grad_norm[i]   for i in range(len(fn_test_loss_grad_norm))   if i % freq == 0]

    #===================================================================================================
    print("==========================================================")
    print(f"Informaion about experiment results '{fname}'")
    print(f"  Content has been created at '{current_data_and_time}'")
    print(f"  Experiment description: {experiment_description}")
    print(f"  Dimension of the optimization proble: {nn_config.D}")
    print(f"  Compressor RAND-K K: {compressors_rand_K}")
    print(f"  Number of Workers: {nn_config.kWorkers}")
    print(f"  Used step-size: {nn_config.gamma}")
    print()
    print("Whole config")
    for k in dir(nn_config):
        v = getattr(nn_config, k)
        if type(v) == int or type(v) == float:
            print(" ", k, "=", v)
    print("==========================================================")

    #=========================================================================================================================
    KMax = nn_config.KMax
    mark_mult = 0.4

    fi_grad_calcs_sum      = np.sum(fi_grad_calcs_by_node, axis = 0)
    transfered_bits_sum    = np.sum(transfered_bits_by_node, axis = 0)        

    for i in range(1, KMax):
        transfered_bits_sum[i] = transfered_bits_sum[i] + transfered_bits_sum[i-1]

    transfered_bits_mean = transfered_bits_sum / nn_config.kWorkers       

    for i in range(1, KMax):
        fi_grad_calcs_sum[i] = fi_grad_calcs_sum[i] + fi_grad_calcs_sum[i-1]

    transfered_bits_mean_sampled = [transfered_bits_mean[i] for i in range(len(transfered_bits_mean)) if i % freq == 0] 

    #=========================================================================================================================

    epochs = (fi_grad_calcs_sum * 1.0) / (nn_config.train_set_full_samples)
    iterations = range(KMax)

    iterations_sampled =  [iterations[i] for i in range(len(iterations)) if i % freq == 0]
    epochs_sampled     =  [epochs[i] for i in range(len(epochs)) if i % freq == 0]

    #=========================================================================================================================
    markevery = [ int(mark_mult*KMax/4.0/freq*4.0), int(mark_mult*KMax/4.0/freq*4.0), int(mark_mult*KMax/3.5/freq), int(mark_mult*KMax/3.0/freq), 
                  int(mark_mult*KMax/5.0/freq), int(mark_mult*KMax/3.5/freq), int(mark_mult*KMax/2.5/freq), int(mark_mult*KMax/2.0/freq), int(mark_mult*KMax/3.0/freq)]

    marker = ["^","d","*","^","d","*", "*", "x", "*"]
    color = ["#dc143c", "#00008b", "#006400", "#dc143c", "#00008b", "#006400", "#c51b7d", "#000000", "#ff5e13"]
    linestyle = ["solid", "solid", "solid", "dashed", "dashed", "dashed", "dotted","dotted", "dotted"]
    #=========================================================================================================================
    g = (g + 1)%len(color)
    #=========================================================================================================================

    fig = plt.figure(main_fig.number)
    ax = fig.add_subplot(1, 1, 1)
    ax.semilogy(iterations_sampled, train_loss, color=color[g], 
                                                marker=marker[g], 
                                                markevery=markevery[g], 
                                                linestyle=linestyle[g], label=algo_name)


    #ax.semilogy(iterations_sampled, test_loss, color=color[g], marker=marker[g], markevery=markevery[g], linestyle=linestyle[g], label=algo_name + " test")
    ax.set_xlabel('Communication Rounds', fontdict = {'fontsize':35})
    ax.set_ylabel('$f(x)$', fontdict = {'fontsize':35})
    #ax.yaxis.set_major_formatter(FormatStrFormatter('%.2g'))

    ax.grid(True)
    ax.legend(loc='best', fontsize = 25)
    plt.title(f'{experiment_description}', fontdict = {'fontsize':35})
    plt.xticks(fontsize=27)
    plt.yticks(fontsize=30)
    fig.tight_layout()
    #=========================================================================================================================
    fig = plt.figure(grad_fig.number)
    ax = fig.add_subplot(1, 1, 1)

    #g = (g + 2)%len(color)
    ax.semilogy(iterations_sampled, fn_train_loss_grad_norm, color=color[g], marker=marker[g], markevery=markevery[g], linestyle=linestyle[g], label=algo_name)
    #ax.semilogy(iterations_sampled, fn_test_loss_grad_norm,  color=color[g], marker=marker[g], markevery=markevery[g], linestyle=linestyle[g], label="test")
    ax.set_xlabel('Communication Rounds', fontdict = {'fontsize':35})
    ax.set_ylabel('$||\\nabla f(x)||^2$', fontdict = {'fontsize':35})
    #ax.yaxis.set_major_formatter(FormatStrFormatter('%.2g'))

    ax.grid(True)
    ax.legend(loc='best', fontsize = 25)
    plt.title(f'{experiment_description}', fontdict = {'fontsize':35})
    plt.xticks(fontsize=27)
    plt.yticks(fontsize=30)
    fig.tight_layout()
    #=========================================================================================================================
    fig = plt.figure(bits_fig_1.number)
    ax = fig.add_subplot(1, 1, 1)

    #g = (g + 1)%len(color)
    ax.semilogy(transfered_bits_mean_sampled, train_loss, color=color[g], marker=marker[g], markevery=markevery[g], linestyle=linestyle[g], label=algo_name)
    #ax.semilogy(transfered_bits_mean_sampled, test_loss, color=color[5], marker=marker[5], markevery=markevery[5], linestyle=linestyle[5], label="test")

    ax.set_xlabel('#bits/n', fontdict = {'fontsize':35})
    ax.set_ylabel('f(x)', fontdict = {'fontsize':35})
    #ax.yaxis.set_major_formatter(FormatStrFormatter('%.2g'))

    ax.grid(True)
    ax.legend(loc='best', fontsize = 25)
    plt.title(f'{experiment_description}', fontdict = {'fontsize':35})
    plt.xticks(fontsize=27)
    plt.yticks(fontsize=30)
    fig.tight_layout()
    #=========================================================================================================================
    fig = plt.figure(bits_fig_2.number)
    ax = fig.add_subplot(1, 1, 1)

    #g = (g + 1)%len(color)
    ax.semilogy(transfered_bits_mean_sampled, fn_train_loss_grad_norm, color=color[g], marker=marker[g], markevery=markevery[g], linestyle=linestyle[g], label=algo_name)
    #ax.semilogy(transfered_bits_mean_sampled, fn_test_loss_grad_norm, color=color[1], marker=marker[1], markevery=markevery[1], linestyle=linestyle[1], label="test")

    ax.set_xlabel('#bits/n', fontdict = {'fontsize':35})
    ax.set_ylabel('$||\\nabla f(x)||^2$', fontdict = {'fontsize':35})
    ax.grid(True)
    ax.legend(loc='best', fontsize = 25)
    plt.title(f'{experiment_description}', fontdict = {'fontsize':35})
    plt.xticks(fontsize=27)
    plt.yticks(fontsize=30)
    fig.tight_layout()
    #=========================================================================================================================
    fig = plt.figure(main_fig_epochs.number)
    ax = fig.add_subplot(1, 1, 1)
    ax.semilogy(epochs_sampled, train_loss, color=color[g], 
                                                marker=marker[g], 
                                                markevery=markevery[g], 
                                                linestyle=linestyle[g], label=algo_name)


    #ax.semilogy(iterations_sampled, test_loss, color=color[g], marker=marker[g], markevery=markevery[g], linestyle=linestyle[g], label=algo_name + " test")
    ax.set_xlabel('Epochs', fontdict = {'fontsize':35})
    ax.set_ylabel('$f(x)$', fontdict = {'fontsize':35})
    #ax.yaxis.set_major_formatter(FormatStrFormatter('%.2g'))

    ax.grid(True)
    ax.legend(loc='best', fontsize = 25)
    plt.title(f'{experiment_description}', fontdict = {'fontsize':35})
    plt.xticks(fontsize=27)
    plt.yticks(fontsize=30)
    fig.tight_layout()
    #=========================================================================================================================
    fig = plt.figure(grad_fig_epochs.number)
    ax = fig.add_subplot(1, 1, 1)

    #g = (g + 2)%len(color)
    ax.semilogy(epochs_sampled, fn_train_loss_grad_norm, color=color[g], marker=marker[g], markevery=markevery[g], linestyle=linestyle[g], label=algo_name)
    #ax.semilogy(iterations_sampled, fn_test_loss_grad_norm,  color=color[g], marker=marker[g], markevery=markevery[g], linestyle=linestyle[g], label="test")
    ax.set_xlabel('Epochs', fontdict = {'fontsize':35})
    ax.set_ylabel('$||\\nabla f(x)||^2$', fontdict = {'fontsize':35})
    #ax.yaxis.set_major_formatter(FormatStrFormatter('%.2g'))

    ax.grid(True)
    ax.legend(loc='best', fontsize = 25)
    plt.title(f'{experiment_description}', fontdict = {'fontsize':35})
    plt.xticks(fontsize=27)
    plt.yticks(fontsize=30)
    fig.tight_layout()
    #=========================================================================================================================

    #fig = plt.figure(acc_fig.number)
    #ax = fig.add_subplot(1, 1, 1)

    #g = (g + 1)%len(color)
    #ax.semilogy(epochs_sampled, train_acc, color=color[g], marker=marker[g], markevery=markevery[g], linestyle=linestyle[g], label=algo_name + " train")

    #g = (g + 1)%len(color)
    #ax.semilogy(epochs_sampled, test_acc, color=color[g], marker=marker[g], markevery=markevery[g], linestyle=linestyle[g], label=algo_name + " test")

    #ax.set_xlabel('Epochs', fontdict = {'fontsize':35})
    #ax.set_ylabel('Accuracy', fontdict = {'fontsize':35})
    #ax.yaxis.set_major_formatter(FormatStrFormatter('%.2g'))

    #ax.grid(True)
    #ax.legend(loc='best', fontsize = 25)
    #plt.title(f'{experiment_description}', fontdict = {'fontsize':35})
    #plt.xticks(fontsize=27)
    #plt.yticks(fontsize=30)
    #fig.tight_layout()
    #=========================================================================================================================

if False:
    plt.show()

main_fig.tight_layout()
grad_fig.tight_layout()
bits_fig_1.tight_layout()
bits_fig_2.tight_layout()
main_fig_epochs.tight_layout()
grad_fig_epochs.tight_layout()

save_to = "1_main_fig.pdf"
main_fig.savefig(save_to, bbox_inches='tight')

save_to = "2_grad_fig.pdf"
grad_fig.savefig(save_to, bbox_inches='tight')

save_to = "3_bits_fig_1.pdf"
bits_fig_1.savefig(save_to, bbox_inches='tight')

save_to = "4_bits_fig_2.pdf"
bits_fig_2.savefig(save_to, bbox_inches='tight')

save_to = "5_main_fig_epochs.pdf"
main_fig_epochs.savefig(save_to, bbox_inches='tight')

save_to = "6_grad_fig_epochs.pdf"
grad_fig_epochs.savefig(save_to, bbox_inches='tight')

