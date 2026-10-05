import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from pathlib import Path
from typing import List, Optional

def plot_measurement(
    num_networks: int, 
    num_epochs: int, 
    network_logs: List[any], 
    measurement: str, 
    scale: str, 
    y_label: str, 
    c_label: str, 
    title: str, 
    filename: str, 
    min_network: int = 0,
    output_dir: str = "Bilder"
):
    """Plots training metrics vs. epochs with a specific color gradient."""
    n_list = [2**n for n in range(min_network, min_network + num_networks)]
        
    # Colormap and norm for log scale
    cmap = cm.autumn_r
    cmap = mcolors.LinearSegmentedColormap.from_list('autumn_r_truncated', cmap(np.linspace(0.1, 1, 256)))
    norm = mcolors.Normalize(vmin=0, vmax=num_networks - 1)
    
    fig, ax = plt.subplots(figsize=(7, 5))

    # Select measurement
    for i in range(num_networks):
        if measurement == "gapAcc":
            x = np.array(network_logs[i].history["accuracy"]) - np.array(network_logs[i].history["val_accuracy"])
        elif measurement == "gapLoss":
            x = np.array(network_logs[i].history["loss"]) - np.array(network_logs[i].history["val_loss"])
        elif measurement == "1-vacc":
            # Keeping exact original logic:
            x = 1 - np.array(network_logs[i].history["accuracy"])     
        else:
            x = network_logs[i].history[measurement]
            
        ax.plot(x, color=cmap(norm(i)), linewidth=2)
        
    plt.xlim([1, num_epochs])

    if scale == 'log':
        plt.xscale('log', base=2)
        ax.set_xticks([2**n for n in range(int(np.log2(num_epochs)) + 1)])
        ax.set_xticklabels([2**n for n in range(int(np.log2(num_epochs)) + 1)])
    elif scale == 'loglog':
        plt.xscale('log', base=2)
        plt.yscale('log', base=2)
        ax.set_xticks([2**n for n in range(int(np.log2(num_epochs)) + 1)])
        ax.set_xticklabels([2**n for n in range(int(np.log2(num_epochs)) + 1)])       
    
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.title(title)
    plt.grid(True)
    
    if measurement == "val_accuracy":
        plt.ylim([0, 1])
        
    # Generate Colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(c_label, fontsize=14)
    cbar.set_ticks(range(num_networks))
    cbar.set_ticklabels(n_list)
    
    # Save
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path / f"{filename}.svg", format="svg", bbox_inches="tight")
    fig.savefig(out_path / f"{filename}.png", format="png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_batchsize(
    num_networks: int, 
    num_epochs: int, 
    network_logs: List[any], 
    measurement: str, 
    scale: str, 
    y_label: str, 
    c_label: str, 
    title: str, 
    filename: str, 
    min_network: int = 0,
    output_dir: str = "Bilder"
):
    """Plots training metrics scaled by batch size."""
    n_list = [2**n for n in range(min_network, min_network + num_networks)]
        
    cmap = cm.autumn_r
    cmap = mcolors.LinearSegmentedColormap.from_list('autumn_r_truncated', cmap(np.linspace(0.1, 1, 256)))
    norm = mcolors.Normalize(vmin=0, vmax=num_networks - 1)
    
    fig, ax = plt.subplots(figsize=(7, 5))

    for i in range(num_networks):
        if measurement == "gapAcc":
            y = np.array(network_logs[i].history["accuracy"]) - np.array(network_logs[i].history["val_accuracy"])
        elif measurement == "gapLoss":
            y = np.array(network_logs[i].history["loss"]) - np.array(network_logs[i].history["val_loss"])
        elif measurement == "1-vacc":
            y = 1 - np.array(network_logs[i].history["accuracy"])     
        else:
            y = network_logs[i].history[measurement]
            
        x = np.array(range(len(network_logs[i].history[measurement]))) / n_list[i]
        ax.plot(x, y, color=cmap(norm(i)), linewidth=2)
        
    plt.xlim([1, num_epochs / 2])

    # Keeping original fixed ticks
    if scale == 'log':
        plt.xscale('log', base=2)
        ax.set_xticks([2**n for n in range(-9, 7)])
        ax.set_xticklabels(["", "1/256", "", "1/64", "", "1/16", "", "1/4", "1/2", "1", "2", "4", "8", "16", "32", "64"])    
    elif scale == 'loglog':
        plt.xscale('log', base=2)
        plt.yscale('log', base=2)
        ax.set_xticks([2**n for n in range(-9, 7)])
        ax.set_xticklabels(["", "1/256", "", "1/64", "", "1/16", "", "1/4", "1/2", "1", "2", "4", "8", "16", "32", "64"])       
    
    plt.xlabel("Epochs / Batchsize", fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.title(title)
    plt.grid(True)
    
    if measurement == "val_accuracy":
        plt.ylim([0, 1])
        
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(c_label, fontsize=14)
    cbar.set_ticks(range(num_networks))
    cbar.set_ticklabels(n_list)
    
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path / f"{filename}.svg", format="svg", bbox_inches="tight")
    fig.savefig(out_path / f"{filename}.png", format="png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_time(
    networks: List[str], 
    labels: List[str], 
    title: str, 
    filename: str,
    data_dir: str = "RawNetworkData",
    output_dir: str = "Bilder"
):
    """Plots training time vs number of neurons."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    data_path = Path(data_dir)
    
    for fn, lb in zip(networks, labels):
        file_name = data_path / f"{fn}.pkl"
        with open(file_name, "rb") as f:
            loaded_data = pickle.load(f)
            
        n_size = loaded_data[1][:, 0].size
        x_vals = [2**n for n in range(n_size)]
        y_vals = loaded_data[1][:, 0]
        ax.plot(x_vals, y_vals, label=lb, linewidth=2)
        
    plt.ylim(bottom=0)
    plt.ylabel("Training Time (s)", fontsize=14)
    plt.xlabel("Neurons", fontsize=14)
    plt.xscale("log", base=2)
    plt.xticks([2**n for n in range(n_size)], labels=[2**n for n in range(n_size)])
    plt.legend(fontsize=12)
    plt.title(title)
    
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path / f"{filename}.svg", format="svg", bbox_inches="tight")
    fig.savefig(out_path / f"{filename}.png", format="png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_activations(
    network_file: str, 
    activations: List[str], 
    measurement: str, 
    y_label: str, 
    scale: str, 
    title: str, 
    filename: str,
    data_dir: str = "RawNetworkData",
    output_dir: str = "Bilder"
):
    """Plots training metrics comparing different activation functions."""
    fig, ax = plt.subplots(figsize=(7, 5))

    file_name = Path(data_dir) / f"{network_file}.pkl"
    with open(file_name, "rb") as f:
        loaded_data = pickle.load(f)

    epochs = np.log2(len(loaded_data[0][0].history["val_accuracy"]))

    if measurement == "1-vacc":
        for i in range(5):
            plt.plot(1 - np.array(loaded_data[0][i].history["val_accuracy"]), label=activations[i]) 
    elif measurement == "1-acc":
        for i in range(5):
            plt.plot(1 - np.array(loaded_data[0][i].history["accuracy"]), label=activations[i])         
    else:
        for i in range(5):
            plt.plot(loaded_data[0][i].history[measurement], label=activations[i])
        
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.grid()
    
    # Keeping original scale behaviors
    if scale == "log":
        plt.xscale("log", base=2)
        plt.xticks([2**n for n in range(int(epochs) + 1)], labels=[2**n for n in range(int(epochs) + 1)])

    if scale == "loglog":
        # Notice: The original code used base=10 for loglog here!
        plt.xscale("log", base=10)
        plt.yscale("log", base=10)
        plt.xticks([2**n for n in range(int(epochs) + 1)], labels=[2**n for n in range(int(epochs) + 1)])  
        
    plt.legend(fontsize=12)
    plt.title(title)
    
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path / f"{filename}.svg", format="svg", bbox_inches="tight")
    fig.savefig(out_path / f"{filename}.png", format="png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_confusion_matrix(
    cm: np.ndarray, 
    symbols: List[str], 
    filename: str,
    output_dir: str = "Bilder"
):
    """Plots a confusion matrix as a percentage-based heatmap."""
    # Row-wise normalization (relative share in %)
    cm_relative = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_percent = cm_relative * 100
    
    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(cm_percent, fmt='.1f', cmap='Greens', xticklabels=symbols, yticklabels=symbols)
    
    plt.tick_params(axis='both', which='major', labelsize=6)
    plt.xlabel('Predicted Class', fontsize=14)
    plt.ylabel('True Class', fontsize=14)
    plt.title('Confusion Matrix (Relative in %)', fontsize=14)
    
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path / f"{filename}.svg", format="svg", bbox_inches="tight")
    fig.savefig(out_path / f"{filename}.png", format="png", dpi=300, bbox_inches="tight")
    plt.show()