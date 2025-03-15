import os
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import tensorflow as tf
import glob
from scipy import optimize
import matplotlib.gridspec as gridspec
from matplotlib.ticker import LogLocator, ScalarFormatter
import matplotlib as mpl
import argparse

# Set high-quality plot settings
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['axes.linewidth'] = 1.0
mpl.rcParams['grid.linewidth'] = 0.5
mpl.rcParams['lines.linewidth'] = 2.0
mpl.rcParams['xtick.major.width'] = 1.0
mpl.rcParams['ytick.major.width'] = 1.0
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'

# Create an argument parser
parser = argparse.ArgumentParser(description='Plot baseline models.')

# Add arguments for each base path
parser.add_argument('--lstm_base_path', type=str, required=True, help='Base path for LSTM model logs')
parser.add_argument('--rnn_base_path', type=str, required=True, help='Base path for RNN model logs')
parser.add_argument('--tds_cnv_base_path', type=str, required=True, help='Base path for TDS-CNV model logs')
parser.add_argument('--small_transformer_base_path', type=str, required=True, help='Base path for Small Transformer model logs')

# Add arguments for alternative paths
parser.add_argument('--alt_lstm_base_path', type=str, required=True, help='Alternative base path for LSTM model logs')
parser.add_argument('--alt_rnn_base_path', type=str, required=True, help='Alternative base path for RNN model logs')
parser.add_argument('--alt_tds_cnv_base_path', type=str, required=True, help='Alternative base path for TDS-CNV model logs')
parser.add_argument('--alt_small_transformer_base_path', type=str, required=True, help='Alternative base path for Small Transformer model logs')

# Parse the arguments
args = parser.parse_args()

# Assign parsed arguments to variables
lstm_base_path = args.lstm_base_path
rnn_base_path = args.rnn_base_path
tds_cnv_base_path = args.tds_cnv_base_path
small_transformer_base_path = args.small_transformer_base_path

alt_lstm_base_path = args.alt_lstm_base_path
alt_rnn_base_path = args.alt_rnn_base_path
alt_tds_cnv_base_path = args.alt_tds_cnv_base_path
alt_small_transformer_base_path = args.alt_small_transformer_base_path

# Create output directory for plots
output_dir = "baseline_plots"
os.makedirs(output_dir, exist_ok=True)

# Model names for better labeling
model_names = {
    "lstm": "LSTM",
    "rnn": "RNN",
    "tds_cnv": "TDS-CNV",
    "small_transformer": "Small Transformer"
}

# Model sizes (approximate number of parameters, in millions)
# These are estimates - update with actual values if known
model_sizes = {
    "lstm": 5,      # ~5M parameters
    "rnn": 3,       # ~3M parameters
    "tds_cnv": 8,   # ~8M parameters
    "small_transformer": 40  # ~40M parameters
}

# Blue color palette for different models
blue_colors = {
    "lstm": "#1f77b4",       # darker blue
    "rnn": "#7bafd2",        # medium blue
    "tds_cnv": "#c6dbef",    # light blue
    "small_transformer": "#08519c"  # very dark blue
}

# Function to fit power law: y = a * x^b
def power_law(x, a, b):
    """Power law function: y = a * x^b"""
    return a * np.power(x, b)

def find_all_event_files(base_path):
    """Find all TensorBoard event files in the given base path and its subdirectories."""
    event_files = []
    
    # Check if the path exists
    if not os.path.exists(base_path):
        print(f"Path does not exist: {base_path}")
        return event_files
    
    # Look for event files in the base directory and all subdirectories
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.startswith("events.out.tfevents"):
                event_files.append(os.path.join(root, file))
    
    return event_files

def extract_tensorboard_data_from_all_files(base_path, tag):
    """Extract data from all TensorBoard event files in the base path."""
    event_files = find_all_event_files(base_path)
    
    if not event_files:
        print(f"No event files found in {base_path}")
        return None, None
    
    all_steps = []
    all_values = []
    
    for event_file in event_files:
        event_dir = os.path.dirname(event_file)
        event_acc = EventAccumulator(event_dir)
        event_acc.Reload()
        
        available_tags = event_acc.Tags().get('scalars', [])
        
        if tag in available_tags:
            events = event_acc.Scalars(tag)
            steps = [event.step for event in events]
            values = [event.value for event in events]
            
            all_steps.extend(steps)
            all_values.extend(values)
        else:
            # Print available tags for debugging
            print(f"Tag '{tag}' not found in {event_dir}")
            print(f"Available tags: {available_tags}")
    
    if not all_steps:
        return None, None
    
    # Sort by steps to ensure correct ordering
    sorted_data = sorted(zip(all_steps, all_values), key=lambda x: x[0])
    sorted_steps, sorted_values = zip(*sorted_data) if sorted_data else ([], [])
    
    return sorted_steps, sorted_values

def list_all_available_tags(base_path):
    """List all available tags in all event files in the base path."""
    event_files = find_all_event_files(base_path)
    all_tags = set()
    
    for event_file in event_files:
        event_dir = os.path.dirname(event_file)
        event_acc = EventAccumulator(event_dir)
        event_acc.Reload()
        
        tags = event_acc.Tags().get('scalars', [])
        all_tags.update(tags)
    
    return sorted(list(all_tags))

def apply_smoothing(values, window_size):
    """Apply moving average smoothing to a list of values."""
    if len(values) <= window_size:
        return values
    
    smoothed_values = []
    for i in range(len(values)):
        # Calculate window boundaries
        window_start = max(0, i - window_size // 2)
        window_end = min(len(values), i + window_size // 2)
        # Calculate moving average
        window_avg = np.mean(values[window_start:window_end])
        smoothed_values.append(window_avg)
    
    return smoothed_values

def create_paper_scaling_plot():
    """Create a high-quality plot with all models on the same grid, including power law fit."""
    # Create a figure with 1x2 grid (loss and CER)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), dpi=300)
    
    # Dictionary mapping model keys to their base paths
    model_paths = {
        "lstm": lstm_base_path,
        "rnn": rnn_base_path,
        "tds_cnv": tds_cnv_base_path,
        "small_transformer": small_transformer_base_path
    }
    
    # Alternative paths to try if the first ones don't work
    alt_model_paths = {
        "lstm": alt_lstm_base_path,
        "rnn": alt_rnn_base_path,
        "tds_cnv": alt_tds_cnv_base_path,
        "small_transformer": alt_small_transformer_base_path
    }
    
    # Smoothing window size and skip percentage
    smoothing_window = 20
    skip_initial_percent = 5
    
    # Store final metrics for power law fitting
    final_losses = []
    final_cers = []
    model_sizes_list = []
    
    # First plot: Train Loss for all models
    for model_key, model_path in model_paths.items():
        # Check if we need to use alternative path
        available_tags = list_all_available_tags(model_path)
        if not available_tags:
            print(f"No tags found in {model_path}, trying alternative path...")
            model_path = alt_model_paths[model_key]
            available_tags = list_all_available_tags(model_path)
        
        print(f"\nAvailable tags for {model_names[model_key]}:")
        print(available_tags)
        
        # Look for train loss tag
        for tag in ['train/loss', 'train_loss', 'training_loss', 'loss']:
            steps, values = extract_tensorboard_data_from_all_files(model_path, tag)
            if steps and values:
                # Skip initial portion with extremely high values
                if len(steps) > 20:
                    skip_steps = max(1, int(len(steps) * skip_initial_percent / 100))
                    steps = steps[skip_steps:]
                    values = values[skip_steps:]
                
                # Apply smoothing
                smoothed_values = apply_smoothing(values, smoothing_window)
                
                # Plot smoothed data
                ax1.plot(steps, smoothed_values, label=f"{model_names[model_key]}", 
                         linewidth=2, color=blue_colors[model_key])
                print(f"Found and plotted '{tag}' for {model_names[model_key]} with smoothing")
                
                # Store final loss for power law fitting
                final_loss = np.mean(smoothed_values[-20:])  # Average of last 20 points
                final_losses.append(final_loss)
                model_sizes_list.append(model_sizes[model_key])
                print(f"Final loss for {model_names[model_key]}: {final_loss:.4f}")
                break
    
    # Set log scale for loss
    ax1.set_yscale('log')
    
    # Set title and labels
    ax1.set_title('Train Loss', fontsize=14, pad=10)
    ax1.set_xlabel('Steps', fontsize=12, labelpad=8)
    ax1.set_ylabel('Loss (log scale)', fontsize=12, labelpad=8)
    
    # Add grid
    ax1.grid(True, which='major', linestyle='-', linewidth=0.5, color='lightgray', alpha=0.3)
    ax1.grid(True, which='minor', linestyle=':', linewidth=0.5, color='lightgray', alpha=0.2)
    
    # Add legend
    ax1.legend(fontsize=10, loc='upper right', frameon=True, framealpha=0.9, 
              edgecolor='lightgray', fancybox=True)
    
    # Second plot: CER for all models
    for model_key, model_path in model_paths.items():
        # Check if we need to use alternative path
        if not list_all_available_tags(model_path):
            model_path = alt_model_paths[model_key]
        
        # Look for CER tag
        for tag in ['train/CER', 'train_cer', 'training_cer', 'cer', 'CER', 'val/CER', 'test/CER']:
            steps, values = extract_tensorboard_data_from_all_files(model_path, tag)
            if steps and values:
                # Skip initial portion with extremely high values
                if len(steps) > 20:
                    skip_steps = max(1, int(len(steps) * skip_initial_percent / 100))
                    steps = steps[skip_steps:]
                    values = values[skip_steps:]
                
                # Apply smoothing
                smoothed_values = apply_smoothing(values, smoothing_window)
                
                # Plot smoothed data
                ax2.plot(steps, smoothed_values, label=f"{model_names[model_key]}", 
                         linewidth=2, color=blue_colors[model_key])
                print(f"Found and plotted '{tag}' for {model_names[model_key]} with smoothing")
                
                # Store final CER for power law fitting
                final_cer = np.mean(smoothed_values[-20:])  # Average of last 20 points
                final_cers.append(final_cer)
                print(f"Final CER for {model_names[model_key]}: {final_cer:.4f}")
                break
    
    # Set log scale for CER
    ax2.set_yscale('log')
    
    # Set title and labels
    ax2.set_title('Character Error Rate (CER)', fontsize=14, pad=10)
    ax2.set_xlabel('Steps', fontsize=12, labelpad=8)
    ax2.set_ylabel('CER (log scale)', fontsize=12, labelpad=8)
    
    # Add grid
    ax2.grid(True, which='major', linestyle='-', linewidth=0.5, color='lightgray', alpha=0.3)
    ax2.grid(True, which='minor', linestyle=':', linewidth=0.5, color='lightgray', alpha=0.2)
    
    # Add legend
    ax2.legend(fontsize=10, loc='upper right', frameon=True, framealpha=0.9, 
              edgecolor='lightgray', fancybox=True)
    
    # Create a third figure for the power law scaling plot
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 6), dpi=300)
    
    # Plot Loss vs Model Size with power law fit
    if len(model_sizes_list) >= 2 and len(final_losses) >= 2:
        # Sort by model size
        sorted_data = sorted(zip(model_sizes_list, final_losses))
        sorted_sizes, sorted_losses = zip(*sorted_data)
        
        # Plot data points
        ax3.scatter(sorted_sizes, sorted_losses, s=100, color='blue', zorder=5)
        
        # Fit power law
        try:
            # Initial parameter guess
            p0 = [sorted_losses[0], -0.5]  # Initial guess for a and b
            
            # Fit the power law
            params, covariance = optimize.curve_fit(power_law, sorted_sizes, sorted_losses, p0=p0, maxfev=10000)
            a, b = params
            
            # Generate fitted curve
            x_fit = np.linspace(min(sorted_sizes) * 0.8, max(sorted_sizes) * 1.2, 100)
            y_fit = power_law(x_fit, a, b)
            
            # Plot the fitted curve
            ax3.plot(x_fit, y_fit, '--', color='black', linewidth=2,
                     label=f"L(C) = {a:.2e}·C^{b:.3f}")
            
            print(f"Loss power law fit: L(C) = {a:.2e}·C^{b:.3f}")
        except Exception as e:
            print(f"Error fitting power law for loss: {e}")
        
        # Label points with model names
        for i, (size, loss) in enumerate(zip(sorted_sizes, sorted_losses)):
            model_key = next(key for key, val in model_sizes.items() if val == size)
            ax3.annotate(model_names[model_key], 
                         (size, loss),
                         textcoords="offset points",
                         xytext=(0,10), 
                         ha='center')
        
        # Set log scales
        ax3.set_xscale('log')
        ax3.set_yscale('log')
        
        # Set labels and title
        ax3.set_xlabel('Model Size (M parameters)', fontsize=12)
        ax3.set_ylabel('Final Loss', fontsize=12)
        ax3.set_title('Loss vs Model Size', fontsize=14)
        ax3.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.3)
        ax3.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.2)
        ax3.legend(fontsize=10)
    
    # Plot CER vs Model Size with power law fit
    if len(model_sizes_list) >= 2 and len(final_cers) >= 2:
        # Sort by model size
        sorted_data = sorted(zip(model_sizes_list, final_cers))
        sorted_sizes, sorted_cers = zip(*sorted_data)
        
        # Plot data points
        ax4.scatter(sorted_sizes, sorted_cers, s=100, color='red', zorder=5)
        
        # Fit power law
        try:
            # Initial parameter guess
            p0 = [sorted_cers[0], -0.5]  # Initial guess for a and b
            
            # Fit the power law
            params, covariance = optimize.curve_fit(power_law, sorted_sizes, sorted_cers, p0=p0, maxfev=10000)
            a, b = params
            
            # Generate fitted curve
            x_fit = np.linspace(min(sorted_sizes) * 0.8, max(sorted_sizes) * 1.2, 100)
            y_fit = power_law(x_fit, a, b)
            
            # Plot the fitted curve
            ax4.plot(x_fit, y_fit, '--', color='black', linewidth=2,
                     label=f"CER(C) = {a:.2e}·C^{b:.3f}")
            
            print(f"CER power law fit: CER(C) = {a:.2e}·C^{b:.3f}")
        except Exception as e:
            print(f"Error fitting power law for CER: {e}")
        
        # Label points with model names
        for i, (size, cer) in enumerate(zip(sorted_sizes, sorted_cers)):
            model_key = next(key for key, val in model_sizes.items() if val == size)
            ax4.annotate(model_names[model_key], 
                         (size, cer),
                         textcoords="offset points",
                         xytext=(0,10), 
                         ha='center')
        
        # Set log scales
        ax4.set_xscale('log')
        ax4.set_yscale('log')
        
        # Set labels and title
        ax4.set_xlabel('Model Size (M parameters)', fontsize=12)
        ax4.set_ylabel('Final CER', fontsize=12)
        ax4.set_title('CER vs Model Size', fontsize=14)
        ax4.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.3)
        ax4.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.2)
        ax4.legend(fontsize=10)
    
    # Adjust layout for both figures
    fig.tight_layout()
    fig2.tight_layout()
    
    # Save the plots
    fig.savefig(os.path.join(output_dir, "paper_plot_training_curves.png"), dpi=300, bbox_inches='tight')
    fig2.savefig(os.path.join(output_dir, "paper_plot_scaling.png"), dpi=300, bbox_inches='tight')
    
    print(f"\nSaved scaling plots to {output_dir}")
    
    plt.close(fig)
    plt.close(fig2)

def create_grid_plots():
    """Create a 2x4 grid of plots with train loss in first row and train+val CER in second row."""
    # Create a figure with a 2x4 grid
    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(2, 4, figure=fig)
    
    # Dictionary mapping model keys to their base paths
    model_paths = {
        "lstm": lstm_base_path,
        "rnn": rnn_base_path,
        "tds_cnv": tds_cnv_base_path,
        "small_transformer": small_transformer_base_path
    }
    
    # Alternative paths to try if the first ones don't work
    alt_model_paths = {
        "lstm": alt_lstm_base_path,
        "rnn": alt_rnn_base_path,
        "tds_cnv": alt_tds_cnv_base_path,
        "small_transformer": alt_small_transformer_base_path
    }
    
    # Smoothing window size and skip percentage
    smoothing_window = 20
    skip_initial_percent = 5
    
    # For each model, create two plots (train loss and train+val CER)
    for col, (model_key, model_path) in enumerate(model_paths.items()):
        # Check if we need to use alternative path
        available_tags = list_all_available_tags(model_path)
        if not available_tags:
            print(f"No tags found in {model_path}, trying alternative path...")
            model_path = alt_model_paths[model_key]
            available_tags = list_all_available_tags(model_path)
        
        print(f"\nAvailable tags for {model_names[model_key]}:")
        print(available_tags)
        
        # First row: Train Loss plot
        ax1 = fig.add_subplot(gs[0, col])
        
        # Look for train loss tag
        for tag in ['train/loss', 'train_loss', 'training_loss', 'loss']:
            steps, values = extract_tensorboard_data_from_all_files(model_path, tag)
            if steps and values:
                # Skip initial portion with extremely high values
                if len(steps) > 20:
                    skip_steps = max(1, int(len(steps) * skip_initial_percent / 100))
                    steps = steps[skip_steps:]
                    values = values[skip_steps:]
                
                # Apply smoothing
                smoothed_values = apply_smoothing(values, smoothing_window)
                
                # Plot both raw and smoothed data
                ax1.plot(steps, values, alpha=0.3, color=blue_colors["train_raw"], linewidth=1)
                ax1.plot(steps, smoothed_values, label="Train Loss", 
                         linewidth=3, color=blue_colors["train"])
                print(f"Found and plotted '{tag}' for {model_names[model_key]} with smoothing")
                break
        
        # Set log scale for loss
        ax1.set_yscale('log')
        
        # Set title and labels
        ax1.set_title(f'{model_names[model_key]} - Train Loss', fontsize=14, pad=10)
        ax1.set_xlabel('Steps', fontsize=12, labelpad=8)
        ax1.set_ylabel('Loss (log scale)', fontsize=12, labelpad=8)
        
        # Add grid
        ax1.grid(True, which='major', linestyle='-', linewidth=0.5, color='lightgray', alpha=0.3)
        ax1.grid(True, which='minor', linestyle=':', linewidth=0.5, color='lightgray', alpha=0.2)
        
        # Second row: Train + Val CER plot
        ax2 = fig.add_subplot(gs[1, col])
        
        # Look for train CER tag
        train_cer_found = False
        for tag in ['train/CER', 'train_cer', 'training_cer', 'cer', 'CER']:
            steps, values = extract_tensorboard_data_from_all_files(model_path, tag)
            if steps and values:
                # Skip initial portion with extremely high values
                if len(steps) > 20:
                    skip_steps = max(1, int(len(steps) * skip_initial_percent / 100))
                    steps = steps[skip_steps:]
                    values = values[skip_steps:]
                
                # Apply smoothing
                smoothed_values = apply_smoothing(values, smoothing_window)
                
                # Plot both raw and smoothed data
                ax2.plot(steps, values, alpha=0.3, color=blue_colors["train_raw"], linewidth=1)
                ax2.plot(steps, smoothed_values, label="Train CER", 
                         linewidth=3, color=blue_colors["train"])
                print(f"Found and plotted '{tag}' for {model_names[model_key]} with smoothing")
                train_cer_found = True
                break
        
        # Look for validation CER tag
        val_cer_found = False
        for tag in ['val/CER', 'val_cer', 'validation_cer', 'test/CER', 'test_cer']:
            steps, values = extract_tensorboard_data_from_all_files(model_path, tag)
            if steps and values:
                # Skip initial portion if needed
                if len(steps) > 20:
                    skip_steps = max(1, int(len(steps) * skip_initial_percent / 100))
                    steps = steps[skip_steps:]
                    values = values[skip_steps:]
                
                # Apply smoothing
                smoothed_values = apply_smoothing(values, smoothing_window)
                
                # Plot both raw and smoothed data
                ax2.plot(steps, values, alpha=0.3, color=blue_colors["val"], linewidth=1)
                ax2.plot(steps, smoothed_values, label="Val CER", 
                         linewidth=3, color=blue_colors["val"])
                print(f"Found and plotted '{tag}' for {model_names[model_key]} with smoothing")
                val_cer_found = True
                break
        
        # Set log scale for CER
        ax2.set_yscale('log')
        
        # Set title and labels
        ax2.set_title(f'{model_names[model_key]} - Character Error Rate', fontsize=14, pad=10)
        ax2.set_xlabel('Steps', fontsize=12, labelpad=8)
        ax2.set_ylabel('CER (log scale)', fontsize=12, labelpad=8)
        
        # Add legend if we found both train and val CER
        if train_cer_found or val_cer_found:
            ax2.legend(fontsize=10, loc='upper right', frameon=True, framealpha=0.9, 
                      edgecolor='lightgray', fancybox=True)
        
        # Add grid
        ax2.grid(True, which='major', linestyle='-', linewidth=0.5, color='lightgray', alpha=0.3)
        ax2.grid(True, which='minor', linestyle=':', linewidth=0.5, color='lightgray', alpha=0.2)
    
    # Add annotation explaining the plot
    plt.figtext(0.5, 0.01, 
               f"Note: Initial {skip_initial_percent}% of training steps skipped. Smoothing window: {smoothing_window} steps.", 
               ha="center", fontsize=10, bbox={"facecolor":"white", "alpha":0.8, "pad":5})
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    output_filename = "all_models_grid_comparison.png"
    full_path = os.path.join(output_dir, output_filename)
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved grid plot to {full_path}")
    
    plt.close()  # Close the figure to free memory

def create_combined_paper_plot():
    """Create a single high-quality plot with both train loss and CER for all models, with power law fits."""
    # Create a figure with a single plot
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    
    # Dictionary mapping model keys to their base paths
    model_paths = {
        "lstm": lstm_base_path,
        "rnn": rnn_base_path,
        "tds_cnv": tds_cnv_base_path,
        "small_transformer": small_transformer_base_path
    }
    
    # Alternative paths to try if the first ones don't work
    alt_model_paths = {
        "lstm": alt_lstm_base_path,
        "rnn": alt_rnn_base_path,
        "tds_cnv": alt_tds_cnv_base_path,
        "small_transformer": alt_small_transformer_base_path
    }
    
    # Smoothing window size and skip percentage
    smoothing_window = 20
    skip_initial_percent = 5
    
    # Store final metrics for power law fitting
    final_losses = []
    final_cers = []
    model_sizes_list = []
    model_keys_list = []
    
    # Line styles for different metrics
    loss_style = '-'
    cer_style = '--'
    
    # Plot both train loss and CER for all models
    for model_key, model_path in model_paths.items():
        # Check if we need to use alternative path
        available_tags = list_all_available_tags(model_path)
        if not available_tags:
            print(f"No tags found in {model_path}, trying alternative path...")
            model_path = alt_model_paths[model_key]
            available_tags = list_all_available_tags(model_path)
        
        print(f"\nAvailable tags for {model_names[model_key]}:")
        print(available_tags)
        
        # Look for train loss tag
        for tag in ['train/loss', 'train_loss', 'training_loss', 'loss']:
            steps, values = extract_tensorboard_data_from_all_files(model_path, tag)
            if steps and values:
                # Skip initial portion with extremely high values
                if len(steps) > 20:
                    skip_steps = max(1, int(len(steps) * skip_initial_percent / 100))
                    steps = steps[skip_steps:]
                    values = values[skip_steps:]
                
                # Apply smoothing
                smoothed_values = apply_smoothing(values, smoothing_window)
                
                # Plot smoothed data
                ax.plot(steps, smoothed_values, label=f"{model_names[model_key]} Loss", 
                         linestyle=loss_style, linewidth=2, color=blue_colors[model_key])
                print(f"Found and plotted '{tag}' for {model_names[model_key]} with smoothing")
                
                # Store final loss for power law fitting
                final_loss = np.mean(smoothed_values[-20:])  # Average of last 20 points
                final_losses.append(final_loss)
                if model_key not in model_keys_list:
                    model_sizes_list.append(model_sizes[model_key])
                    model_keys_list.append(model_key)
                print(f"Final loss for {model_names[model_key]}: {final_loss:.4f}")
                break
        
        # Look for CER tag
        for tag in ['train/CER', 'train_cer', 'training_cer', 'cer', 'CER', 'val/CER', 'test/CER']:
            steps, values = extract_tensorboard_data_from_all_files(model_path, tag)
            if steps and values:
                # Skip initial portion with extremely high values
                if len(steps) > 20:
                    skip_steps = max(1, int(len(steps) * skip_initial_percent / 100))
                    steps = steps[skip_steps:]
                    values = values[skip_steps:]
                
                # Apply smoothing
                smoothed_values = apply_smoothing(values, smoothing_window)
                
                # Plot smoothed data
                ax.plot(steps, smoothed_values, label=f"{model_names[model_key]} CER", 
                         linestyle=cer_style, linewidth=2, color=blue_colors[model_key])
                print(f"Found and plotted '{tag}' for {model_names[model_key]} with smoothing")
                
                # Store final CER for power law fitting
                final_cer = np.mean(smoothed_values[-20:])  # Average of last 20 points
                final_cers.append(final_cer)
                print(f"Final CER for {model_names[model_key]}: {final_cer:.4f}")
                break
    
    # Set log scale for y-axis
    ax.set_yscale('log')
    
    # Set title and labels
    ax.set_title('Training Metrics for All Models', fontsize=16, pad=15)
    ax.set_xlabel('Steps', fontsize=14, labelpad=10)
    ax.set_ylabel('Metric Value (log scale)', fontsize=14, labelpad=10)
    
    # Add grid
    ax.grid(True, which='major', linestyle='-', linewidth=0.5, color='lightgray', alpha=0.3)
    ax.grid(True, which='minor', linestyle=':', linewidth=0.5, color='lightgray', alpha=0.2)
    
    # Add legend with two columns
    ax.legend(fontsize=10, loc='upper right', frameon=True, framealpha=0.9, 
              edgecolor='lightgray', fancybox=True, ncol=2)
    
    # Create a second figure for the power law scaling plot
    fig2, (ax2_loss, ax2_cer) = plt.subplots(1, 2, figsize=(12, 6), dpi=300)
    
    # Plot both Loss and CER vs Model Size with power law fits
    if len(model_sizes_list) >= 2 and len(final_losses) >= 2 and len(final_cers) >= 2:
        # Sort by model size
        sorted_data = sorted(zip(model_sizes_list, final_losses, final_cers, model_keys_list))
        sorted_sizes, sorted_losses, sorted_cers, sorted_keys = zip(*sorted_data)
        
        # Plot Loss data points
        ax2_loss.scatter(sorted_sizes, sorted_losses, s=100, color='blue', marker='o', label="Loss", zorder=5)
        
        # Fit power law for Loss
        loss_a = None
        loss_b = None
        try:
            # Initial parameter guess
            p0 = [sorted_losses[0], -0.5]  # Initial guess for a and b
            
            # Fit the power law
            params, covariance = optimize.curve_fit(power_law, sorted_sizes, sorted_losses, p0=p0, maxfev=10000)
            loss_a, loss_b = params
            
            # Generate fitted curve
            x_fit = np.linspace(min(sorted_sizes) * 0.8, max(sorted_sizes) * 1.2, 100)
            y_fit = power_law(x_fit, loss_a, loss_b)
            
            # Plot the fitted curve
            ax2_loss.plot(x_fit, y_fit, '-', color='blue', linewidth=2,
                     label=f"L(C) = {loss_a:.2e}·C^{loss_b:.3f}")
            
            print(f"Loss power law fit: L(C) = {loss_a:.2e}·C^{loss_b:.3f}")
        except Exception as e:
            print(f"Error fitting power law for loss: {e}")
        
        # Label points with model names
        for i, (size, loss, key) in enumerate(zip(sorted_sizes, sorted_losses, sorted_keys)):
            # Label for Loss points
            ax2_loss.annotate(model_names[key], 
                         (size, loss),
                         textcoords="offset points",
                         xytext=(0,10), 
                         ha='center',
                         color='blue')
        
        # Set log scales for loss plot
        ax2_loss.set_xscale('log')
        ax2_loss.set_yscale('log')
        
        # Set labels and title for loss plot
        ax2_loss.set_xlabel('Model Size (M parameters)', fontsize=14, labelpad=10)
        ax2_loss.set_ylabel('Final Loss', fontsize=14, labelpad=10)
        ax2_loss.set_title('Loss vs Model Size', fontsize=16, pad=15)
        ax2_loss.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.3)
        ax2_loss.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.2)
        ax2_loss.legend(fontsize=10, loc='upper right')
        
        # Plot CER data points
        ax2_cer.scatter(sorted_sizes, sorted_cers, s=100, color='red', marker='s', label="CER", zorder=5)
        
        # Apply the same power law fit from Loss to CER
        if loss_a is not None and loss_b is not None:
            # Generate fitted curve using loss parameters
            x_fit = np.linspace(min(sorted_sizes) * 0.8, max(sorted_sizes) * 1.2, 100)
            
            # Scale the loss fit to match CER range
            # Find a scaling factor to match the first point
            scaling_factor = sorted_cers[0] / power_law(sorted_sizes[0], loss_a, loss_b)
            
            # Apply the scaled loss fit
            y_fit = scaling_factor * power_law(x_fit, loss_a, loss_b)
            
            # Plot the fitted curve
            ax2_cer.plot(x_fit, y_fit, '-', color='blue', linewidth=2,
                     label=f"Using Loss Exponent: C^{loss_b:.3f}")
            
            print(f"Applied loss power law exponent to CER: CER ~ C^{loss_b:.3f}")
        
        # Also fit a separate power law for CER for comparison
        try:
            # Initial parameter guess
            p0 = [sorted_cers[0], -0.5]  # Initial guess for a and b
            
            # Fit the power law
            params, covariance = optimize.curve_fit(power_law, sorted_sizes, sorted_cers, p0=p0, maxfev=10000)
            cer_a, cer_b = params
            
            # Generate fitted curve
            x_fit = np.linspace(min(sorted_sizes) * 0.8, max(sorted_sizes) * 1.2, 100)
            y_fit = power_law(x_fit, cer_a, cer_b)
            
            # Plot the fitted curve
            ax2_cer.plot(x_fit, y_fit, '--', color='red', linewidth=2,
                     label=f"CER(C) = {cer_a:.2e}·C^{cer_b:.3f}")
            
            print(f"CER power law fit: CER(C) = {cer_a:.2e}·C^{cer_b:.3f}")
        except Exception as e:
            print(f"Error fitting power law for CER: {e}")
        
        # Label points with model names
        for i, (size, cer, key) in enumerate(zip(sorted_sizes, sorted_cers, sorted_keys)):
            # Label for CER points
            ax2_cer.annotate(model_names[key], 
                         (size, cer),
                         textcoords="offset points",
                         xytext=(0,10), 
                         ha='center',
                         color='red')
        
        # Set log scales for CER plot
        ax2_cer.set_xscale('log')
        ax2_cer.set_yscale('log')
        
        # Set labels and title for CER plot
        ax2_cer.set_xlabel('Model Size (M parameters)', fontsize=14, labelpad=10)
        ax2_cer.set_ylabel('Final CER', fontsize=14, labelpad=10)
        ax2_cer.set_title('CER vs Model Size', fontsize=16, pad=15)
        ax2_cer.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.3)
        ax2_cer.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.2)
        ax2_cer.legend(fontsize=10, loc='upper right')
    
    # Adjust layout for both figures
    fig.tight_layout()
    fig2.tight_layout()
    
    # Save the plots
    fig.savefig(os.path.join(output_dir, "paper_plot_combined_metrics.png"), dpi=300, bbox_inches='tight')
    fig2.savefig(os.path.join(output_dir, "paper_plot_scaling_laws.png"), dpi=300, bbox_inches='tight')
    
    print(f"\nSaved combined plots to {output_dir}")
    
    plt.close(fig)
    plt.close(fig2)

def main():
    print(f"Creating output directory: {output_dir}")
    
    # Create the grid of plots
    create_grid_plots()
    
    # Create the paper scaling plot
    create_paper_scaling_plot()
    
    # Create the combined paper plot
    create_combined_paper_plot()
    
    print(f"\nAll outputs have been saved to the '{output_dir}' directory.")

if __name__ == "__main__":
    main() 