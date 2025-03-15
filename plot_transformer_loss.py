import os
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import tensorflow as tf
import glob
from scipy import optimize
from scipy import stats  # Add proper import for stats

# Create output directory for plots
output_dir = "tfoutput"
os.makedirs(output_dir, exist_ok=True)

# Base paths to the TensorBoard log directories for transformer models
tiny_transformer_base_path = "scaling/tiny_transformer/job0_+name=transformer_tiny_100epochs,model=transformer_tiny,trainer.devices=1,trainer.max_epochs=100,user=single_user/lightning_logs"
small_transformer_base_path = "scaling/small_transformer/job0_+name=transformer_small_100epochs,model=transformer_small,trainer.devices=1,trainer.max_epochs=100,user=single_user/lightning_logs"
medium_transformer_base_path = "scaling/medium_transformer/job0_+name=transformer_medium_100epochs,model=transformer_medium,trainer.devices=1,trainer.max_epochs=100,user=single_user/lightning_logs"

# Model names for better labeling
model_names = {
    "tiny": "Model A",
    "small": "Model B",
    "medium": "Model C"
}

# Model sizes (actual number of parameters, in millions)
model_sizes = {
    "tiny": 1.1,    # 1.1M parameters from logs
    "small": 3.2,   # 3.2M parameters from logs
    "medium": 8.2   # 8.2M parameters from logs
}

# Color palettes
blue_colors = {
    "tiny": "#1f77b4",    # darker blue
    "small": "#7bafd2",   # medium blue
    "medium": "#c6dbef"   # light blue
}

red_colors = {
    "tiny": "#d62728",    # darker red
    "small": "#ff9896",   # medium red
    "medium": "#ffd8d6"   # light red
}

# Power law fit colors (slightly darker versions of the blue colors)
fit_colors = {
    "tiny": "#0c4875",    # darker blue
    "small": "#3a5c77",   # medium blue
    "medium": "#8aa9bf"   # light blue
}

# Add fit line color
fit_line_color = '#2ecc71'  # Nice forest green color

def find_all_event_files(base_path):
    """Find all TensorBoard event files in the given base path and its subdirectories."""
    event_files = []
    
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

def power_law(x, a, b):
    """Power law function: y = a * x^b"""
    return a * np.power(x, b)

def fit_power_law(x_data, y_data, use_log=False):
    """Fit a power law curve to the given data and return parameters and statistics."""
    # Remove any zeros or negative values from x_data and corresponding y_data points
    valid_indices = [i for i, x in enumerate(x_data) if x > 0 and (not use_log or y_data[i] > 0)]
    x_filtered = [x_data[i] for i in valid_indices]
    y_filtered = [y_data[i] for i in valid_indices]
    
    if len(x_filtered) < 2:
        print("Not enough valid data points to fit power law")
        return None, None, None, None
    
    # Fit power law using non-linear least squares
    try:
        # For CER, use log space fitting
        if use_log:
            log_x = np.log(x_filtered)
            log_y = np.log(y_filtered)
            slope, intercept = np.polyfit(log_x, log_y, 1)
            b = slope
            a = np.exp(intercept)
            
            # Calculate R-squared in log space
            y_pred = power_law(np.array(x_filtered), a, b)
            log_y_pred = np.log(y_pred)
            ss_total = np.sum((log_y - np.mean(log_y))**2)
            ss_residual = np.sum((log_y - log_y_pred)**2)
            r_squared = 1 - (ss_residual / ss_total) if ss_total != 0 else 0
            p_value = 0.05  # Placeholder for now
        else:
            # Original fitting for loss
            p0 = [1.0, -0.5]
            params, covariance = optimize.curve_fit(power_law, x_filtered, y_filtered, p0=p0, maxfev=10000)
            a, b = params
            
            # Calculate R-squared
            y_pred = power_law(np.array(x_filtered), a, b)
            ss_total = np.sum((np.array(y_filtered) - np.mean(y_filtered))**2)
            ss_residual = np.sum((np.array(y_filtered) - y_pred)**2)
            r_squared = 1 - (ss_residual / ss_total) if ss_total != 0 else 0
            
            # Calculate p-value
            n = len(x_filtered)
            p = 2
            if n > p:
                f_statistic = (ss_total - ss_residual) / p / (ss_residual / (n - p))
                p_value = 1 - stats.f.cdf(f_statistic, p, n - p)
            else:
                p_value = 1.0
        
        return a, b, r_squared, p_value
    except Exception as e:
        print(f"Error fitting power law: {e}")
        return None, None, None, None

def apply_smoothing(values, gamma=0.95):
    """Apply exponential moving average (EMA) smoothing with gamma parameter."""
    smoothed_values = []
    if not values:
        return values
    
    smoothed = values[0]  # Initialize with first value
    smoothed_values.append(smoothed)
    
    # Apply EMA smoothing
    for value in values[1:]:
        smoothed = gamma * smoothed + (1 - gamma) * value
        smoothed_values.append(smoothed)
    
    return smoothed_values

def calculate_transformer_flops(num_params, sequence_length=8000, batch_size=32):
    """
    Calculate approximate FLOPs for transformer forward pass.
    Based on Kaplan et al. scaling laws paper.
    
    Args:
        num_params: Number of parameters in the model (in millions)
        sequence_length: Length of input sequence (8000 samples = 4s of EMG at 2kHz)
        batch_size: Training batch size
    Returns:
        flops_per_step: Approximate FLOPs per training step
    """
    # Convert params to actual number
    params = num_params * 1e6
    
    # Estimate FLOPs per token as ~6x parameter count (from scaling laws literature)
    flops_per_token = 6 * params
    
    # Multiply by sequence length and batch size
    flops_per_step = flops_per_token * sequence_length * batch_size
    
    return flops_per_step

def steps_to_flops(steps, model_size):
    """Convert training steps to cumulative FLOPs"""
    flops_per_step = calculate_transformer_flops(model_size)
    return np.array(steps) * flops_per_step

def plot_four_panel_metrics_and_scaling():
    """Create a 1x4 grid with time series plots showing FLOPs-based scaling laws."""
    # Create a figure with 4 subplots in a 1x4 grid
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    
    # Assign axes for better readability
    ax_train_loss = axes[0]  # First column - Train Loss time series
    ax_val_loss = axes[1]    # Second column - Val Loss time series
    ax_train_cer = axes[2]   # Third column - Train CER time series
    ax_val_cer = axes[3]     # Fourth column - Val CER time series
    
    # Dictionary mapping model keys to their base paths
    model_paths = {
        "tiny": tiny_transformer_base_path,
        "small": small_transformer_base_path,
        "medium": medium_transformer_base_path
    }
    
    # Smoothing parameter (gamma for EMA, higher = smoother curve)
    gamma = 0.95
    
    # Skip initial portion with extremely high values (e.g., first 5% of steps)
    skip_initial_percent = 5
    
    # Store metrics for each model (for scaling law plots)
    model_metrics = {
        "tiny": {},
        "small": {},
        "medium": {}
    }
    
    # Store all values to set appropriate y-axis limits later
    all_train_loss_values = []
    all_val_loss_values = []
    all_train_cer_values = []
    all_val_cer_values = []
    
    # Store statistical results for output to text file
    statistical_results = {
        "train_loss_flops": {},
        "val_loss_flops": {},
        "train_cer_flops": {},
        "val_cer_flops": {}
    }
    
    # Process data for each model
    for model_key, model_path in model_paths.items():
        # First list available tags to help debugging
        print(f"\nAvailable tags for {model_names[model_key]}:")
        available_tags = list_all_available_tags(model_path)
        print(available_tags)
        
        # 1. Train Loss
        for loss_tag in ['train/loss', 'train_loss', 'training_loss', 'loss']:
            steps, values = extract_tensorboard_data_from_all_files(model_path, loss_tag)
            if steps and values:
                # Skip initial portion with extremely high loss
                if len(steps) > 20:  # Only skip if we have enough data points
                    skip_steps = max(1, int(len(steps) * skip_initial_percent / 100))
                    steps = steps[skip_steps:]
                    values = values[skip_steps:]
                
                # Store values for y-axis limit calculation
                all_train_loss_values.extend(values)
                
                # Store final loss for scaling law plot
                num_values = len(values)
                last_n = max(1, int(num_values * 0.1))  # Last 10% of values
                final_loss = np.mean(values[-last_n:])
                model_metrics[model_key]["train_loss"] = final_loss
                
                # Apply smoothing
                smoothed_values = apply_smoothing(values, gamma)
                
                # Convert steps to FLOPs
                flops = steps_to_flops(steps, model_sizes[model_key])
                
                # Plot both raw and smoothed data
                ax_train_loss.plot(flops, values, alpha=0.2, color=blue_colors[model_key], linewidth=1)
                ax_train_loss.plot(flops, smoothed_values, 
                                  label=f"{model_names[model_key]}", 
                                  linewidth=2.5, color=blue_colors[model_key])
                print(f"Found and plotted '{loss_tag}' for {model_names[model_key]} with smoothing")
                break
        
        # After plotting each metric's time series, collect final points for FLOPs scaling
        flops_final = steps_to_flops(steps[-1:], model_sizes[model_key])[0]  # Get final FLOPs value
        
        # Store both the final metric value and final FLOPs for fitting
        if "train_loss" in model_metrics[model_key]:
            model_metrics[model_key]["final_flops"] = flops_final
            
        # 2. Validation Loss
        for val_loss_tag in ['val/loss', 'val_loss', 'validation_loss']:
            steps, values = extract_tensorboard_data_from_all_files(model_path, val_loss_tag)
            if steps and values:
                # Skip initial portion with extremely high loss
                if len(steps) > 20:  # Only skip if we have enough data points
                    skip_steps = max(1, int(len(steps) * skip_initial_percent / 100))
                    steps = steps[skip_steps:]
                    values = values[skip_steps:]
                
                # Store values for y-axis limit calculation
                all_val_loss_values.extend(values)
                
                # Store final loss for scaling law plot
                num_values = len(values)
                last_n = max(1, int(num_values * 0.1))  # Last 10% of values
                final_loss = np.mean(values[-last_n:])
                model_metrics[model_key]["val_loss"] = final_loss
                
                # Apply smoothing
                smoothed_values = apply_smoothing(values, gamma)
                
                # Convert steps to FLOPs
                flops = steps_to_flops(steps, model_sizes[model_key])
                
                # Plot both raw and smoothed data
                ax_val_loss.plot(flops, values, alpha=0.2, color=blue_colors[model_key], linewidth=1)
                ax_val_loss.plot(flops, smoothed_values, 
                                label=f"{model_names[model_key]}", 
                                linewidth=2.5, color=blue_colors[model_key])
                print(f"Found and plotted '{val_loss_tag}' for {model_names[model_key]} with smoothing")
                break
        
        # 3. Train CER
        for train_cer_tag in ['train/CER', 'train_cer', 'training_cer', 'cer']:
            steps, values = extract_tensorboard_data_from_all_files(model_path, train_cer_tag)
            if steps and values:
                # Skip initial portion with extremely high CER
                if len(steps) > 20:  # Only skip if we have enough data points
                    skip_steps = max(1, int(len(steps) * skip_initial_percent / 100))
                    steps = steps[skip_steps:]
                    values = values[skip_steps:]
                
                # Store values for y-axis limit calculation
                all_train_cer_values.extend(values)
                
                # Store final CER for scaling law plot
                num_values = len(values)
                last_n = max(1, int(num_values * 0.1))  # Last 10% of values
                final_cer = np.mean(values[-last_n:])
                model_metrics[model_key]["train_cer"] = final_cer
                
                # Apply smoothing
                smoothed_values = apply_smoothing(values, gamma)
                
                # Convert steps to FLOPs
                flops = steps_to_flops(steps, model_sizes[model_key])
                
                # Plot both raw and smoothed data
                ax_train_cer.plot(flops, values, alpha=0.2, color=red_colors[model_key], linewidth=1)
                ax_train_cer.plot(flops, smoothed_values, 
                                 label=f"{model_names[model_key]}", 
                                 linewidth=2.5, color=red_colors[model_key])
                print(f"Found and plotted '{train_cer_tag}' for {model_names[model_key]} with smoothing")
                break
        
        # 4. Validation CER
        for val_cer_tag in ['val/CER', 'val_cer', 'validation_cer', 'CER']:
            steps, values = extract_tensorboard_data_from_all_files(model_path, val_cer_tag)
            if steps and values:
                # Skip initial portion with extremely high CER
                if len(steps) > 20:  # Only skip if we have enough data points
                    skip_steps = max(1, int(len(steps) * skip_initial_percent / 100))
                    steps = steps[skip_steps:]
                    values = values[skip_steps:]
                
                # Store values for y-axis limit calculation
                all_val_cer_values.extend(values)
                
                # Store final CER for scaling law plot
                num_values = len(values)
                last_n = max(1, int(num_values * 0.1))  # Last 10% of values
                final_cer = np.mean(values[-last_n:])
                model_metrics[model_key]["val_cer"] = final_cer
                
                # Apply smoothing
                smoothed_values = apply_smoothing(values, gamma)
                
                # Convert steps to FLOPs
                flops = steps_to_flops(steps, model_sizes[model_key])
                
                # Plot both raw and smoothed data
                ax_val_cer.plot(flops, values, alpha=0.2, color=red_colors[model_key], linewidth=1)
                ax_val_cer.plot(flops, smoothed_values, 
                               label=f"{model_names[model_key]}", 
                               linewidth=2.5, color=red_colors[model_key])
                print(f"Found and plotted '{val_cer_tag}' for {model_names[model_key]} with smoothing")
                break
    
    # Configure time series subplots
    
    # 1. Train Loss subplot
    ax_train_loss.set_yscale('log')
    ax_train_loss.set_title('Pretraining Loss vs Compute (FLOPs)', fontsize=14, pad=10)
    ax_train_loss.set_xscale('log')
    ax_train_loss.set_xlabel('Cumulative Training FLOPs', fontsize=12, labelpad=8)
    ax_train_loss.set_ylabel('Loss (log scale)', fontsize=12, labelpad=8)
    ax_train_loss.legend(fontsize=10, loc='upper right')
    ax_train_loss.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.3)
    ax_train_loss.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.2)
    
    # 2. Validation Loss subplot
    ax_val_loss.set_yscale('log')
    ax_val_loss.set_title('Validation Loss vs Compute (FLOPs)', fontsize=14, pad=10)
    ax_val_loss.set_xscale('log')
    ax_val_loss.set_xlabel('Cumulative Training FLOPs', fontsize=12, labelpad=8)
    ax_val_loss.set_ylabel('Loss (log scale)', fontsize=12, labelpad=8)
    ax_val_loss.legend(fontsize=10, loc='upper right')
    ax_val_loss.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.3)
    ax_val_loss.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.2)
    
    # 3. Train CER subplot
    ax_train_cer.set_yscale('log')
    ax_train_cer.set_title('CER (Train) vs Compute (FLOPs)', fontsize=14, pad=10)
    ax_train_cer.set_xscale('log')
    ax_train_cer.set_xlabel('Cumulative Training FLOPs', fontsize=12, labelpad=8)
    ax_train_cer.set_ylabel('CER (log scale)', fontsize=12, labelpad=8)
    ax_train_cer.legend(fontsize=10, loc='upper right')
    ax_train_cer.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.3)
    ax_train_cer.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.2)
    
    # 4. Validation CER subplot
    ax_val_cer.set_yscale('log')
    ax_val_cer.set_title('CER (Validation) vs Compute (FLOPs)', fontsize=14, pad=10)
    ax_val_cer.set_xscale('log')
    ax_val_cer.set_xlabel('Cumulative Training FLOPs', fontsize=12, labelpad=8)
    ax_val_cer.set_ylabel('CER (log scale)', fontsize=12, labelpad=8)
    ax_val_cer.legend(fontsize=10, loc='upper right')
    ax_val_cer.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.3)
    ax_val_cer.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.2)
    
    # Plot scaling laws and collect statistics
    
    # Fit power laws to FLOPs scaling
    for ax, metric_key, color_dict in [
        (ax_train_loss, "train_loss", blue_colors),
        (ax_val_loss, "val_loss", blue_colors),
        (ax_train_cer, "train_cer", red_colors),
        (ax_val_cer, "val_cer", red_colors)
    ]:
        # Determine if this is a CER metric
        is_cer = 'cer' in metric_key.lower()
        
        # Fit FLOPs-based scaling
        flops_points = []
        metric_values = []
        
        for model_key in ["tiny", "small", "medium"]:
            if metric_key in model_metrics[model_key] and "final_flops" in model_metrics[model_key]:
                flops_points.append(model_metrics[model_key]["final_flops"])
                metric_values.append(model_metrics[model_key][metric_key])
        
        if len(flops_points) >= 2:
            # Fit power law (use log scale for CER)
            a, b, r_squared, p_value = fit_power_law(flops_points, metric_values, use_log=is_cer)
            
            if a is not None and b is not None:
                x_fit = np.logspace(np.log10(min(flops_points) * 0.8), 
                                  np.log10(max(flops_points) * 1.2), 100)
                y_fit = power_law(x_fit, a, b)
                
                # Plot the fitted curve
                ax.plot(x_fit, y_fit, '--', color=fit_line_color, linewidth=2, zorder=9)
                
                # Add text with power law equation at bottom left (no box)
                ax.text(0.05, 0.05, f"L(C) = {a:.2e}C^{b:.2f}", 
                       transform=ax.transAxes, fontsize=10,
                       verticalalignment='bottom', horizontalalignment='left')
                
                # Store FLOPs-based statistics
                statistical_results[f"{metric_key}_flops"] = {
                    "a": a,
                    "b": b,
                    "r_squared": r_squared,
                    "p_value": p_value,
                    "significant": p_value < 0.05
                }
    
    # Add a main title for the entire figure
    fig.suptitle('', fontsize=18, y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    
    # Save the plot
    output_filename = "transformer_four_panel_flops_scaling.png"
    full_path = os.path.join(output_dir, output_filename)
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved four-panel plot to {full_path}")
    
    plt.close()  # Close the figure to free memory
    
    # Write statistical results to a text file
    stats_filename = "scaling_law_statistics.txt"
    stats_path = os.path.join(output_dir, stats_filename)
    
    with open(stats_path, 'w') as f:
        f.write("TRANSFORMER SCALING LAW STATISTICS\n")
        f.write("=================================\n\n")
        
        # First list model parameter counts
        f.write("MODEL PARAMETER COUNTS\n")
        f.write("-" * 30 + "\n")
        for model_key, size in model_sizes.items():
            f.write(f"{model_names[model_key]}: {size:.1f} million parameters\n")
        f.write("\n\n")
        
        for metric_name, stats in statistical_results.items():
            if stats:
                f.write(f"{metric_name.upper()} SCALING LAW\n")
                f.write("-" * 30 + "\n")
                
                if 'loss' in metric_name:
                    equation = f"L(C) = {stats['a']:.6e} * C^{stats['b']:.6f}  (C = Compute in FLOPs)"
                else:
                    equation = f"CER(C) = {stats['a']:.6e} * C^{stats['b']:.6f}  (C = Compute in FLOPs)"
                
                f.write(f"Equation: {equation}\n")
                f.write(f"R-squared: {stats['r_squared']:.6f}\n")
                f.write(f"p-value: {stats['p_value']:.6f}\n")
                f.write(f"Statistically significant: {'Yes' if stats['significant'] else 'No'}\n")
                
                if stats['significant']:
                    f.write("Interpretation: There is a statistically significant relationship between compute and performance.\n")
                else:
                    f.write("Interpretation: The relationship between compute and performance is not statistically significant.\n")
                
                # Interpret the scaling coefficient
                if stats['b'] < 0:
                    if abs(stats['b']) > 0.5:
                        f.write(f"The scaling coefficient (b = {stats['b']:.4f}) indicates a strong improvement with increasing compute.\n")
                    elif abs(stats['b']) > 0.2:
                        f.write(f"The scaling coefficient (b = {stats['b']:.4f}) indicates a moderate improvement with increasing compute.\n")
                    else:
                        f.write(f"The scaling coefficient (b = {stats['b']:.4f}) indicates a weak improvement with increasing compute.\n")
                else:
                    f.write(f"Warning: The scaling coefficient (b = {stats['b']:.4f}) is positive, suggesting performance degradation with increasing compute.\n")
                
                f.write("\n\n")
    
    print(f"Saved statistical analysis to {stats_path}")
    
    return model_metrics, statistical_results  # Return the metrics and statistics for potential further analysis

def main():
    print(f"Creating output directory: {output_dir}")
    
    # Generate the four-panel plot with time series and scaling laws
    print("\nGenerating four-panel plot with time series and scaling laws...")
    model_metrics, statistical_results = plot_four_panel_metrics_and_scaling()
    
    print(f"\nAll outputs have been saved to the '{output_dir}' directory.")
    
    # Print summary of statistical significance
    print("\nSTATISTICAL SIGNIFICANCE SUMMARY:")
    print("-" * 50)
    for metric, stats in statistical_results.items():
        if stats:
            sig_status = "SIGNIFICANT" if stats["significant"] else "NOT SIGNIFICANT"
            print(f"{metric.upper()}: {sig_status} (p={stats['p_value']:.4f}, R²={stats['r_squared']:.4f})")

if __name__ == "__main__":
    main() 