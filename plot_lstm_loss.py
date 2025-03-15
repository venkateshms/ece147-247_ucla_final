import os
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import tensorflow as tf
import glob
from scipy import optimize
from scipy import stats

# Create output directory for plots
output_dir = "tfoutput"
os.makedirs(output_dir, exist_ok=True)

# Base paths to the TensorBoard log directories for LSTM models
default_lstm_base_path = "/Users/madhavanvenkatesh/Desktop/ece_paper/scaling/lstm_scaling/default/lightning_logs"
small_lstm_base_path = "/Users/madhavanvenkatesh/Desktop/ece_paper/scaling/lstm_scaling/small/lightning_logs"
medium_lstm_base_path = "/Users/madhavanvenkatesh/Desktop/ece_paper/scaling/lstm_scaling/medium/lightning_logs"

# Model names for internal mapping (not used for legend anymore)
model_names = {
    "default": "B",
    "small": "C",
    "medium": "A"
}

# Model sizes (in millions)
model_sizes = {
    "default": 4.9,   # 4.9M parameters
    "small": 1.8,     # 1.8M parameters
    "medium": 12.2    # 12.2M parameters
}

# Create legend labels that show parameter counts
legend_labels = { key: f"{model_sizes[key]}M Parameters" for key in model_sizes }

# Color palettes for loss metrics using oranges
orange_colors = {
    "small": "#d35400",    # For small (C)
    "default": "#e67e22",  # For default (B)
    "medium": "#f7c07e"    # For medium (A)
}

# Reds for CER metrics
red_colors = {
    "default": "#d62728",
    "small": "#ff9896",
    "medium": "#ffd8d6"
}

# Fit line color remains forest green
fit_line_color = '#2ecc71'

def find_all_event_files(base_path):
    event_files = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.startswith("events.out.tfevents"):
                event_files.append(os.path.join(root, file))
    if event_files:
        print(f"Found {len(event_files)} event files in {base_path}")
        for file in event_files:
            print(f"  - {file}")
    else:
        print(f"No event files found in {base_path}")
    return event_files

def extract_tensorboard_data_from_all_files(base_path, tag):
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
            print(f"Tag '{tag}' not found in {event_dir}")
            print(f"Available tags: {available_tags}")
    if not all_steps:
        return None, None
    sorted_data = sorted(zip(all_steps, all_values), key=lambda x: x[0])
    sorted_steps, sorted_values = zip(*sorted_data) if sorted_data else ([], [])
    return sorted_steps, sorted_values

def list_all_available_tags(base_path):
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
    try:
        if np.any(x > 1e10) or abs(b) > 10:
            return np.exp(np.log(a) + b * np.log(x))
        else:
            return a * np.power(x, b)
    except:
        return np.exp(np.log(a) + b * np.log(x))

def fit_power_law(x_data, y_data, use_log=False):
    valid_indices = [i for i, x in enumerate(x_data) if x > 0 and (not use_log or y_data[i] > 0)]
    x_filtered = [x_data[i] for i in valid_indices]
    y_filtered = [y_data[i] for i in valid_indices]
    if len(x_filtered) < 2:
        print("Not enough valid data points to fit power law")
        return None, None, None, None
    try:
        if use_log:
            log_x = np.log(x_filtered)
            log_y = np.log(y_filtered)
            slope, intercept = np.polyfit(log_x, log_y, 1)
            b = slope
            a = np.exp(intercept)
            y_pred = power_law(np.array(x_filtered), a, b)
            log_y_pred = np.log(y_pred)
            ss_total = np.sum((log_y - np.mean(log_y))**2)
            ss_residual = np.sum((log_y - log_y_pred)**2)
            r_squared = 1 - (ss_residual / ss_total) if ss_total != 0 else 0
            p_value = 0.05  # Placeholder
        else:
            p0 = [1.0, -0.5]
            params, covariance = optimize.curve_fit(power_law, x_filtered, y_filtered, p0=p0, maxfev=10000)
            a, b = params
            y_pred = power_law(np.array(x_filtered), a, b)
            ss_total = np.sum((np.array(y_filtered) - np.mean(y_filtered))**2)
            ss_residual = np.sum((np.array(y_filtered) - y_pred)**2)
            r_squared = 1 - (ss_residual / ss_total) if ss_total != 0 else 0
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

def apply_smoothing(values, gamma=0.55):
    smoothed_values = []
    if not values:
        return values
    smoothed = values[0]
    smoothed_values.append(smoothed)
    for value in values[1:]:
        smoothed = gamma * smoothed + (1 - gamma) * value
        smoothed_values.append(smoothed)
    return smoothed_values

def calculate_lstm_flops(num_params, sequence_length=8000, batch_size=32):
    params = num_params * 1e6
    flops_per_token = 4 * params
    flops_per_step = flops_per_token * sequence_length * batch_size
    return flops_per_step

def steps_to_flops(steps, model_size):
    flops_per_step = calculate_lstm_flops(model_size)
    return np.array(steps) * flops_per_step

def reorder_legend(ax, model_order, legend_labels):
    handles, labels = ax.get_legend_handles_labels()
    ordered_handles = []
    ordered_labels = []
    for key in model_order:
        label = legend_labels[key]
        for h, l in zip(handles, labels):
            if l == label:
                ordered_handles.append(h)
                ordered_labels.append(l)
                break
    ax.legend(ordered_handles, ordered_labels, fontsize=10, loc='upper right')

def plot_four_panel_metrics_and_scaling():
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    ax_train_loss = axes[0]
    ax_val_loss = axes[1]
    ax_train_cer = axes[2]
    ax_val_cer = axes[3]
    
    # New mapping: "medium" is A (12.2M), "default" is B (4.9M), "small" is C (1.8M)
    model_order = ["medium", "default", "small"]
    model_paths = {
        "default": default_lstm_base_path,
        "small": small_lstm_base_path,
        "medium": medium_lstm_base_path
    }
    
    gamma = 0.95
    skip_initial_percent = 5
    model_metrics = { "default": {}, "small": {}, "medium": {} }
    statistical_results = {
        "train_loss_flops": {},
        "val_loss_flops": {},
        "train_cer_flops": {},
        "val_cer_flops": {}
    }
    
    for model_key in model_order:
        model_path = model_paths[model_key]
        print(f"\nAvailable tags for {model_names[model_key]}:")
        available_tags = list_all_available_tags(model_path)
        print(available_tags)
        found_valid_data = False
        
        # 1. Train Loss using orange_colors
        for loss_tag in ['train/loss', 'train_loss', 'training_loss', 'loss', 'test/loss']:
            steps, values = extract_tensorboard_data_from_all_files(model_path, loss_tag)
            if steps and values:
                if len(steps) > 20:
                    skip_steps = max(1, int(len(steps) * skip_initial_percent / 100))
                    steps = steps[skip_steps:]
                    values = values[skip_steps:]
                num_values = len(values)
                last_n = max(1, int(num_values * 0.1))
                final_loss = np.mean(values[-last_n:])
                model_metrics[model_key]["train_loss"] = final_loss
                smoothed_values = apply_smoothing(values, gamma)
                flops = steps_to_flops(steps, model_sizes[model_key])
                ax_train_loss.plot(flops, values, alpha=0.2, color=orange_colors[model_key], linewidth=1)
                ax_train_loss.plot(flops, smoothed_values, 
                                   label=f"{legend_labels[model_key]}",
                                   linewidth=2.5, color=orange_colors[model_key])
                print(f"Found and plotted '{loss_tag}' for {legend_labels[model_key]} with smoothing")
                found_valid_data = True
                break
        
        if found_valid_data and steps is not None and len(steps) > 0:
            flops_final = steps_to_flops(steps[-1:], model_sizes[model_key])[0]
            model_metrics[model_key]["final_flops"] = flops_final
        else:
            print(f"No valid data found for {legend_labels[model_key]} train_loss, skipping FLOPs calculation")
        
        # 2. Validation Loss using orange_colors
        val_steps = None
        for val_loss_tag in ['val/loss', 'val_loss', 'validation_loss', 'test/loss']:
            val_steps, values = extract_tensorboard_data_from_all_files(model_path, val_loss_tag)
            if val_steps and values:
                if len(val_steps) > 20:
                    skip_steps = max(1, int(len(val_steps) * skip_initial_percent / 100))
                    val_steps = val_steps[skip_steps:]
                    values = values[skip_steps:]
                num_values = len(values)
                last_n = max(1, int(num_values * 0.1))
                final_loss = np.mean(values[-last_n:])
                model_metrics[model_key]["val_loss"] = final_loss
                smoothed_values = apply_smoothing(values, gamma)
                flops = steps_to_flops(val_steps, model_sizes[model_key])
                ax_val_loss.plot(flops, values, alpha=0.2, color=orange_colors[model_key], linewidth=1)
                ax_val_loss.plot(flops, smoothed_values, 
                                 label=f"{legend_labels[model_key]}",
                                 linewidth=2.5, color=orange_colors[model_key])
                print(f"Found and plotted '{val_loss_tag}' for {legend_labels[model_key]} with smoothing")
                break
        
        # 3. Train CER using red_colors
        train_cer_steps = None
        for train_cer_tag in ['train/CER', 'train_cer', 'training_cer', 'cer', 'test/CER']:
            train_cer_steps, values = extract_tensorboard_data_from_all_files(model_path, train_cer_tag)
            if train_cer_steps and values:
                if len(train_cer_steps) > 20:
                    skip_steps = max(1, int(len(train_cer_steps) * skip_initial_percent / 100))
                    train_cer_steps = train_cer_steps[skip_steps:]
                    values = values[skip_steps:]
                num_values = len(values)
                last_n = max(1, int(num_values * 0.1))
                final_cer = np.mean(values[-last_n:])
                model_metrics[model_key]["train_cer"] = final_cer
                smoothed_values = apply_smoothing(values, gamma)
                flops = steps_to_flops(train_cer_steps, model_sizes[model_key])
                ax_train_cer.plot(flops, values, alpha=0.2, color=red_colors[model_key], linewidth=1)
                ax_train_cer.plot(flops, smoothed_values, 
                                  label=f"{legend_labels[model_key]}",
                                  linewidth=2.5, color=red_colors[model_key])
                print(f"Found and plotted '{train_cer_tag}' for {legend_labels[model_key]} with smoothing")
                break
        
        # 4. Validation CER using red_colors
        val_cer_steps = None
        for val_cer_tag in ['val/CER', 'val_cer', 'validation_cer', 'CER', 'test/CER']:
            val_cer_steps, values = extract_tensorboard_data_from_all_files(model_path, val_cer_tag)
            if val_cer_steps and values:
                if len(val_cer_steps) > 20:
                    skip_steps = max(1, int(len(val_cer_steps) * skip_initial_percent / 100))
                    val_cer_steps = val_cer_steps[skip_steps:]
                    values = values[skip_steps:]
                num_values = len(values)
                last_n = max(1, int(num_values * 0.1))
                final_cer = np.mean(values[-last_n:])
                model_metrics[model_key]["val_cer"] = final_cer
                smoothed_values = apply_smoothing(values, gamma)
                flops = steps_to_flops(val_cer_steps, model_sizes[model_key])
                ax_val_cer.plot(flops, values, alpha=0.2, color=red_colors[model_key], linewidth=1)
                ax_val_cer.plot(flops, smoothed_values, 
                                label=f"{legend_labels[model_key]}",
                                linewidth=2.5, color=red_colors[model_key])
                print(f"Found and plotted '{val_cer_tag}' for {legend_labels[model_key]} with smoothing")
                break
    
    # Configure subplots and reorder legends to ensure order A, B, C (parameter counts order)
    ax_train_loss.set_yscale('log')
    ax_train_loss.set_title('Pretraining Loss vs Compute (FLOPs)', fontsize=14, pad=10)
    ax_train_loss.set_xscale('log')
    ax_train_loss.set_xlabel('Cumulative Training FLOPs', fontsize=12, labelpad=8)
    ax_train_loss.set_ylabel('Loss (log scale)', fontsize=12, labelpad=8)
    ax_train_loss.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.3)
    ax_train_loss.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.2)
    reorder_legend(ax_train_loss, model_order, legend_labels)
    
    ax_val_loss.set_yscale('log')
    ax_val_loss.set_title('Validation Loss vs Compute (FLOPs)', fontsize=14, pad=10)
    ax_val_loss.set_xscale('log')
    ax_val_loss.set_xlabel('Cumulative Training FLOPs', fontsize=12, labelpad=8)
    ax_val_loss.set_ylabel('Loss (log scale)', fontsize=12, labelpad=8)
    ax_val_loss.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.3)
    ax_val_loss.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.2)
    reorder_legend(ax_val_loss, model_order, legend_labels)
    
    ax_train_cer.set_yscale('log')
    ax_train_cer.set_title('CER (Train) vs Compute (FLOPs)', fontsize=14, pad=10)
    ax_train_cer.set_xscale('log')
    ax_train_cer.set_xlabel('Cumulative Training FLOPs', fontsize=12, labelpad=8)
    ax_train_cer.set_ylabel('CER (log scale)', fontsize=12, labelpad=8)
    ax_train_cer.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.3)
    ax_train_cer.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.2)
    reorder_legend(ax_train_cer, model_order, legend_labels)
    
    ax_val_cer.set_yscale('log')
    ax_val_cer.set_title('CER (Validation) vs Compute (FLOPs)', fontsize=14, pad=10)
    ax_val_cer.set_xscale('log')
    ax_val_cer.set_xlabel('Cumulative Training FLOPs', fontsize=12, labelpad=8)
    ax_val_cer.set_ylabel('CER (log scale)', fontsize=12, labelpad=8)
    ax_val_cer.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.3)
    ax_val_cer.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.2)
    reorder_legend(ax_val_cer, model_order, legend_labels)
    
    # Keep the scaling-law annotation (fitted curve equation) as before.
    for ax, metric_key, color_dict in [
        (ax_train_loss, "train_loss", orange_colors),
        (ax_val_loss, "val_loss", orange_colors),
        (ax_train_cer, "train_cer", red_colors),
        (ax_val_cer, "val_cer", red_colors)
    ]:
        is_cer = 'cer' in metric_key.lower()
        flops_points = []
        metric_values = []
        for model_key in model_order:
            if metric_key in model_metrics[model_key] and "final_flops" in model_metrics[model_key]:
                flops_points.append(model_metrics[model_key]["final_flops"])
                metric_values.append(model_metrics[model_key][metric_key])
        if len(flops_points) >= 2:
            a, b, r_squared, p_value = fit_power_law(flops_points, metric_values, use_log=is_cer)
            if a is not None and b is not None:
                x_fit = np.logspace(np.log10(min(flops_points) * 0.8),
                                    np.log10(max(flops_points) * 1.2), 100)
                y_fit = power_law(x_fit, a, b)
                ax.plot(x_fit, y_fit, '--', color=fit_line_color, linewidth=2, zorder=9)
                # The scaling-law annotation (fitted equation) remains as before.
                ax.text(0.05, 0.05, f"L(C) = {a:.2e} * C^{b:.2f}", 
                        transform=ax.transAxes, fontsize=10,
                        verticalalignment='bottom', horizontalalignment='left')
                statistical_results[f"{metric_key}_flops"] = {
                    "a": a,
                    "b": b,
                    "r_squared": r_squared,
                    "p_value": p_value,
                    "significant": p_value < 0.05
                }
        else:
            print(f"\nWARNING: Not enough data points to fit {metric_key} scaling law.")
            ax.text(0.5, 0.5, "Insufficient data\nto fit scaling law",
                    transform=ax.transAxes, fontsize=12,
                    verticalalignment='center', horizontalalignment='center',
                    bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5))
    
    fig.suptitle('', fontsize=18, y=0.98)
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    output_filename = "lstm_four_panel_flops_scaling.png"
    full_path = os.path.join(output_dir, output_filename)
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved four-panel plot to {full_path}")
    plt.close()
    
    stats_filename = "lstm_scaling_law_statistics.txt"
    stats_path = os.path.join(output_dir, stats_filename)
    with open(stats_path, 'w') as f:
        f.write("LSTM SCALING LAW STATISTICS\n")
        f.write("=================================\n\n")
        f.write("MODEL PARAMETER COUNTS\n")
        f.write("-" * 30 + "\n")
        for model_key, size in model_sizes.items():
            f.write(f"{legend_labels[model_key]}\n")
        f.write("\n\n")
        for metric_name, stats in statistical_results.items():
            if stats:
                f.write(f"{metric_name.upper()} SCALING LAW\n")
                f.write("-" * 30 + "\n")
                equation = (f"L(C) = {stats['a']:.6e} * C^{stats['b']:.6f}"
                            if 'loss' in metric_name else
                            f"CER(C) = {stats['a']:.6e} * C^{stats['b']:.6f}")
                f.write(f"Equation: {equation}\n")
                f.write(f"R-squared: {stats['r_squared']:.6f}\n")
                f.write(f"p-value: {stats['p_value']:.6f}\n")
                f.write(f"Statistically significant: {'Yes' if stats['significant'] else 'No'}\n\n")
    print(f"Saved statistical analysis to {stats_path}")
    return model_metrics, statistical_results

def main():
    print(f"Creating output directory: {output_dir}")
    print("\nGenerating four-panel plot with time series and scaling laws...")
    model_metrics, statistical_results = plot_four_panel_metrics_and_scaling()
    print(f"\nAll outputs have been saved to the '{output_dir}' directory.")
    print("\nSTATISTICAL SIGNIFICANCE SUMMARY:")
    print("-" * 50)
    for metric, stats in statistical_results.items():
        if stats:
            sig_status = "SIGNIFICANT" if stats["significant"] else "NOT SIGNIFICANT"
            print(f"{metric.upper()}: {sig_status} (p={stats['p_value']:.4f}, R²={stats['r_squared']:.4f})")

if __name__ == "__main__":
    main()
