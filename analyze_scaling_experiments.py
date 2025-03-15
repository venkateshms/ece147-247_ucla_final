#!/usr/bin/env python
# Analyze results from transformer scaling experiments

import os
import re
import glob
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Model sizes (parameters in millions, approximate)
MODEL_PARAMS = {
    'transformer_tiny': 0.5,   # ~0.5M parameters
    'transformer_small': 2.0,  # ~2M parameters
    'transformer_medium': 5.0, # ~5M parameters
    'transformer_large': 15.0, # ~15M parameters
    'transformer_xlarge': 35.0 # ~35M parameters
}

def extract_metrics(results_dir):
    """Extract metrics from the experiment results."""
    metrics = defaultdict(dict)
    
    for model_name in MODEL_PARAMS.keys():
        model_dir = os.path.join(results_dir, model_name)
        if not os.path.exists(model_dir):
            print(f"Warning: No results found for {model_name}")
            continue
        
        # Find metrics file
        metrics_files = glob.glob(os.path.join(model_dir, "lightning_logs/version_*/metrics.csv"))
        if not metrics_files:
            print(f"Warning: No metrics file found for {model_name}")
            continue
        
        # Use the most recent version
        metrics_file = sorted(metrics_files)[-1]
        
        # Parse metrics file to get the final test CER and WER
        try:
            with open(metrics_file, 'r') as f:
                lines = f.readlines()
                headers = lines[0].strip().split(',')
                values = lines[-1].strip().split(',')
                
                metrics_dict = {headers[i]: values[i] for i in range(len(headers))}
                
                # Extract test CER and WER if available
                for key, value in metrics_dict.items():
                    if 'test/cer' in key:
                        metrics[model_name]['cer'] = float(value)
                    if 'test/wer' in key:
                        metrics[model_name]['wer'] = float(value)
                    if 'test/loss' in key:
                        metrics[model_name]['loss'] = float(value)
                
                # Also record model size
                metrics[model_name]['params'] = MODEL_PARAMS[model_name]
        except Exception as e:
            print(f"Error processing metrics for {model_name}: {e}")
    
    return metrics

def plot_scaling_curves(metrics, output_dir):
    """Plot scaling curves based on the metrics."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data for plotting
    models = []
    params = []
    cer_values = []
    wer_values = []
    loss_values = []
    
    for model_name, model_metrics in metrics.items():
        if 'cer' in model_metrics and 'wer' in model_metrics:
            models.append(model_name)
            params.append(model_metrics['params'])
            cer_values.append(model_metrics['cer'])
            wer_values.append(model_metrics['wer'])
            if 'loss' in model_metrics:
                loss_values.append(model_metrics['loss'])
    
    # Sort by model size
    sorted_indices = np.argsort(params)
    models = [models[i] for i in sorted_indices]
    params = [params[i] for i in sorted_indices]
    cer_values = [cer_values[i] for i in sorted_indices]
    wer_values = [wer_values[i] for i in sorted_indices]
    loss_values = [loss_values[i] for i in sorted_indices if i < len(loss_values)]
    
    # Plot CER vs model size
    plt.figure(figsize=(10, 6))
    plt.plot(params, cer_values, 'o-', linewidth=2, markersize=8)
    plt.xscale('log')
    plt.xlabel('Model Parameters (millions)')
    plt.ylabel('Character Error Rate (CER)')
    plt.title('Scaling Law: CER vs Model Size')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    for i, model in enumerate(models):
        plt.annotate(model.replace('transformer_', ''), 
                    (params[i], cer_values[i]),
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scaling_cer.png'))
    
    # Plot WER vs model size
    plt.figure(figsize=(10, 6))
    plt.plot(params, wer_values, 'o-', linewidth=2, markersize=8)
    plt.xscale('log')
    plt.xlabel('Model Parameters (millions)')
    plt.ylabel('Word Error Rate (WER)')
    plt.title('Scaling Law: WER vs Model Size')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    for i, model in enumerate(models):
        plt.annotate(model.replace('transformer_', ''), 
                    (params[i], wer_values[i]),
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scaling_wer.png'))
    
    # Plot Loss vs model size if available
    if loss_values:
        plt.figure(figsize=(10, 6))
        plt.plot(params[:len(loss_values)], loss_values, 'o-', linewidth=2, markersize=8)
        plt.xscale('log')
        plt.xlabel('Model Parameters (millions)')
        plt.ylabel('Test Loss')
        plt.title('Scaling Law: Loss vs Model Size')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        for i, model in enumerate(models[:len(loss_values)]):
            plt.annotate(model.replace('transformer_', ''), 
                        (params[i], loss_values[i]),
                        textcoords="offset points", 
                        xytext=(0,10), 
                        ha='center')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'scaling_loss.png'))
    
    # Save metrics as JSON for further analysis
    scaling_data = {
        'models': models,
        'params': params,
        'cer': cer_values,
        'wer': wer_values,
        'loss': loss_values if loss_values else None
    }
    
    with open(os.path.join(output_dir, 'scaling_results.json'), 'w') as f:
        json.dump(scaling_data, f, indent=2)
    
    print(f"Scaling curves saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Analyze transformer scaling experiments")
    parser.add_argument('--results-dir', default='scaling_results', help='Directory containing experiment results')
    parser.add_argument('--output-dir', default='scaling_analysis', help='Directory to save analysis results')
    args = parser.parse_args()
    
    metrics = extract_metrics(args.results_dir)
    if metrics:
        plot_scaling_curves(metrics, args.output_dir)
    else:
        print("No metrics found to analyze.")

if __name__ == "__main__":
    main() 