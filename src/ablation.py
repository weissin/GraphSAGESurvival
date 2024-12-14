import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from sklearn.metrics.pairwise import cosine_similarity
from statistics import mean, stdev
from tqdm import tqdm
import matplotlib.pyplot as plt
from graphsage import GraphSAGEModel
from cnn_encoder import PreprocessedPatientDataset
from cnn_encoder import SurvivalPredictionModel

def create_results_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_metrics(model, dataloader, device):
    model.eval()
    total_mse = 0.0
    total_mae = 0.0
    total_relative_mae = 0.0
    total_samples = 0
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for images, tabulars, targets in dataloader:
            images = images.to(device)
            tabulars = tabulars.to(device)
            targets = targets.to(device)

            outputs = model(images, tabulars)
            predictions = outputs.squeeze()
            targets = targets.squeeze()

            mse = ((predictions - targets) ** 2).mean().item()
            mae = torch.abs(predictions - targets).mean().item()

            nonzero_mask = targets != 0
            if nonzero_mask.sum() > 0:
                relative_mae = (torch.abs(predictions[nonzero_mask] - targets[nonzero_mask]) / targets[nonzero_mask]).mean().item() * 100
            else:
                relative_mae = 0.0

            batch_size = images.size(0)
            total_mse += mse * batch_size
            total_mae += mae * batch_size
            total_relative_mae += relative_mae * batch_size
            total_samples += batch_size

            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    mean_mse = total_mse / total_samples
    mean_mae = total_mae / total_samples
    mean_relative_mae = total_relative_mae / total_samples
    return mean_mse, mean_mae, mean_relative_mae, all_predictions, all_targets

def compute_mean_variance_survival(dataset):
    survival_lengths = []
    for idx in range(len(dataset)):
        _, _, survival_length = dataset[idx]
        survival_lengths.append(survival_length.item())
    survival_lengths = np.array(survival_lengths)
    mean_survival = np.mean(survival_lengths)
    variance_survival = np.var(survival_lengths)
    return mean_survival, variance_survival

def plot_train_val_curves(train_losses, val_losses, figure_path):
    plt.figure(figsize=(8,6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('GNN Training/Validation Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path)
    plt.close()

def save_results_to_csv(results_dict, csv_path):
    df = pd.DataFrame(results_dict)
    df.to_csv(csv_path, index=False)

def train_one_epoch_gnn(model, optimizer, criterion, data):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate_gnn(model, criterion, data, mask):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        loss = criterion(out[mask], data.y[mask])
    return loss.item()

def compute_metrics_gnn(model, data, mask, device):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        predictions = out[mask].squeeze()
        targets = data.y[mask].squeeze()
        predictions = predictions.cpu()
        targets = targets.cpu()

        mse = ((predictions - targets) ** 2).mean().item()
        mae = torch.abs(predictions - targets).mean().item()

        nonzero_mask = targets != 0
        if nonzero_mask.sum() > 0:
            relative_mae = (torch.abs(predictions[nonzero_mask] - targets[nonzero_mask]) / targets[nonzero_mask]).mean().item() * 100
        else:
            relative_mae = 0.0
    return mse, mae, relative_mae, predictions.numpy(), targets.numpy()

def run_gnn_experiment(data, lr, hidden_channels, device, epochs=100, seeds=[0,1,2]):
    """
    Run the GNN training multiple times (for different seeds) given hyperparams.
    Return mean/std for test MSE, MAE, Relative MAE.
    """
    criterion = nn.MSELoss()
    test_mse_list = []
    test_mae_list = []
    test_rmae_list = []

    # For the first seed, store training curves
    first_seed_train_losses = []
    first_seed_val_losses = []

    for run_idx, seed in enumerate(seeds):
        set_seed(seed)
        model = GraphSAGEModel(in_channels=data.x.size(1),
                               hidden_channels=hidden_channels,
                               out_channels=1).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        best_val_loss = float('inf')
        best_model_weights = None

        train_losses = []
        val_losses = []

        for epoch in range(1, epochs+1):
            train_loss = train_one_epoch_gnn(model, optimizer, criterion, data)
            val_loss = evaluate_gnn(model, criterion, data, data.val_mask)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_weights = model.state_dict()

        # Load best weights
        model.load_state_dict(best_model_weights)
        test_mse, test_mae, test_rmae, _, _ = compute_metrics_gnn(model, data, data.test_mask, device)
        test_mse_list.append(test_mse)
        test_mae_list.append(test_mae)
        test_rmae_list.append(test_rmae)

        # Store the curves from the first run only for plotting
        if run_idx == 0:
            first_seed_train_losses = train_losses
            first_seed_val_losses = val_losses

    # Compute mean/std
    mse_mean, mse_std = mean(test_mse_list), stdev(test_mse_list)
    mae_mean, mae_std = mean(test_mae_list), stdev(test_mae_list)
    rmae_mean, rmae_std = mean(test_rmae_list), stdev(test_rmae_list)

    return (mse_mean, mse_std, mae_mean, mae_std, rmae_mean, rmae_std,
            first_seed_train_losses, first_seed_val_losses)

def extract_node_features(processed_dataset, cnn_model, device, ablation='none'):
    """
    Extract node features based on the ablation setting.

    Parameters:
    - processed_dataset: The dataset containing all samples.
    - cnn_model: The pre-trained CNN model to extract image features.
    - device: The device to perform computations on.
    - ablation: Type of ablation ('none', 'image_only', 'tabular_only').

    Returns:
    - node_features: Tensor of extracted node features.
    - y: Tensor of target survival lengths.
    """
    cnn_model.eval()
    all_features = []
    all_survival = []

    full_loader = DataLoader(processed_dataset, batch_size=8, shuffle=False, num_workers=8)
    with torch.no_grad():
        for (image_tensor, tabular_tensor, survival_length) in tqdm(full_loader, desc="Extracting node features"):
            image_tensor = image_tensor.to(device)
            tabular_tensor = tabular_tensor.to(device)
            
            if ablation in ['none', 'image_only', 'image_tabular']:
                # Extract image features
                img_feat = cnn_model.image_feature_extractor(image_tensor)  # [B,256,D',H',W']
                img_feat = cnn_model.image_pool(img_feat)                   # [B, 256, 1, 1, 1]
                img_feat = cnn_model.flatten(img_feat)                      # [B, 256]
            if ablation in ['none', 'tabular_only', 'image_tabular']:
                # Extract tabular features
                tab_feat = cnn_model.tabular_feature_extractor(tabular_tensor)  # [B, 32]
            
            if ablation == 'none' or ablation == 'image_tabular':
                # Combine both features
                node_feat = torch.cat([img_feat, tab_feat], dim=1)  # [B, 288]
            elif ablation == 'image_only':
                node_feat = img_feat  # [B, 256]
            elif ablation == 'tabular_only':
                node_feat = tab_feat  # [B, 32]
            
            all_features.append(node_feat.cpu().numpy())
            all_survival.extend(survival_length.numpy())
    
    all_features = np.vstack(all_features).astype(np.float32)
    all_survival = np.array(all_survival, dtype=np.float32)
    node_features = torch.tensor(all_features, dtype=torch.float)
    y = torch.tensor(all_survival, dtype=torch.float).view(-1, 1)
    
    return node_features, y

def construct_graph(node_features, processed_dataset, K=10):
    """
    Construct a graph based on cosine similarity of tabular features.

    Parameters:
    - node_features: Tensor of node features.
    - processed_dataset: The dataset containing all samples.
    - K: Number of nearest neighbors.

    Returns:
    - data: PyTorch Geometric Data object with node features and edge index.
    """
    # For edge construction, we use cosine similarity of tabular features
    # Extract tabular features
    all_tab = []
    for i in range(len(processed_dataset)):
        _, tab, _ = processed_dataset[i]
        all_tab.append(tab.numpy())
    
    all_tabular_features_np = np.array(all_tab)
    similarity_matrix = cosine_similarity(all_tabular_features_np)
    edges_src = []
    edges_dst = []
    num_nodes = similarity_matrix.shape[0]

    for i in range(num_nodes):
        sim_scores = similarity_matrix[i]
        neighbors = np.argsort(sim_scores)[-K-1:-1]  # Exclude self
        for nbr in neighbors:
            edges_src.append(i)
            edges_dst.append(nbr)

    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    data = Data(x=node_features, edge_index=edge_index)
    return data

def prepare_data_for_ablation(processed_dataset, cnn_model, device, ablation='none'):
    """
    Prepare PyTorch Geometric Data object for a specific ablation.

    Parameters:
    - processed_dataset: The dataset containing all samples.
    - cnn_model: The pre-trained CNN model to extract features.
    - device: The device to perform computations on.
    - ablation: Type of ablation ('none', 'image_only', 'tabular_only').

    Returns:
    - data: PyTorch Geometric Data object ready for training/evaluation.
    - y: Tensor of target survival lengths.
    """
    node_features, y = extract_node_features(processed_dataset, cnn_model, device, ablation=ablation)
    data = construct_graph(node_features, processed_dataset, K=10)
    data.y = y.to(device)
    
    num_nodes = data.num_nodes
    num_samples = len(processed_dataset)
    if num_nodes != num_samples:
        raise ValueError("Number of nodes does not match number of samples.")
    
    # Split indices
    num_samples = len(processed_dataset)
    train_samples = 2500
    val_samples = 145
    test_samples = num_samples - train_samples - val_samples

    train_indices = list(range(train_samples))
    val_indices = list(range(train_samples, train_samples + val_samples))
    test_indices = list(range(train_samples + val_samples, num_samples))

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_indices] = True
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask[val_indices] = True
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask[test_indices] = True

    data.train_mask = train_mask.to(device)
    data.val_mask = val_mask.to(device)
    data.test_mask = test_mask.to(device)
    data = data.to(device)
    
    return data, y

def ablation_study(processed_dataset, cnn_model, device, ablation_settings, hyperparams, epochs=100, seeds=[0,1,2]):
    """
    Perform ablation studies based on different ablation settings.

    Parameters:
    - processed_dataset: The dataset containing all samples.
    - cnn_model: The pre-trained CNN model to extract features.
    - device: The device to perform computations on.
    - ablation_settings: List of ablation settings to perform.
    - hyperparams: Dictionary of hyperparameters (e.g., learning rates).
    - epochs: Number of training epochs.
    - seeds: List of random seeds for reproducibility.

    Returns:
    - results_dict: Dictionary containing results for each ablation setting.
    """
    results_dir = 'ablation_studies'
    create_results_dir(results_dir)
    
    results_dict = {
        'Ablation': [],
        'LearningRate': [],
        'Mean_Test_MSE': [],
        'Std_Test_MSE': [],
        'Mean_Test_MAE': [],
        'Std_Test_MAE': [],
        'Mean_Test_Relative_MAE(%)': [],
        'Std_Test_Relative_MAE(%)': [],
        'Mean_Survival': [],
        'Variance_Survival': []
    }
    
    for ablation in ablation_settings:
        print(f"\n=== Ablation Setting: {ablation} ===")
        # Prepare data
        data, y = prepare_data_for_ablation(processed_dataset, cnn_model, device, ablation=ablation)
        
        # Compute mean and variance of survival
        mean_survival, variance_survival = compute_mean_variance_survival(processed_dataset)
        
        for lr in hyperparams['learning_rates']:
            print(f"\n--- Learning Rate: {lr} ---")
            # Define GraphSAGE model
            if ablation == 'image_tabular' or ablation == 'none':
                in_channels = 256 + 32  # Image + Tabular
            elif ablation == 'image_only':
                in_channels = 256
            elif ablation == 'tabular_only':
                in_channels = 32
            else:
                raise ValueError(f"Unknown ablation setting: {ablation}")
            
            hidden_channels = hyperparams.get('hidden_channels', 64)
            num_layers = hyperparams.get('num_layers', 2)
            model = GraphSAGEModel(in_channels=in_channels,
                                   hidden_channels=hidden_channels,
                                   out_channels=1,
                                   num_layers=num_layers).to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            criterion = nn.MSELoss()
            
            # Run training with multiple seeds
            test_mse_list = []
            test_mae_list = []
            test_rmae_list = []
            first_seed_train_losses = []
            first_seed_val_losses = []
            
            for run_idx, seed in enumerate(seeds):
                print(f"Run {run_idx+1}/{len(seeds)} with seed {seed}")
                set_seed(seed)
                model = GraphSAGEModel(in_channels=in_channels,
                                       hidden_channels=hidden_channels,
                                       out_channels=1,
                                       num_layers=num_layers).to(device)
                optimizer = optim.Adam(model.parameters(), lr=lr)
                
                best_val_loss = float('inf')
                best_model_weights = None
                train_losses = []
                val_losses = []
                
                for epoch in range(1, epochs+1):
                    train_loss = train_one_epoch_gnn(model, optimizer, criterion, data)
                    val_loss = evaluate_gnn(model, criterion, data, data.val_mask)
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model_weights = model.state_dict()
                
                # Load best weights
                model.load_state_dict(best_model_weights)
                test_mse, test_mae, test_rmae, _, _ = compute_metrics_gnn(model, data, data.test_mask, device)
                test_mse_list.append(test_mse)
                test_mae_list.append(test_mae)
                test_rmae_list.append(test_rmae)
                
                if run_idx == 0:
                    first_seed_train_losses = train_losses
                    first_seed_val_losses = val_losses
            
            # Compute mean and std
            mse_mean, mse_std = mean(test_mse_list), stdev(test_mse_list) if len(test_mse_list) >1 else 0.0
            mae_mean, mae_std = mean(test_mae_list), stdev(test_mae_list) if len(test_mae_list) >1 else 0.0
            rmae_mean, rmae_std = mean(test_rmae_list), stdev(test_rmae_list) if len(test_rmae_list) >1 else 0.0
            
            # Compute MAE as percentage of mean survival
            mae_percentage_mean = (mae_mean / mean_survival)*100 if mean_survival != 0 else 0
            
            print(f"Test MSE: {mse_mean:.4f} ± {mse_std:.4f}")
            print(f"Test MAE: {mae_mean:.4f} ± {mae_std:.4f}")
            print(f"Test Relative MAE: {rmae_mean:.2f}% ± {rmae_std:.2f}%")
            print(f"MAE as % of Mean Survival: {mae_percentage_mean:.2f}%")
            
            # Save results
            results_dict['Ablation'].append(ablation)
            results_dict['LearningRate'].append(lr)
            results_dict['Mean_Test_MSE'].append(mse_mean)
            results_dict['Std_Test_MSE'].append(mse_std)
            results_dict['Mean_Test_MAE'].append(mae_mean)
            results_dict['Std_Test_MAE'].append(mae_std)
            results_dict['Mean_Test_Relative_MAE(%)'].append(rmae_mean)
            results_dict['Std_Test_Relative_MAE(%)'].append(rmae_std)
            results_dict['Mean_Survival'].append(mean_survival)
            results_dict['Variance_Survival'].append(variance_survival)
            
            # Plot training curves for the first seed
            curve_fig_path = os.path.join(results_dir, f'ablation_{ablation}_lr_{lr}_train_val_curve.png')
            plot_train_val_curves(first_seed_train_losses, first_seed_val_losses, curve_fig_path)
        
    # Save all results to CSV
    results_csv_path = os.path.join(results_dir, 'ablation_study_results.csv')
    save_results_to_csv(results_dict, results_csv_path)
    
    # Additionally, plot comparison of ablation settings for key metrics
    plot_ablation_results(results_dict, results_dir)
    
    return results_dict

def plot_ablation_results(results_dict, results_dir):
    """
    Plot the ablation study results for comparison.

    Parameters:
    - results_dict: Dictionary containing ablation results.
    - results_dir: Directory to save the plots.
    """
    ablations = list(set(results_dict['Ablation']))
    metrics = ['Mean_Test_MSE', 'Mean_Test_MAE', 'Mean_Test_Relative_MAE(%)']
    
    for metric in metrics:
        plt.figure(figsize=(10,6))
        for ablation in ablations:
            indices = [i for i, x in enumerate(results_dict['Ablation']) if x == ablation]
            lr = [results_dict['LearningRate'][i] for i in indices]
            metric_values = [results_dict[metric][i] for i in indices]
            plt.plot(lr, metric_values, marker='o', label=ablation)
        plt.xscale('log')
        plt.xlabel('Learning Rate')
        plt.ylabel(metric)
        plt.title(f'Ablation Study: {metric} vs Learning Rate')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'ablation_{metric}.png'))
        plt.close()


if __name__ == "__main__":
    processed_data_path = '/data37/xl693/RADCURE2024_processed'
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    
    # Load Dataset
    processed_dataset = PreprocessedPatientDataset(processed_data_path)
    _, sample_tabular_tensor, _ = processed_dataset[0]
    tabular_input_size = sample_tabular_tensor.shape[0]
    
    # Initialize the CNN-based Survival Prediction Model
    cnn_model = SurvivalPredictionModel(image_in_channels=1, tabular_input_size=tabular_input_size).to(device)
    
    # Load pre-trained weights
    model_save_path = '/data37/xl693/RADCURE2024_model_checkpoints'
    model_path = os.path.join(model_save_path, 'model_latest.pth')
    cnn_model.load_state_dict(torch.load(model_path, map_location=device))
    cnn_model.eval()

    ablation_settings = ['image_tabular', 'image_only', 'tabular_only']
    
    # Define Hyperparameters for the GNN
    hyperparams = {
        'learning_rates': [0.0005, 0.001, 0.005],
        'hidden_channels': 64,
        'num_layers': 2
    }
    
    # Perform Ablation Studies
    results = ablation_study(
        processed_dataset=processed_dataset,
        cnn_model=cnn_model,
        device=device,
        ablation_settings=ablation_settings,
        hyperparams=hyperparams,
        epochs=100,
        seeds=[0,1,2]
    )
    
    print("\n=== Ablation Study Completed ===")
    print(f"Results saved in the 'ablation_studies' directory.")
