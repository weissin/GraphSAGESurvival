import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import pickle
from sklearn.preprocessing import LabelEncoder



class PreprocessedPatientDataset(Dataset):
    def __init__(self, processed_data_path):
        """
        Dataset class for loading preprocessed patient data.

        Parameters:
        - processed_data_path: Path to the directory containing preprocessed patient data files.
        """
        self.processed_data_path = processed_data_path
        
        # Collect all preprocessed patient data files
        self.patient_files = [os.path.join(processed_data_path, f) 
                              for f in os.listdir(processed_data_path) if f.endswith('.pkl')]
        if not self.patient_files:
            raise ValueError(f"No preprocessed files found in directory: {processed_data_path}")
        
        # Preload tabular features to compute normalization statistics
        self.tabular_features_list = []
        self.categorical_columns = set()
        self.label_encoders = {}
        
        for file_path in self.patient_files:
            with open(file_path, 'rb') as f:
                patient_data = pickle.load(f)
                tabular_dict = patient_data['tabular_data']
                treatment_data = tabular_dict['treatment_data']
                status_data = tabular_dict['status_data']
                
                # Convert dict values to a list and remove any Timestamp values
                tabular_features = [
                    v for v in list(treatment_data.values()) + list(status_data.values())
                    if not isinstance(v, (pd.Timestamp, np.datetime64))
                ]
                
                # Identify categorical columns (containing strings or mixed types)
                for idx, value in enumerate(tabular_features):
                    if isinstance(value, str) or (
                        not np.issubdtype(type(value), np.number) and not isinstance(value, (float, int))
                    ):
                        self.categorical_columns.add(idx)

                self.tabular_features_list.append(tabular_features)

        # Process categorical features using LabelEncoder
        self.tabular_features_array = np.array(self.tabular_features_list, dtype=object)
        for col_idx in self.categorical_columns:
            # Convert all column values to strings
            self.tabular_features_array[:, col_idx] = self.tabular_features_array[:, col_idx].astype(str)
            encoder = LabelEncoder()
            self.tabular_features_array[:, col_idx] = encoder.fit_transform(self.tabular_features_array[:, col_idx])
            self.label_encoders[col_idx] = encoder

        # Convert to float after encoding
        self.tabular_features_array = self.tabular_features_array.astype(np.float32)
        
        # Compute normalization statistics for numerical data
        self.tabular_mean = np.mean(self.tabular_features_array, axis=0)
        self.tabular_std = np.std(self.tabular_features_array, axis=0)
        self.tabular_std[self.tabular_std == 0] = 1.0  # Avoid division by zero

        print(f"Found {len(self.patient_files)} preprocessed patient data files.")
        print(f"Processed categorical columns: {self.categorical_columns}")
        print(f"Computed normalization statistics for tabular data.")

    def __len__(self):
        """
        Returns the number of patient files in the dataset.
        """
        return len(self.patient_files)

    def normalize_image(self, image_tensor):
        """
        Normalize the image tensor on a per-sample basis.

        Parameters:
        - image_tensor: A tensor of shape [1, D, H, W] representing the image data.

        Returns:
        - Normalized image tensor.
        """
        image_mean = image_tensor.mean()
        image_std = image_tensor.std()
        if image_std == 0:
            image_std = 1.0  # Avoid division by zero
        return (image_tensor - image_mean) / image_std

    def normalize_tabular(self, tabular_features):
        """
        Normalize tabular data using dataset-wide statistics.

        Parameters:
        - tabular_features: A numpy array of raw tabular features.

        Returns:
        - Normalized tabular features.
        """
        return (tabular_features - self.tabular_mean) / self.tabular_std

    def __getitem__(self, idx):
        """
        Retrieve a single sample from the dataset.

        Parameters:
        - idx: Index of the sample to retrieve.

        Returns:
        - A tuple containing:
          - image_tensor: Normalized image tensor of shape [1, D, H, W].
          - tabular_tensor: Normalized tabular data tensor.
          - survival_length: Target tensor representing the survival length.
        """
        # Load the patient data file
        file_path = self.patient_files[idx]
        with open(file_path, 'rb') as f:
            patient_data = pickle.load(f)

        # Process image data
        image_array = patient_data['image_data']
        image_tensor = torch.tensor(image_array, dtype=torch.float32).unsqueeze(0)  # Add channel dimension [1, D, H, W]
        image_tensor = self.normalize_image(image_tensor)

        # Process tabular data
        tabular_dict = patient_data['tabular_data']
        treatment_data = tabular_dict['treatment_data']
        status_data = tabular_dict['status_data']
        tabular_features = [
            v for v in list(treatment_data.values()) + list(status_data.values())
            if not isinstance(v, (pd.Timestamp, np.datetime64))
        ]

        # Encode categorical features
        for col_idx in self.categorical_columns:
            tabular_features[col_idx] = self.label_encoders[col_idx].transform([str(tabular_features[col_idx])])[0]

        # Convert to float and normalize
        tabular_features = np.array(tabular_features, dtype=np.float32)
        tabular_tensor = torch.tensor(self.normalize_tabular(tabular_features), dtype=torch.float32)

        # Survival length as target
        survival_length = torch.tensor([tabular_dict['length_fu']], dtype=torch.float32)

        return image_tensor, tabular_tensor, survival_length
    


def conv_block(in_channels, out_channels, use_dropout=False, dropout_rate=0.1):
    layers = [
        nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True)
    ]
    if use_dropout:
        layers.append(nn.Dropout3d(dropout_rate))
    return nn.Sequential(*layers)

# Define the UNet-like 3D feature extractor
class UNet3D_FeatureExtractor(nn.Module):
    def __init__(self, in_channels, use_dropout=False, dropout_rate=0.1):
        super(UNet3D_FeatureExtractor, self).__init__()
        self.encoder1 = conv_block(in_channels, 32, use_dropout, dropout_rate)
        self.pool1 = nn.MaxPool3d(2)
        self.encoder2 = conv_block(32, 64, use_dropout, dropout_rate)
        self.pool2 = nn.MaxPool3d(2)
        self.encoder3 = conv_block(64, 128, use_dropout, dropout_rate)
        self.pool3 = nn.MaxPool3d(2)
        self.middle = conv_block(128, 256, use_dropout, dropout_rate)
        # Not including the decoder part as we're only extracting features

    def forward(self, x):
        x1 = self.encoder1(x)                  # [B, 32, D, H, W]
        x2 = self.encoder2(self.pool1(x1))     # [B, 64, D/2, H/2, W/2]
        x3 = self.encoder3(self.pool2(x2))     # [B, 128, D/4, H/4, W/4]
        x4 = self.middle(self.pool3(x3))       # [B, 256, D/8, H/8, W/8]
        return x4                              # Returning the deepest features

# Define the tabular data feature extractor
class TabularFeatureExtractor(nn.Module):
    def __init__(self, input_size, hidden_sizes=[64, 32]):
        super(TabularFeatureExtractor, self).__init__()
        layers = []
        in_features = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU(inplace=True))
            in_features = hidden_size
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Define the combined model for survival prediction
class SurvivalPredictionModel(nn.Module):
    def __init__(self, image_in_channels, tabular_input_size, use_dropout=False, dropout_rate=0.1):
        super(SurvivalPredictionModel, self).__init__()
        # Image feature extractor
        self.image_feature_extractor = UNet3D_FeatureExtractor(
            image_in_channels, use_dropout, dropout_rate
        )
        self.image_pool = nn.AdaptiveAvgPool3d(1)  # Global average pooling
        self.flatten = nn.Flatten()

        # Tabular data feature extractor
        self.tabular_feature_extractor = TabularFeatureExtractor(tabular_input_size)

        # Define the sizes for the combined features
        image_feature_size = 256  # Output channels from the image feature extractor
        tabular_feature_size = 32  # Output size from the tabular feature extractor
        combined_feature_size = image_feature_size + tabular_feature_size

        # Fully connected layers for prediction
        self.fc = nn.Sequential(
            nn.Linear(combined_feature_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(128),  # Adding BatchNorm after the first FC layer
            nn.Linear(128, 64),  # New additional FC layer with 64 units
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),  # BatchNorm after the new FC layer
            nn.Linear(64, 1)  # Output layer for regression (survival length)
        )

    def forward(self, image, tabular):
        # Extract features from the image
        image_features = self.image_feature_extractor(image)  # [B, 256, D', H', W']
        image_features = self.image_pool(image_features)      # [B, 256, 1, 1, 1]
        image_features = self.flatten(image_features)         # [B, 256]

        # Extract features from the tabular data
        tabular_features = self.tabular_feature_extractor(tabular)  # [B, 32]

        # Combine both feature sets
        combined_features = torch.cat([image_features, tabular_features], dim=1)  # [B, 288]

        # Pass through the fully connected layers to get the prediction
        output = self.fc(combined_features)  # [B, 1]

        return output