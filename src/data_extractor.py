import pandas as pd
import os
import SimpleITK as sitk

class ExtractData:
    def __init__(self, xlsx_path, metadata_csv_path, base_dicom_path):
        """
        Initialize the ExtractData class by loading the tabular data and metadata.

        Parameters:
        - xlsx_path: Path to the Excel file containing the tabular data.
        - metadata_csv_path: Path to the CSV file containing the DICOM metadata.
        - base_dicom_path: The base path where DICOM folders are located.
        """
        # Load the tabular data from the Excel file
        self.df_tabular = pd.read_excel(xlsx_path)
        
        # Load the metadata and filter for DICOM series with more than one image
        self.df_metadata = pd.read_csv(metadata_csv_path)
        self.filtered_df_metadata = self.df_metadata[self.df_metadata['Number of Images'] > 1]
        
        # Store the base path for DICOM files
        self.base_dicom_path = base_dicom_path
        
        # Define the column names based on your Excel file structure
        self.status_columns = self.df_tabular.columns[1:14]    # Columns B-N
        self.treatment_columns = self.df_tabular.columns[14:20]  # Columns O-T
        
        # Check that required columns are present
        required_columns = ['patient_id', 'Date of Death', 'Length FU']
        for col in required_columns:
            if col not in self.df_tabular.columns:
                raise ValueError(f"Required column '{col}' not found in the tabular data.")

    def extract_tabular_data(self, patient_id):
        """
        Extracts the tabular data for a single patient.

        Parameters:
        - patient_id: The unique identifier for the patient.

        Returns:
        - A dictionary containing the extracted tabular data.
        """
        # Locate the row corresponding to the patient_id
        row = self.df_tabular[self.df_tabular['patient_id'] == patient_id]
        if row.empty:
            print(f"No tabular data found for patient_id {patient_id}")
            return None
        
        # Extract the required data
        row = row.iloc[0]
        treatment_data = row[self.treatment_columns].to_dict()
        status_data = row[self.status_columns].to_dict()
        length_fu = row['Length FU']
        date_of_death = row['Date of Death']
        is_dead = 1 if pd.notnull(date_of_death) else 0
        
        # Combine all data into a single dictionary
        data = {
            'patient_id': patient_id,
            'treatment_data': treatment_data,
            'status_data': status_data,
            'length_fu': length_fu,
            'is_dead': is_dead
        }
        return data

    def process_dicom_series(self, input_folder):
        """
        Processes the DICOM series from the input folder.

        Parameters:
        - input_folder: Path to the folder containing the DICOM files.

        Returns:
        - A numpy array representing the processed 3D image data.
        """
        # Construct the full path to the DICOM folder
        dicom_folder = os.path.join(self.base_dicom_path, input_folder.strip('./'))
        dicom_folder = os.path.normpath(dicom_folder)

        reader = sitk.ImageSeriesReader()
        dicom_files = reader.GetGDCMSeriesFileNames(dicom_folder)
        if not dicom_files:
            print(f"No DICOM files found in folder {dicom_folder}")
            return None
        
        reader.SetFileNames(dicom_files)
        image_3d = reader.Execute()
        image_array = sitk.GetArrayFromImage(image_3d)
        total_slices = image_array.shape[0]
        
        # Define slicing and cropping parameters
        start_slice = 30
        new_start_slice = max(start_slice, total_slices - ((total_slices - start_slice) // 32) * 32)
        x_range = (100, 401)
        y_range = (100, 401)
        
        # Check slice bounds
        if new_start_slice + 32 > total_slices:
            print(f"Not enough slices in DICOM data for folder {dicom_folder}")
            return None
        
        # Crop the image array
        cropped_image_array = image_array[
            new_start_slice:new_start_slice+32,
            x_range[0]:x_range[1],
            y_range[0]:y_range[1]
        ]
        
        return cropped_image_array

    def get_patient_data(self, patient_id):
        """
        Retrieves both tabular and image data for a single patient.

        Parameters:
        - patient_id: The unique identifier for the patient.

        Returns:
        - A dictionary containing both the tabular data and the image data.
        """
        # Extract tabular data
        tabular_data = self.extract_tabular_data(patient_id)
        if not tabular_data:
            return None
        
        # Locate the DICOM folder for the patient
        row = self.filtered_df_metadata[self.filtered_df_metadata['Subject ID'] == patient_id]
        if row.empty:
            print(f"No DICOM data found for patient_id {patient_id}")
            return None
        
        input_folder = row.iloc[0]['File Location']
        
        # Process DICOM images
        image_data = self.process_dicom_series(input_folder)
        if image_data is None:
            return None
        
        # Combine tabular and image data
        patient_data = {
            'tabular_data': tabular_data,
            'image_data': image_data
        }
        return patient_data