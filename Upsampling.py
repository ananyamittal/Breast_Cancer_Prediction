import pandas as pd
from sklearn.utils import resample

# Load your CSV file into a DataFrame
file_path = 'balanced_dataset.csv'
df = pd.read_csv(file_path)

# Determine the class with fewer instances
min_class_size = df.shape[0]

# Upsample the entire DataFrame
upsampled_data = resample(df, replace=True, n_samples=min_class_size, random_state=42)

# Save the upsampled DataFrame to a new CSV file
output_file_path = 'upsampled_data.csv'
upsampled_data.to_csv(output_file_path, index=False)

print(f"Upsampling completed for all columns. Upsampled data saved to {output_file_path}")
