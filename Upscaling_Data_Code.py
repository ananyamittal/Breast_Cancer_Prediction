import pandas as pd

# Load your CSV file into a DataFrame
file_path = 'balanced_dataset.csv'
df = pd.read_csv(file_path)

# Set the desired number of times to upscale the data
upscale_factor = 2  # You can adjust this value as needed

# Upscale the DataFrame
upscaled_data = pd.concat([df] * upscale_factor, ignore_index=True)

# Save the upscaled DataFrame to a new CSV file
output_file_path = 'upscaled_data.csv'
upscaled_data.to_csv(output_file_path, index=False)

print(f"Upscaling completed for all columns. Upscaled data saved to {output_file_path}")
