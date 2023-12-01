#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.utils import resample

# Load your dataset
data = pd.read_csv("Cancer_Data.csv")

# Split the data into malignant and benign cases
malignant_data = data[data['diagnosis'] == 'M']
benign_data = data[data['diagnosis'] == 'B']

# Oversample the malignant cases to match the number of benign cases
oversampled_malignant_data = resample(malignant_data, replace=True, n_samples=len(benign_data), random_state=42)

# Combine the datasets
balanced_data = pd.concat([benign_data, oversampled_malignant_data])

# Shuffle the data
balanced_data = balanced_data.sample(frac=1, random_state=42)

# Now, `balanced_data` contains an equal number of malignant and benign cases.
# Replace "balanced_dataset.csv" with the desired filename for your updated dataset
balanced_file_path = "balanced_dataset.csv"

# Save the balanced dataset to a new CSV file
balanced_data.to_csv(balanced_file_path, index=False)

print(f'Balanced data saved to: {balanced_file_path}')



# In[3]:





# In[4]:





# In[7]:





# In[8]:





# In[ ]:




