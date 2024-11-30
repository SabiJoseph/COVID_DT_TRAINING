#import library
import pandas as pd

# Load
file_path = r"E:\Research\PRODIGY_DS_03\Dataset\full_grouped.csv"
data = pd.read_csv(file_path)

# Convert
data['Date'] = pd.to_datetime(data['Date'])

# Fill
data.fillna(0, inplace=True)

# Feature Engineering
data['Case_Fatality_Rate'] = data['Deaths'] / data['Confirmed']  # Case Fatality Rate
data['Recovery_Rate'] = data['Recovered'] / data['Confirmed']    # Recovery Rate

# Encode
data = pd.get_dummies(data, columns=['Country/Region', 'WHO Region'])

# Target
data['High_Risk'] = (data['Deaths'] > 1000).astype(int)

# Save
cleaned_file_path = r"E:\Research\PRODIGY_DS_03\Dataset\cleaned_full_grouped.csv"
data.to_csv(cleaned_file_path, index=False)

# Display
print(data.head())
