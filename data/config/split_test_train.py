import pandas as pd

# Read the data from NEW_FILE_PATH.csv
data = pd.read_csv(r'C:\Users\petef\Documents\Western 5\Capstone\repo\backend\data\case_id_tags_merged.csv')

# Create a dataframe with rows that have a value in Name and Product Number
df1 = data.dropna(subset=['Name', 'Product Number'])

# Create a dataframe with rows that only have a value in Name
df2 = data[data['Name'].notna() & data['Product Number'].isna()]

# Save the dataframes as separate CSV files
df1.to_csv('Trainset.csv', index=False)
df2.to_csv('Testset.csv', index=False)
