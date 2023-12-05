import pandas as pd

# Read the CSV file
df = pd.read_csv('data\case_ids.csv')
df['Name'] = df['Name'].str[:8]

df_2 = pd.read_csv(r'data\tags_data.csv')

# Perform left join and keep all columns
df_merged = df_2.merge(df, how='right', left_on='Product Number', right_on='Name')


# Save the modified DataFrame to a new CSV file in the parent folder, data
df_merged.to_csv('case_id_tags_merged.csv', index=False)


