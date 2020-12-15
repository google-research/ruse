import pandas as pd
df = pd.read_csv ('text', sep=' ')
print(df.head())
print(df['JobName'])
