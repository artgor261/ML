import pandas as pd

base = 'C:/Users/Артем/Datasets/'
data = pd.read_csv(base + 'organisations.csv')
df = data[(data["average_bill"].notna()) & (data["average_bill"] <= 2500)]
print(df)
