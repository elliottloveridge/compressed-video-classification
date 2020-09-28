import pandas as pd

path = '/app/compressed-3d-cnn/model_summary/mobilenetv2.csv'

with open(path, newline='') as f:
    reader = pd.read_csv(f)
    reader = reader[reader['Type'] == 'Conv3d']

print(reader)

# for index, row in reader.iterrows():
#     print((reader['Name'], reader['Sparsity']))
