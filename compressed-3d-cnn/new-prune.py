import pandas as pd

path = '/app/compressed-3d-cnn/model_summary/mobilenetv2.csv'

with open(path, newline='') as f:
    df = pd.read_csv(f)
    df = df[df['Type']=='Conv3d']

for i in df.index:
    print((df['Name'][i], df['Sparsity'][i]))
