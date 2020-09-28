import pandas as pd

path = '/app/compressed-3d-cnn/model_summary/mobilenetv2.csv'

with open(path, newline='') as f:
    reader = pd.read_csv(f)
    # data = list(reader)

# print(data)
for row in reader:
    print(row)
