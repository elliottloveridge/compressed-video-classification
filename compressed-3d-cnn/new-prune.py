import csv

path = '~/research-project/compressed-3d-cnn/model_summary/mobilenetv2.csv'

with open(path, newline='') as f:
    reader = csv.reader(f)
    data = list(reader)

print(data)
