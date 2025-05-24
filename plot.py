import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('unfairDataset.csv', header=None, names=['x', 'y', 'group'])

plt.figure(figsize=(10, 8))

group_a = df[df['group'] == 'A']
group_b = df[df['group'] == 'B']

plt.scatter(group_a['x'], group_a['y'], c='blue', label='Group A', alpha=0.6, s=10)
plt.scatter(group_b['x'], group_b['y'], c='red', label='Group B', alpha=0.6, s=10)

plt.title('Cluster Visualization')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()
