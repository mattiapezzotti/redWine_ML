import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('winequality-red.csv')
output = open("output.txt", "w") 

#output.write(df.head().to_string())
#output.write(df.corr().to_string())
#df.info()
#sns.pairplot(df, hue="quality", height=1.5, corner=True)
sns.heatmap(df.corr(), annot=True, fmt=".1f")
plt.show()

output.close()