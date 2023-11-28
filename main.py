import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    df = pd.read_csv('winequality-red.csv')

    plt.show()
    

def loadCorrelationHeatmap(df):
    corrMatrix = df.corr()
    heatmapMask = np.triu(np.ones_like(corrMatrix))
    sns.heatmap(corrMatrix, linewidths=1, annot=True, fmt=".1f", cmap=sns.color_palette("vlag"), mask=heatmapMask)

def loadPairPlot(df):
    sns.pairplot(df, hue="quality", height=1.5, corner=True)

def getBasicInfo(df):
    output = open("output.txt", "w")
    output.write("----HEAD----\n\n" 
                 + df.head(10).to_string() 
                 + "\n\n"
                )
    
    output.write("----INFO----\n\n")
    df.info(buf=output)
    output.close()

    
if __name__ == "__main__":
    main()