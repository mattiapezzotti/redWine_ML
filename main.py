import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    df = pd.read_csv('winequality-red.csv')
    df.drop_duplicates(inplace = True)

    printBasicInfo(df)
    
    plt.show()
    

def loadCorrelationHeatmap(df):
    corrMatrix = df.corr()
    heatmapMask = np.triu(np.ones_like(corrMatrix))
    sns.heatmap(corrMatrix, linewidths=1, annot=True, fmt=".1f", cmap=sns.color_palette("vlag"), mask=heatmapMask)

def loadPairPlot(df):
    sns.pairplot(df, hue="quality", height=1.5, corner=True)

def printBasicInfo(df):
    output = open("output.txt", "w")

    output.write("----HEAD----\n\n" 
                 + df.head(10).to_string())
    
    output.write("\n\n----INFO----\n\n")
    df.info(buf=output)

    output.write("\n\n----NULL VALUES----\n\n" 
                 + df.isnull().sum().to_string())
    

    output.close()

    
if __name__ == "__main__":
    main()