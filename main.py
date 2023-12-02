import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def main():
    df = pd.read_csv('winequality-red.csv')

    #df = dfAnalysis(df)
    columns = [col for col in df.columns if(col != "quality")]

    studyPCA(df, columns)

    #pcaData = applyPCA(df, columns)

    plt.show()
    

def dfAnalysis(df):
    output = open("basicInfo.txt", "w")

    output.write("----HEAD----\n\n" 
                + df.head(10).to_string())
    
    output.write("\n\n----INFO----\n\n")
    df.info(buf=output)

    output.write("\n\n----NULL VALUES----\n\n" 
                + str(df.isnull().sum()))
    
    output.write("\n\n----DUPLICATED VALUES----\n\n" 
                + str(df.duplicated().sum()))
    
    if(df.duplicated().sum() > 300):
        output.write("\n\ndropping duplicates...")
        df.drop_duplicates(inplace = True)
        df.reset_index(drop=True)
    
    output.write("\n\n----DESCRIBE----\n\n" 
                + str(df.describe().T))
    
    output.write("\n\n----QUALITY COUNT----\n\n" 
                + str(df["quality"].value_counts()))
    
    output.write("\n\n----QUALITY CORRELATION----\n\n" 
                + str(df.corr()['quality'].sort_values(ascending=False)))
    
    output.close()
    return df

def studyPCA(df, columns):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[columns])

    pca = PCA(n_components=11).fit(scaled_data) 
    plt.plot(range(1, pca.n_components_ + 1), pca.explained_variance_ratio_)
    plt.xlabel('Componenti')
    plt.ylabel('Varianza')

def applyPCA(df, columns):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[columns])
    pcaData = PCA(n_components=7).fit_transform(scaled_data)

    return pcaData


def loadCorrelationHeatmap(df):
    corrMatrix = df.corr()
    heatmapMask = np.triu(np.ones_like(corrMatrix))
    sns.heatmap(corrMatrix, linewidths=1, annot=True, fmt=".1f", vmin=-1, vmax=1, cmap=sns.color_palette("vlag"), mask=heatmapMask)

def loadPairPlot(df):
    sns.pairplot(df, hue="quality", height=1.5, corner=True)

def loadBoxPlot(df, columns):
    for i, column in enumerate(columns):
        plt.subplot( len( columns ) // 2 + 1, 2, i + 1 )
        sns.boxplot( x = "quality", y = column, data = df, palette="flare")

def loadDistributionPlot(df, columns):
    for i,column in enumerate(columns):
        plt.subplot( len( columns ) // 2 + 1, 2, i + 1)
        sns.histplot( x = column, hue = "quality", data=df, kde=True, palette="Spectral")
    
def loadCountPlot(df, target):
    sns.countplot(df, x=target, palette = "flare")

if __name__ == "__main__":
    main()