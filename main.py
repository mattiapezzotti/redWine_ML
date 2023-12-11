import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def main():
    df = pd.read_csv('winequality-red.csv')

    #df = dfAnalysis(df)
    columns = [col for col in df.columns if(col != "quality")]

    plt.figure(figsize=(14,len(columns)*3))

    #loadQualityCorrHP(df, columns)
    #loadQualityCorrBP(df, columns)
    #loadQualityCorrHM(df)

    #loadCategorizedQualityCorrBP(df, columns)
    #loadCategorizedQualityCorrHM(df)

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

def loadCategorizedQualityCorrBP(df, columns):
    df["qualityRange"] = pd.cut(df["quality"], bins=[-np.inf, 4, 6, np.inf], labels=["3-4","5-6","7-8"])
    for i,column in enumerate(columns):
        plt.subplot( len( columns ) // 2 + 1, 2, i + 1 )
        sns.boxplot(x="qualityRange", y=column, data=df,palette="flare")
        plt.title(f"{column} Distribution")
    plt.tight_layout()
    plt.savefig('qualityCatCorrBox.png')

def loadCategorizedQualityCorrHM(df):
    df["quality"] = pd.cut(df["quality"], bins=[-np.inf, 4, 6, np.inf], labels=["3-4","5-6","7-8"])
    plt.figure(figsize=(14,14))
    corrMatrix = df.corr(numeric_only=True)
    heatmapMask = np.triu(np.ones_like(corrMatrix))
    sns.heatmap(corrMatrix, linewidths=1, annot=True, fmt=".1f", vmin=-1, vmax=1, cmap=sns.color_palette("vlag"), mask=heatmapMask)
    plt.tight_layout()
    plt.savefig("images/qualityCatCorrHM.png")

def loadQualityCorrHM(df):
    plt.figure(figsize=(14,14))
    corrMatrix = df.corr()
    heatmapMask = np.triu(np.ones_like(corrMatrix))
    sns.heatmap(corrMatrix, linewidths=1, annot=True, fmt=".1f", vmin=-1, vmax=1, cmap=sns.color_palette("vlag"), mask=heatmapMask)
    plt.tight_layout()
    plt.savefig("images/qualityCorrHM.png")

def loadPairPlot(df):
    sns.pairplot(df, hue="quality", height=1.5, corner=True)

def loadQualityCorrBP(df, columns):
    for i, column in enumerate(columns):
        plt.subplot( len( columns ) // 2 + 1, 2, i + 1 )
        sns.boxplot( x = "quality", y = column, data = df, palette="flare")
    plt.tight_layout()
    plt.savefig("images/qualityCorrBox.png")

def loadQualityCorrHP(df, columns):
    for i,column in enumerate(columns):
        plt.subplot( len( columns ) // 2 + 1, 2, i + 1)
        sns.histplot( x = column, hue = "quality", data=df, kde=True, palette="Spectral")
    plt.tight_layout()
    plt.savefig("images/qualityHistPlot.png")
    
def loadCountPlot(df, target):
    sns.countplot(df, x=target, palette = "flare")

def studyPCA(df, columns):
    qualityColumn = df['quality']
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[columns])
    x_train, x_test, y_train, y_test = train_test_split(scaled_data, qualityColumn, test_size = 0.25, random_state=42)

    pca = PCA(n_components=None)
    
    x_train = pca.fit_transform(x_train)
    x_test = pca.transform(x_test)
    explained_variance = pca.explained_variance_ratio_

    print(sorted(explained_variance, reverse = True))

    plt.plot(range(1, pca.n_components_ + 1), explained_variance)
    plt.xlabel('Componenti')
    plt.ylabel('Varianza')

def applyPCA(df, columns):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[columns])
    pcaData = PCA(n_components=6).fit_transform(scaled_data)

    return pcaData

if __name__ == "__main__":
    main()