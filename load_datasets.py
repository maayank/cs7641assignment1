import pandas as pd

def load_cancer():
    names = ['id'] + ['fut' + str(i) for i in range(1, 10)] +  ['malignant']
    df = pd.read_csv('data/cancer/breast-cancer-wisconsin.data.cleaned', sep=',', names=names)
    df.drop(columns=['id'], inplace=True)
    df['malignant'] = df['malignant'] == 4
    return df

def load_wine():
    df1 = pd.read_csv('data/wine/winequality-red.csv', sep=';')
    df1.insert(0, 'type', 1)
    df2 = pd.read_csv('data/wine/winequality-white.csv', sep=';')
    df2.insert(0, 'type', 2)
    df = df1.append(df2, ignore_index=True)
    df['quality'] = df['quality'] > 5
    return df

def load_mushroom():
    df = pd.read_csv('data/mushroom/agaricus-lepiota.data')
#    df.drop(columns=['id'], inplace=True)
#    df['malignant'] = df['malignant'] == 4
    df = df.astype("category")
    return df

if __name__=='__main__':
    print(load_cancer())
    print(load_wine())
#    print(load_mushroom())
