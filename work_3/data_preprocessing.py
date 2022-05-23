import pandas as pd
import pdb
from scipy.io import arff
from sklearn import preprocessing as skpre

def load_arff_data_fold(path_datasets, name_dataset, nfolds):
    # Function to create unique tran and test df, with a column related to the fold
    # Loop over files by folds and data type
    df = {}
    l_data_type = ['train', 'test']
    for fold in range(nfolds):
        for data_type in l_data_type:
            # Load data
            name = f'{name_dataset}.fold.{str(fold).zfill(6)}.{data_type}.arff'
            data = arff.loadarff(f'{path_datasets}{name_dataset}/{name}')
            df_i = pd.DataFrame(data[0])
            list_obj_columns = df_i.select_dtypes(include=['object']).columns.tolist()
            # Decode object columns
            df_i[list_obj_columns] = df_i[list_obj_columns].stack().str.decode('utf-8').unstack()
            # Change object variables to string to reduce memory
            dict_newtypes = dict.fromkeys(list_obj_columns, 'category')
            df_i = df_i.astype(dict_newtypes)
            # Add column with fold
            df_i['fold'] = fold

            if fold == 0:
                # Initialize dataframes
                df[data_type] = df_i
            else:
                df[data_type] = df[data_type].append(df_i)
    return df


def preprocessing(dic_df, y_var):
    # Both train and test set should have same distribution of samples
    dic_df_pp = {}
    for data_type in ['train', 'test']:
        df = dic_df[data_type]
        # Dataframe without Y class and fold
        df_X = df[df.columns.difference([y_var, 'fold'])]
        for column in df_X:
            if pd.isnull(df_X[column]).all():
                print('Empty column, to be removed:', column)
                del df_X[column]
            # All columns with only 1 or  2 values will be mapped to true/false
            elif df_X[column].nunique() == 1:
                print('Variable with single value:', column)
                counts = df_X[column].value_counts().sort_index()
                df_X['{}_{}'.format(column, counts.index[0])] = df_X[column].replace(
                    {counts.index[0]: 1.0})
                del df_X[column]
            elif df_X[column].nunique() == 2:
                counts = df_X[column].value_counts().sort_index()
                df_X['{}_{}'.format(column, counts.index[1])] = df_X[column].replace(
                    {counts.index[0]: 0.0,
                     counts.index[1]: 1.0})
                del df_X[column]
        # Use OneHot enconding for the other categorical features
        df_X = pd.get_dummies(df_X)
        # Mean imputation
        df_X = df_X.apply(lambda x: x.fillna(x.mean()), axis=0)
        # Normalise
        min_max_scaler = skpre.MinMaxScaler()
        df_X_val_scaled = min_max_scaler.fit_transform(df_X)
        df_X_val_scaled = pd.DataFrame(df_X_val_scaled , columns=df_X.columns)
        df_final = pd.concat([df_X_val_scaled.reset_index(drop=True), df[[y_var, 'fold']].reset_index(drop=True)], axis=1)
        dic_df_pp[data_type] = df_final

    return dic_df_pp

