import pandas as pd
from scipy.io import arff
from utils.utils import *
from sklearn import preprocessing as skpre


def load_arff_data(dataset_path):
    # Load data
    data = arff.loadarff(dataset_path)
    df = pd.DataFrame(data[0])

    list_obj_columns = df.select_dtypes(include=['object']).columns.tolist()

    # Decode object columns
    df[list_obj_columns] = df[list_obj_columns].stack().str.decode('utf-8').unstack()

    # Change object variables to string to reduce memory
    dict_newtypes = dict.fromkeys(list_obj_columns, 'category')
    df = df.astype(dict_newtypes)
    return df


def preprocessing(df, dataset_name, y_column_name):
    # Store target variable and drop it from the main dataframe
    df = df.apply(lambda x: x.str.decode("utf-8") if x.dtype == object else x)

    df_y = df[[y_column_name]]
    df.drop(columns=[y_column_name], inplace=True)
    list_cat_columns = df.select_dtypes(include=['category']).columns.tolist()

    if dataset_name == 'vote.arff':
        # Categorical values changed to ordinal numerical equally spaced
        df = ordinal_vote_representation(df[list_cat_columns], ['n', '?', 'y'], [-1, 1])

    if dataset_name == 'heart-c.arff':
        # Fill nans
        df['ca'] = df['ca'].fillna(df['ca'].mode()[0])
        # all columns with only 2 values will be mapped to true/false
        for col in df:
            if df[col].nunique() == 2:
                counts = df[col].value_counts()
                df['{}_{}'.format(col, counts.index[0])] = df[col].replace(
                    {counts.index[0]: 1.0,
                    counts.index[1]: 0.0})
                del df[col]
        # Use OneHot enconding for the other categorical features
        df = pd.get_dummies(df)

        # Normalise
        min_max_scaler = skpre.MinMaxScaler()
        df_val_scaled = min_max_scaler.fit_transform(df)
        df = pd.DataFrame(df_val_scaled, columns=df.columns)

    if dataset_name == 'breast-w.arff':
        # Fill with mean value
        df = df.apply(lambda x: x.fillna(x.mean()), axis=0)

        # Normalize columns
        df = (df - df.min()) / (df.max() - df.min())

    return df, df_y
