import pandas as pd
import numpy as np
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split
from category_encoders import LeaveOneOutEncoder
from numpy import genfromtxt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler

# prepossing

def remove_unused_column(data):
    # Unused column is the column that have only one unique value
    unused_list = []
    for col in data.columns:
        uni = len(data[col].unique())
        if uni <= 1:
            unused_list.append(col)
    data.drop(columns=unused_list, inplace=True)
    return data

def split_data(data, target, test_size):
    # Split the data into two sets: train and test
    label = data[target]
    data = data.drop([target], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=test_size, random_state=123, shuffle=True)
    return X_train, y_train.values, X_test, y_test.values


def quantile_transform(X_train, X_test):
    # Use quantile transform to make the data more normal distribution
    quantile_train = np.copy(X_train)
    qt = QuantileTransformer(random_state=55688, output_distribution='normal').fit(quantile_train)
    X_train = qt.transform(X_train)
    X_test = qt.transform(X_test)

    return X_train, X_test

def add_missing_columns(d, columns) :
    # Add missing columns into the dataframe
    missing_col = set(columns) - set(d.columns)
    for col in missing_col :
        d[col] = 0
        
def fix_columns(d, columns):  
    # Fix the column order of the dataframe
    add_missing_columns(d, columns)
    assert(set(columns) - set(d.columns) == set())
    d = d[columns]
    return d

#dataset

def arrhythmia():
    data = genfromtxt('./data/arrhythmia', delimiter=',',dtype=float, missing_values='?', filling_values=0.0 )
    # Identify classes with fewer than 10 samples
    unique, counts = np.unique(data[:,-1], return_counts=True)
    underrepresented_classes = unique[counts < 10]

    # Filter out samples from underrepresented classes
    filtered_data = data[~np.isin(data[:,-1], underrepresented_classes)]

    # Verify the correctness
    unique_filtered, counts_filtered = np.unique(filtered_data[:,-1], return_counts=True)
    trainset, testset = train_test_split(filtered_data, random_state=42)
    X_train = trainset[:,:-1]
    y_train = trainset[:,-1]
    X_test = testset[:,:-1]
    y_test = testset[:,-1]
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)
    X_train, X_test = quantile_transform(X_train, X_test)
    return X_train, y_train, X_test, y_test
def kdd():
    # load data and column names
    features = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent","hot",
           "num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root","num_file_creations","num_shells",
           "num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
           "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count", 
           "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
           "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty"]
    trainset = pd.read_csv('./data/kdd/KDDTrain+.csv',names=features)
    testset = pd.read_csv('./data/kdd/KDDTest+.csv',names=features)

    # change label
    def change_label(df):
        df.label.replace(['apache2','back','land','neptune','mailbomb','pod','processtable','smurf','teardrop','udpstorm','worm'],'Dos',inplace=True)
        df.label.replace(['ftp_write','guess_passwd','httptunnel','imap','multihop','named','phf','sendmail','snmpgetattack','snmpguess','spy','warezclient','warezmaster','xlock','xsnoop'],'R2L',inplace=True)      
        df.label.replace(['ipsweep','mscan','nmap','portsweep','saint','satan'],'Probe',inplace=True)
        df.label.replace(['buffer_overflow','loadmodule','perl','ps','rootkit','sqlattack','xterm'],'U2R',inplace=True)
    change_label(trainset), change_label(testset)

    
    # Drop the difficulty column from both the training and test data
    trainset.drop(['difficulty'],axis=1,inplace=True)
    testset.drop(['difficulty'],axis=1,inplace=True)
    # Split the data into X and y components
    X_train = trainset.iloc[:,:41]
    y_train = trainset.iloc[:,-1]
    X_test = testset.iloc[:,:41]
    y_test = testset.iloc[:,-1]
    # Encode the categorical features using OrdinalEncoder
    ode = OrdinalEncoder()
    X_train_ode = ode.fit_transform(X_train[['protocol_type', 'service', 'flag']])
    X_test_ode = ode.fit_transform(X_test[['protocol_type', 'service', 'flag']])
    # Combine the encoded features with the original data
    X_train = pd.concat([X_train.drop(columns=['protocol_type', 'service', 'flag']),
                                pd.DataFrame(X_train_ode,columns=['protocol_type', 'service', 'flag'])], axis=1)
    X_test = pd.concat([X_test.drop(columns=['protocol_type', 'service', 'flag']),
                                pd.DataFrame(X_test_ode,columns=['protocol_type', 'service', 'flag'])], axis=1)
    # Encode the labels using LabelEncoder
    le = LabelEncoder()
    y_train = le.fit_transform(y_train.values.ravel())
    y_test = le.fit_transform(y_test.values.ravel())
    # Convert the data into numpy arrays
    X_train = X_train.values
    X_test = X_test.values
    # Scale the data using StandardScaler
    scale = StandardScaler()
    X_train = scale.fit_transform(X_train)
    X_test = scale.transform(X_test)
    return X_train, y_train, X_test, y_test
def adult():
    features = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship',  'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
    df_train = pd.read_csv('./data/adult/adult.csv', names = features)
    df_test = pd.read_csv('./data/adult/adult_test.csv', skiprows = 1, names = features)
    def data_process(df, model) :
        df.replace(" ?", pd.NaT, inplace = True)
        if model == 'train' :
            df.replace(" >50K", 1, inplace = True)
            df.replace(" <=50K", 0, inplace = True)
        if model == 'test':
            df.replace(" >50K.", 1, inplace = True)
            df.replace(" <=50K.", 0, inplace = True)
        trans = {'workclass' : df['workclass'].mode()[0], 'occupation' : df['occupation'].mode()[0], 'native-country' : df['native-country'].mode()[0]}
        df.fillna(trans, inplace = True)
        
        target = df["income"]
        df = df.drop('income', axis=1)
        
        ode = OrdinalEncoder()

        df_object_col = df.select_dtypes(include=['object']).columns.tolist()
        for i in range(len(df_object_col)):
            df[df_object_col[i]] = ode.fit_transform(df[df_object_col[i]].values.reshape(-1,1))
        return target, df
    train_target, train_dataset = data_process(df_train, 'train')
    test_target, test_dataset = data_process(df_test, 'test')

    test_dataset = fix_columns(test_dataset, train_dataset.columns)
    columns = train_dataset.columns

    train_target, test_target = np.array(train_target), np.array(test_target)
    train_dataset, test_dataset = np.array(train_dataset), np.array(test_dataset)
    
    # scale = MinMaxScaler()
    scale = StandardScaler()
    train_dataset = scale.fit_transform(train_dataset)
    test_dataset = scale.transform(test_dataset)
        
    return train_dataset, train_target,test_dataset, test_target
def crowdsourcemap():
    train_data = genfromtxt('./data/crowdsourcemap/train_crowdsourcemap.csv', dtype=str, skip_header=1, delimiter=',' )
    test_data = genfromtxt('./data/crowdsourcemap/test_crowdsourcemap.csv', dtype=str, skip_header=1, delimiter=',' )
    X_train = train_data[:,1:].astype(float)
    y_train = train_data[:,0]
    X_test = test_data[:,1:].astype(float)
    y_test = test_data[:,0]

    scale = StandardScaler()
    X_train = scale.fit_transform(X_train)
    X_test = scale.transform(X_test)

    le = LabelEncoder()

    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)
    return X_train, y_train, X_test, y_test
def academic():
    # Load the data
    data = pd.read_csv('./data/academic.csv', sep=';')

    # Separate features and target
    X = data.drop(columns=['Target'])
    y = data['Target']

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split data set
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
    return X_train, y_train, X_test, y_test
def metro():
    df=pd.read_csv("./data/MetroPT3(AirCompressor).csv")
    df['label'] = 0
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    start_time = pd.Timestamp("2020-04-18 00:00:00")
    end_time = pd.Timestamp("2020-04-18 23:59:00")
    df.loc[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time), 'label'] = 1

    start_time = pd.Timestamp("2020-05-29 23:30:00")
    end_time = pd.Timestamp("2020-05-30 06:00:00")
    df.loc[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time), 'label'] = 1

    start_time = pd.Timestamp("2020-06-05 10:00:00")
    end_time = pd.Timestamp("2020-06-07 14:30:00")
    df.loc[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time), 'label'] = 1

    start_time = pd.Timestamp("2020-07-15 14:30:00")
    end_time = pd.Timestamp("2020-07-15 19:00:00")
    df.loc[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time), 'label'] = 1

    df_train = df[df['timestamp'].dt.month < 5]
    df_test = df[df['timestamp'].dt.month >= 5]
    columns_to_drop = ['timestamp', 'Unnamed: 0']
    df_train = df_train.drop(columns=columns_to_drop)
    df_test = df_test.drop(columns=columns_to_drop)

    X_train = df_train.drop(columns=['label']).values
    y_train = df_train['label'].values
    X_test = df_test.drop(columns=['label']).values
    y_test = df_test['label'].values
    return X_train, y_train,X_test, y_test
def accelerometer():
    # Read entire data set
    data = pd.read_csv('./data/accelerometer_gyro_mobile_phone_dataset.csv')

    # Remove timestamp
    data.drop(columns=['timestamp'], inplace=True)

    # Standardize features
    features = ['accX', 'accY', 'accZ', 'gyroX', 'gyroY', 'gyroZ']
    scaler = StandardScaler()
    data[features] = scaler.fit_transform(data[features])

    # Encode labels
    label_encoder = LabelEncoder()
    data['Activity'] = label_encoder.fit_transform(data['Activity'])

    # Split data set
    X = data.drop(columns=['Activity']).values
    y = data['Activity'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, y_train, X_test, y_test
def diabetes():
    # Load the dataset
    data_path = "./data/diabetic_data.csv"  # Replace with your file path
    diabetic_data = pd.read_csv(data_path)

    # Replace special characters like '?' with NaN for proper identification of missing values
    diabetic_data.replace('?', np.nan, inplace=True)

    # Drop columns with a high percentage of missing values
    columns_to_drop = ['weight', 'medical_specialty', 'payer_code']
    diabetic_data.drop(columns=columns_to_drop, inplace=True)

    # Fill missing values with the most frequent value in each column
    columns_to_fill = ['race', 'diag_3', 'diag_2', 'diag_1']
    for column in columns_to_fill:
        most_frequent_value = diabetic_data[column].mode()[0]
        diabetic_data[column].fillna(most_frequent_value, inplace=True)

    # Identify numerical columns for standardization
    numerical_columns = diabetic_data.select_dtypes(include=[np.number]).columns.tolist()
    numerical_columns = [col for col in numerical_columns if col not in ['encounter_id', 'patient_nbr', 'readmitted']]

    # Initialize the StandardScaler
    scaler = StandardScaler()

    # Perform standardization on the numerical columns
    diabetic_data[numerical_columns] = scaler.fit_transform(diabetic_data[numerical_columns])

    # Identify categorical columns for label encoding
    categorical_columns = diabetic_data.select_dtypes(include=[object]).columns.tolist()

    # Initialize the LabelEncoder
    label_encoder = LabelEncoder()

    # Perform label encoding on the categorical columns
    for column in categorical_columns:
        diabetic_data[column] = label_encoder.fit_transform(diabetic_data[column])

    # Separate features and target variable from the dataset
    X = diabetic_data.drop(columns=['readmitted', 'encounter_id', 'patient_nbr'])
    y = diabetic_data['readmitted']

    # Split the data into training and testing sets (80% training and 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert the DataFrames to numpy arrays for machine learning models
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()
    return X_train, y_train, X_test, y_test

def get_data(datasetname):

    if datasetname == 'arrhythmia':
        return arrhythmia()
    elif datasetname == 'kdd':
        return kdd()
    elif datasetname == 'adult':
        return adult()
    elif datasetname == 'academic':
        return academic()
    elif datasetname == 'metro':
        return metro()
    elif datasetname == 'accelerometer':
        return accelerometer()
    elif datasetname == 'diabetes':
        return diabetes()
    elif datasetname == 'crowdsourcemap':
        return crowdsourcemap()


# if __name__ == '__main__':
#     adult()
