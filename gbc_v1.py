
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier


# load training data
df = pd.read_csv('training.csv')


# map y values to integers
df['Label'] = df['Label'].map({'b':0, 's':1})

# rearrange columns for convenience
cols = df.columns.tolist()
cols = [cols[-1]] + cols[:-1]
df = df[cols]

# convert into numpy array
train_data = df.values


# pick a random seed for reproducible results
np.random.seed(42)

# random number for training/validation splitting
r = np.random.rand(data_train.shape[0])


# call model
gbc = GradientBoostingClassifier(n_estimators=50, max_depth=10,
                                    min_samples_leaf=200,
                                    max_features=10, verbose=1)

# train model
gbc.fit(train_data[:,2:32][r<0.9], train_data[:,0][r<0.9])


# load test data
df_test = pd.read_csv('test.csv')

# convert into numpy array
test_data = df_test.values

# predict probability of classification
output = gbc.predict_proba(test_data[:,1:])[:,1]

# output to pandas dataframe
result = np.c_[test_data[:,0], output]
df_result = pd.DataFrame(result, columns=['EventId', 'Class'])

# sort by probability in descending order, then re-index
df_result = df_result.sort(columns=['Class'], ascending=False)
df_result.index = range(0, len(df_result))

# map top 15% to signal, rest to background
signal_count = int(len(df_result) * 0.15)
df_result['Class'] = np.where(df_result.index < signal_count, 's', 'b')


# prepare for submission
df_result['EventId'] = df_result['EventId'].map(lambda x: int(x))
df_result['RankOrder'] = range(len(df_result), 0, -1)
df_result = df_result[['EventId', 'RankOrder', 'Class']]

# print result to csv
df_result.to_csv('gbc_v1.csv', index=False)
