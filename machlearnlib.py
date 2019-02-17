import numpy as np 
import pandas as pd 
import hashlib
import sklearn

# Splitting data for test, training, and validating sets
'''
split_train_test is NOT optimal
function will create different sets
'''
def split_train_test(data, test_ratio):
	shuffled_indices = np.random.permutation(len(data))
	test_set_size: int = int(len(data) * test_ratio)
	test_indices = shuffled_indices[:test_set_size]
	train_indices = shuffled_indices[test_set_size:]
	return data.iloc[train_indices], data.iloc[test_indices]

	# train_set, test_set = split_train_test(df, 0.2)


'''
split_train_test_by_id goes around the issue of different sets by
checking the last digit of the hash of each instance
and puts it in a test set if that digit is smaller than
256 * test_ratio.

Requires dataset to have unique ID. 
If none exist, df.reset_index() adds an 'index' column.

If this method is used, future data must be appended to the bottom
and rows should never be deleted. 

If keeping the data intact is not possible, you can try and use the most
stable features of the dataset to build a unique identifier.
'''
def test_set_check(identifier, test_ratio, hash):
	return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio

def split_train_test_by_id(data, test_ratio, id_column, hash=hashlib.md5):
	ids = data[id_column] 
	in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
	return data.loc[~in_test_set], data.loc[in_test_set]   
	'''
	df_with_id = df.reset_index()
	train_set, test_set = split_train_test_by_id(df_with_id, 0.2, 'index')
	'''

'''
scikit-learn provides a few function to split dataset into multiple subsets
from sklearn.model_selection import train_test_split
'''
# train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

