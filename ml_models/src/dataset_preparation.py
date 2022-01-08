# imports
import nltk
import numpy as np
import pandas as pd
from nltk import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from sklearn.preprocessing import LabelEncoder

nltk.download('wordnet')
pd.options.mode.chained_assignment = None  # default='warn'

def tokenizer(x):
 tokenizer = RegexpTokenizer(r'\w+')
 return tokenizer.tokenize(x)
 
def stemmer(x):
 stemmer = PorterStemmer()
 return ' '.join([stemmer.stem(word) for word in x])
 
def lemmatizer(x):
 lemmatizer = WordNetLemmatizer()
 return ' '.join([lemmatizer.lemmatize(word) for word in x])

# read the datasets (only the specified columns)
data0 = pd.read_csv("../data/raw/data_fec.csv")
data1 = pd.read_csv("../data/raw/data_monitoring.csv")
data2 = pd.read_csv("../data/raw/data_multipath.csv")
data3 = pd.read_csv("../data/raw/data_no_plugin.csv")

# merge the datasets to build the dataset with all classes
data = pd.DataFrame(np.concatenate((data0, data1, data2, data3), axis=0), columns=data0.columns)

data.rename(columns={"classe": "target"}, inplace=True)

# this column is not mandatory but it is used to keep the classes names
data['plugin'] = np.where(data.target == 0, "fec", np.where(data.target == 1, "monitoring", np.where(data.target == 2, "multipath", "no_plugin"))) 

# convert classes from 0 to n. eg: [1, 8, 3] --> [0, 1, 2]
enc = LabelEncoder()
data['target'] = enc.fit_transform(data.target)

# shuffle the dataset
data = data.sample(frac=1).reset_index(drop=True)

# create two datasets, one using network features, second using prints
net_stat_features = ["c_durat", "c_bytes_all", "c_pkts_all", "s_durat", "s_bytes_all", "s_pkts_all", "average_c_bytes", "average_s_bytes", "variance_out", "variance_in", "variance"]
net_stat_data = data[net_stat_features + ['plugin', 'target']]
log_data = data[['logs', 'console_out', 'plugin', 'target']]
 
## preprocess texts
# split text in tokens
print("Performing nltk tokens")
log_data['tokens'] = log_data['logs'].map(tokenizer)
log_data['tokens_console'] = log_data['console_out'].map(tokenizer)
# lemmatize each token
print("Performing nltk lemmas")
log_data['lemmas'] = log_data['tokens'].map(lemmatizer)
log_data['lemmas_console'] = log_data['tokens_console'].map(lemmatizer)
# get stems from tokens
print("Performing nltk stemming")
log_data['stems'] = log_data['tokens'].map(stemmer)
log_data['stems_console'] = log_data['tokens_console'].map(stemmer)

# save the different datasets
print("Start dumps")
net_stat_data.to_feather('../data/net_stat_data.feather')

log_data[['tokens', 'plugin', 'target']].to_feather('../data/tokens_wo_head.feather')
# add console tokens to log tokens
log_data['tokens'] = log_data['tokens_console'] + log_data['tokens']
log_data[['tokens', 'plugin', 'target']].to_feather('../data/tokens.feather')

log_data[['lemmas', 'plugin', 'target']].to_feather('../data/lemmas_wo_head.feather')
# add console lemmas to log lemmas
log_data['lemmas'] = log_data['lemmas_console'] + log_data['lemmas']
log_data[['lemmas', 'plugin', 'target']].to_feather('../data/tokens.feather')

log_data[['stems', 'plugin', 'target']].to_feather('../data/stems_wo_head.feather')
# add console stems to log stems
log_data['stems'] = log_data['stems_console'] + log_data['stems']
log_data[['stems', 'plugin', 'target']].to_feather('../data/stems.feather')
