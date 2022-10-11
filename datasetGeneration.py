with open('.../A_Concise_Guide_to_Statistics.csv', 'rb') as data:
     df1 = csv.reader(data)
with open('.../A_modern_introduction_to_probability.csv', 'rb') as data:
     df2 = csv.reader(data)
with open('.../Modern_Mathematical_Statistics_With_Applications.csv', 'rb') as data:
     df3 = csv.reader(data)
with open('.../Openintro_Statistics.csv', 'rb') as data:
     df4 = csv.reader(data)
with open('.../Statistics_and_Probability_Theory.csv', 'rb') as data:
     df5 = csv.reader(data)
with open('...Statistics_for_Non_Statisticians.csv', 'rb') as data:
     df6 = csv.reader(data)
with open('...Probability_and_statistics_for_engineers_and_scientists.csv', 'rb') as data:
     df7 = csv.reader(data)
with open('...Statistics_for_Scientists_and_Engineers.csv', 'rb') as data:
     df8 = csv.reader(data)
with open('...Introductory_Statistics_With_R.csv', 'rb') as data:
     df9 = csv.reader(data)
    
# add datasets in the train and test lists
train = [...]
test = [...] 

train_df = shuffle(pd.concat(train))
test_df = shuffle(pd.concat(test))

BATCH_SIZE = CONFIG.train_batch
batch_train_df = dataBatcher(train_df, BATCH_SIZE)
batch_test_df = dataBatcher(test_df, BATCH_SIZE)
train_features, test_features, train_labels, test_labels = batch_train_df['Sentence'], batch_test_df['Sentence'], batch_train_df[['Tags', 'Relevance']], batch_test_df[['Tags', 'Relevance']]
