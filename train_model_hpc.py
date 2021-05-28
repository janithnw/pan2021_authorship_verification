import pickle
import numpy as np
from tqdm.auto import trange, tqdm
from features import get_transformer, merge_entries
import json
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc
from utills import chunker
from sklearn.utils.fixes import loguniform
from sklearn.model_selection import RandomizedSearchCV

DATA_DIR = '/scratch/jnw301/av/data/pan20_large/'
GROUND_TRUTH_PATH = DATA_DIR + 'pan20-authorship-verification-training-large-truth.jsonl'
DATA_PATH = DATA_DIR + 'pan20-authorship-verification-training-large.jsonl'
TEMP_DATA_PATH = '/scratch/jnw301/pan2021_av/temp_data/large_model_training_data/'
PREPROCESSED_DATA_PATH = '/scratch/jnw301/pan2021_av/temp_data/large_model_training_data/'


def fit_transformers(data_fraction=0.01):
    docs_1 = []
    docs_2 = []

    with open(PREPROCESSED_DATA_PATH + 'preprocessed_train.jsonl', 'r') as f:
        for l in tqdm(f):
            if np.random.rand() < data_fraction:
                d = json.loads(l)
                docs_1.append(d['pair'][0])
                docs_2.append(d['pair'][1])
                
    transformer = get_transformer()
    scaler = StandardScaler()
    secondary_scaler = StandardScaler()

    X = transformer.fit_transform(docs_1 + docs_2).todense()
    X = scaler.fit_transform(X)

    X1 = X[:len(docs_1)]
    X2 = X[len(docs_1):]
    secondary_scaler.fit(np.abs(X1 - X2))
    
    return transformer, scaler, secondary_scaler, np.abs(X1 - X2)

def vectorize(XX, Y, ordered_idxs, transformer, scaler, secondary_scaler, preprocessed_path, vector_sz):
    with open(preprocessed_path, 'r') as f:
        batch_size = 5000
        i = 0;
        docs1 = []
        docs2 = []
        idxs = []
        labels = []
        for l in tqdm(f, total=vector_sz):
            d = json.loads(l)
            docs1.append(d['pair'][0])
            docs2.append(d['pair'][1])

            labels.append(ground_truth[d['id']])
            idxs.append(ordered_idxs[i])
            i += 1
            if len(labels) >= batch_size:
                x1 = scaler.transform(transformer.transform(docs1).todense())
                x2 = scaler.transform(transformer.transform(docs2).todense())
                XX[idxs, :] = secondary_scaler.transform(np.abs(x1-x2))
                Y[idxs] = labels

                docs1 = []
                docs2 = []
                idxs = []
                labels = []

        x1 = scaler.transform(transformer.transform(docs1).todense())
        x2 = scaler.transform(transformer.transform(docs2).todense())
        XX[idxs, :] = secondary_scaler.transform(np.abs(x1-x2))
        Y[idxs] = labels
        XX.flush()
        Y.flush()
        
        
if __name__ == "__main__":
    ground_truth = {}
    with open(GROUND_TRUTH_PATH, 'r') as f:
        for l in f:
            d = json.loads(l)
            ground_truth[d['id']] = d['same']
    
    
    train_sz = 206445
    test_sz = 69054
    '''
    with open(PREPROCESSED_DATA_PATH + 'preprocessed_train.jsonl', 'r') as f:
        for l in f:
            train_sz += 1

    with open(PREPROCESSED_DATA_PATH + 'preprocessed_test.jsonl', 'r') as f:
        for l in f:
            test_sz += 1
    '''
    print('Train Sz:', train_sz, flush=True)
    print('Test Sz:', test_sz, flush=True)
    
    print('Fitting transformer...', flush=True)
    transformer, scaler, secondary_scaler, sampled_X = fit_transformers(data_fraction=0.1)
    feature_sz = len(transformer.get_feature_names())
    
    with open(TEMP_DATA_PATH + 'experiment_data.p', 'rb') as f:
        (
            transformer, 
            scaler,
            secondary_scaler,
            feature_sz,
            train_sz,
            test_sz
        ) = pickle.load(f)
        
        
    print('Vectorizing train set...', flush=True)
    XX_train = np.memmap(TEMP_DATA_PATH + 'vectorized_XX_train.npy', dtype='float32', mode='w+', shape=(train_sz, feature_sz))
    Y_train = np.memmap(TEMP_DATA_PATH + 'Y_train.npy', dtype='int32', mode='w+', shape=(train_sz))
    train_idxs = np.array(range(train_sz))
    np.random.shuffle(train_idxs)

    vectorize(
        XX_train, 
        Y_train, 
        train_idxs, 
        transformer, 
        scaler, 
        secondary_scaler, 
        PREPROCESSED_DATA_PATH + 'preprocessed_train.jsonl',
        train_sz
    )
   
    with open(TEMP_DATA_PATH + 'experiment_data_1.p', 'rb') as f:
        (
            transformer, 
            scaler,
            secondary_scaler,
            feature_sz,
            train_sz,
            train_idxs,
            test_sz
        ) = pickle.load(f) 
        
    print('Vectorizing test set...', flush=True)
    XX_test = np.memmap(TEMP_DATA_PATH + 'vectorized_XX_test.npy', dtype='float32', mode='w+', shape=(test_sz, feature_sz))
    Y_test = np.memmap(TEMP_DATA_PATH + 'Y_test.npy', dtype='int32', mode='w+', shape=(test_sz))
    test_idxs = np.array(range(test_sz))
    np.random.shuffle(test_idxs)

    vectorize(
        XX_test, 
        Y_test, 
        test_idxs, 
        transformer, 
        scaler, 
        secondary_scaler, 
        PREPROCESSED_DATA_PATH + 'preprocessed_test.jsonl',
        test_sz
    )
    
        
    print('Tuning parameters...', flush=True)


    param_dist = {'alpha': loguniform(1e-4, 1e0)}
    batch_size=10000
    clf = SGDClassifier(loss='log', alpha=0.01)
    n_iter_search = 20
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=n_iter_search, verbose=2)
    for idxs in chunker(range(train_sz), batch_size):
            random_search.fit(XX_train[idxs, :], Y_train[idxs])
            break

    print('Best params:', random_search.best_params_)
    

    print('Training classifier...', flush=True)
    clf = SGDClassifier(loss='log', alpha=random_search.best_params_['alpha'])
    batch_size=5000
    num_epochs = 200
    aucs = []
    for i in trange(num_epochs):
        print('Epoch - ', i)
        print('-' * 30)
        # clf.partial_fit(XX_test, Y_test) # Merge in test set too
        for idxs in chunker(range(train_sz), batch_size):
            clf.partial_fit(XX_train[idxs, :], Y_train[idxs], classes=[0, 1])
        
        
        probs = clf.predict_proba(XX_test)[:, 1]
        fpr, tpr, thresh = roc_curve(Y_test, probs)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        print('AUC: ', roc_auc)
        with open(TEMP_DATA_PATH + 'experiment_data.p', 'wb') as f:
            pickle.dump((
                aucs,
                clf,
                roc_auc,
                transformer, 
                scaler,
                secondary_scaler,
                feature_sz,
                train_sz,
                train_idxs,
                test_sz,
                test_idxs
            ), f)