import os
import pickle
import json
import glob
from tqdm.auto import trange, tqdm
import sys
import numpy as np
from features import prepare_entry

'''
DATA_DIR = '/media/disk1/social/troll_tweets/data/pan_clustering/pan20_large/'
GROUND_TRUTH_PATH = DATA_DIR + 'pan20-authorship-verification-training-large-truth.jsonl'
DATA_PATH = DATA_DIR + 'pan20-authorship-verification-training-large.jsonl'
TEMP_DATA_PATH = 'temp_data/large_model_training_data/'


DATA_DIR = '/scratch/jnw301/av/data/pan20_large/'
GROUND_TRUTH_PATH = DATA_DIR + 'pan20-authorship-verification-training-large-truth.jsonl'
DATA_PATH = DATA_DIR + 'pan20-authorship-verification-training-large.jsonl'
TEMP_DATA_PATH = '/scratch/jnw301/pan2021_av/temp_data/large_model_training_data/'
'''

DATA_DIR = '/scratch/jnw301/av/data/pan20_large/'
GROUND_TRUTH_PATH = DATA_DIR + 'pan20-authorship-verification-training-large-truth.jsonl'
DATA_PATH = DATA_DIR + 'pan20-authorship-verification-training-large.jsonl'
TEMP_DATA_PATH = '/scratch/jnw301/pan2021_av/temp_data/large_model_training_data/'


NUM_MACHINES = 20


if __name__ == "__main__":
    instance_id = int(sys.argv[1])
    print('Instance ID for this machine:', instance_id, flush=True)
    
    train_ids, test_ids, _, _ = pickle.load(open(TEMP_DATA_PATH + 'dataset_partition.p', 'rb'))
    
    total_recs = len(train_ids) + len(test_ids)
    job_sz = total_recs // NUM_MACHINES
    start_rec = instance_id * job_sz
    end_rec = (instance_id + 1) * job_sz
    
    print('Recs on this machine:', (end_rec - start_rec), flush=True)
    i = 0
    with open(DATA_PATH, 'r') as f, \
        open(TEMP_DATA_PATH + 'preprocessed_train_' + str(instance_id) + '.jsonl', 'w') as f_train, \
        open(TEMP_DATA_PATH + 'preprocessed_test_' + str(instance_id) + '.jsonl', 'w') as f_test:
        for l in tqdm(f, total=total_recs):
            i += 1
            if i < start_rec or i > end_rec:
                continue
            d = json.loads(l)
            preprocessed = {
                'id': d['id'],
                'fandoms': d['fandoms'],
                'pair': [
                    prepare_entry(d['pair'][0], mode='accurate', tokenizer='casual'),
                    prepare_entry(d['pair'][1], mode='accurate', tokenizer='casual')
                ]
            }
            if d['id'] in train_ids:
                json.dump(preprocessed, f_train)
                f_train.write('\n')
            else:
                json.dump(preprocessed, f_test)
                f_test.write('\n')