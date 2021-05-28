import pandas as pd
import numpy as np
import pickle
import nltk
from nltk.tokenize import TweetTokenizer
from features import get_transformer, prepare_entry
from tqdm import  tqdm
import json
import re
import os
import string
import argparse
import sys

MODEL_FILE = 'temp_data/large_model_training_data/large_model.p'

def process_batch(transformer, scaler, secondary_scaler, clf, ids, preprocessed_docs1, preprocessed_docs2, output_file):
    print('Extracting features:', len(ids), file=sys.stderr)
    X1 = scaler.transform(transformer.transform(preprocessed_docs1).todense())
    X2 = scaler.transform(transformer.transform(preprocessed_docs2).todense())
    X = secondary_scaler.transform(np.abs(X1 - X2))
    print('Predicting...', file=sys.stderr)
    probs = clf.predict_proba(X)[:, 1]
    print('Writing to', output_file, file=sys.stderr)
    with open(output_file, 'a') as f:
        for i in range(len(ids)):
            d = {
                'id': ids[i],
                'value': probs[i]
            }
            json.dump(d, f)
            f.write('\n')
            
def process_single_entry(transformer, scaler, secondary_scaler, clf, idx, preprocessed_doc1, preprocessed_doc2, f_output_file):    
    try:
        X1 = scaler.transform(transformer.transform([preprocessed_doc1]).todense())
        X2 = scaler.transform(transformer.transform([preprocessed_doc2]).todense())
        X = secondary_scaler.transform(np.abs(X1 - X2))
        prob = clf.predict_proba(X)[0, 1]
    except Exception as e:
        print('Exception predicting:', e)
        prob = 0.5
    d = {
        'id': idx,
        'value': prob
    }
    json.dump(d, f_output_file)
    f_output_file.write('\n')

    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prediction Script: PAN 2021 - Janith Weerasinghe')
    parser.add_argument('-i', type=str,
                        help='Evaluaiton dir')
    parser.add_argument('-o', type=str, 
                        help='Output dir')
    args = parser.parse_args()
    
    # validate:
    if not args.i:
        raise ValueError('Eval dir path is required')
    if not args.o:
        raise ValueError('Output dir path is required')
        
        
    input_file = os.path.join(args.i, 'pairs.jsonl')
    output_file = os.path.join(args.o, 'answers.jsonl')
    print("Writing answers to:", output_file , file=sys.stdout, flush=True)
    

    with open(MODEL_FILE, 'rb') as f:
        clf, transformer, scaler, secondary_scaler = pickle.load(f)

    with open(input_file, 'r') as f, open(output_file, 'w') as f_output_file:
        i = 0
        for l in tqdm(f):
            if i % 100 == 0:
                print(i, flush=True)
            i += 1
            d = json.loads(l)
            idx = d['id']
            preprocessed_doc1 = prepare_entry(d['pair'][0], mode='accurate', tokenizer='casual')
            preprocessed_doc2 = prepare_entry(d['pair'][1], mode='accurate', tokenizer='casual')
            process_single_entry(transformer, scaler, secondary_scaler, clf, idx, preprocessed_doc1, preprocessed_doc2, f_output_file) 
            
    print("Execution complete", file=sys.stderr)
                