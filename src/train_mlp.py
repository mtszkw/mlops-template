import sys
import os
import json
from joblib import dump
import numpy as np
from sklearn.neural_network import MLPClassifier
import yaml

def read_prepared_data(data_files_dir: str):
    X_train = np.load(os.path.join(data_files_dir, 'X_train.npy'))
    X_test  = np.load(os.path.join(data_files_dir, 'X_test.npy'))
    y_train = np.load(os.path.join(data_files_dir, 'y_train.npy'))
    y_test  = np.load(os.path.join(data_files_dir, 'y_test.npy'))
    return (X_train, y_train), (X_test, y_test)
 

if __name__ == "__main__":
    params = yaml.safe_load(open('params.yaml'))['train']

    if len(sys.argv) != 5:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython train.py data-files-dir model-bin-file accuracy-json-file loss-history-json-file\n")
        sys.exit(1)

    data_files_dir      = sys.argv[1]
    model_bin_file   = sys.argv[2]
    accuracy_score_file = sys.argv[3]
    loss_history_file   = sys.argv[4]
    
    (X_train, y_train), (X_test, y_test) = read_prepared_data(data_files_dir)
    
    # Training
    clf = MLPClassifier(
        random_state=params['random_state'],
        batch_size=params['batch_size'],
        learning_rate_init=params['learning_rate_init'],
        max_iter=params['max_iter'],
        validation_fraction=params['split_val'],
        early_stopping=True,
        shuffle=True,
        verbose=True)
    
    
    clf.fit(X_train, y_train)
    dump(clf, model_bin_file)

    with open(accuracy_score_file, 'w') as f:
        score = clf.score(X_test, y_test)
        print('Accuracy:', score)
        json.dump({'accuracy': score}, f)
        
    with open(loss_history_file, 'w') as f:
        loss_history = clf.loss_curve_
        print('Loss curve:', loss_history)
        json.dump({'train': [{'train_loss': l} for l in loss_history]}, f)

    
