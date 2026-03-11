from sklearn.neural_network import MLPClassifier
import time
import numpy as np

def mlp(X,y,n_hidden_layers,hidden_dim,epochs=50,batch_size=64,seed=42):
    mlp = MLPClassifier(hidden_layer_sizes=(n_hidden_layers, hidden_dim),learning_rate='adaptive',batch_size=batch_size ,activation='identity', solver='lbfgs', max_iter=epochs, random_state=seed)
    return mlp.fit(X, y)