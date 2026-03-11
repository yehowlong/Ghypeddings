# G-Hypeddings

## 1. Overview

G-hypeddings is a **Python library** designed for **graph hyperbolic embeddings**, It provides a complete pipeline that enables the use of these embeddings for **anomaly detection** in graphs. It includes 06 distinct models with various configurations, all of which utilize **hyperbolic geometry** for their operations. The library is built on top of the [PyTorch framework](https://pytorch.org/).

### 1.1. Models

The models can be divided into three main categories based on the model's overall architecture namely Shallow models (Poincaré), Convolutional-based models (HGCN & HGNN), and Autoencoder-based models (HGCAE & PVAE).

| Name     | Year     | Encoder  | Decoder | Manifold                  | Ref   |
|----------|----------|----------|---------|---------------------------|-------|
| Poincaré | 2017     | /        | MLP     | Poincaré Ball             | [1]   |
| HGNN     | 2019     | HGCN     | MLP     | Poincaré Ball, Lorentz    | [2]   |
| HGCN     | 2019     | HGCN     | MLP     | Lorentz                   | [3]   |
| P-VAE    | 2019     | GCN      | MLP     | Poincaré Ball             | [4]   |
| H2H-GCN  | 2021     | HGCN     | MLP     | Lorentz                   | [5]   |
| HGCAE    | 2021     | HGCN     | HGCN    | Poincaré Ball             | [6]   |

In this library, we also provide a variety of binary classifiers, clustering algorithms, and unsupervised anomaly detection algorithms to use with the autoencoder-based models (HGCAE & PVAE). All of these are [Scikit-learn](https://scikit-learn.org/) models tuned using the Grid-Search technique.

| Name                                        | Type                        |
|---------------------------------------------|-----------------------------|
| Support Vector Machine (SVM)                | Binary Classifier           |
| Multilayer Perceptrone (MLP)                | Binary Classifier           |
| Decision Tree                               | Binary Classifier           |
| Random Forest                               | Binary Classifier           |
| AdaBoost                                    | Binary Classifier           |
| K-Nearest Neighbors (KNN)                   | Binary Classifier           |
| Naive Bayes                                 | Binary Classifier           |
| Agglomerative Hierarchical Clustering (AHC) | Clustering Algorithm        |
| DBSCAN                                      | Clustering Algorithm        |
| Fuzzy C mean                                | Clustering Algorithm        |
| Gaussian Mixture                            | Clustering Algorithm        |
| k-Means                                     | Clustering Algorithm        |
| Mean shift                                  | Clustering Algorithm        |
| Isolation Forest                            | Anomaly Detection Algorithm |
| One-class SVM                               | Anomaly Detection Algorithm |
| Local Outlier Factor                        | Anomaly Detection Algorithm |
| DBSCAN                                      | Anomaly Detection Algorithm |
| k-Means                                     | Anomaly Detection Algorithm |

### 1.2. Datasets

To evaluate hyperbolique embeddings on anomaly detection, we use the following known datasets. 
Due to usage restrictions, this library provides only a single graph of each dataset, with 5,000 nodes, already pre-processed and normalized.

| Name                   | Ref   |
|------------------------|-------|
|  Darknet               | [7]   |
|  CICDDoS2019           | [8]   |
|  DGraphFin             | [9]   |
|  Elliptic              | [10]  |
|  Cora                  | [11]  |
|  YelpNYC               | [12]  |


## 2. Installation

```bash
git clone https://gitlab.liris.cnrs.fr/gladis/ghypeddings.git
mv ghypeddings\ Ghypeddings\
```

## 3. Usage

Training and evaluating a model using our library is done in lines of code only!

### 3.1. Models
```python
from Ghypeddings import PVAE

# adj: adjacency matrix 
# features: node features matrix
model = PVAE(adj=adj,
             features=features,
             labels=labels,
             dim=20,
             hidden_dim=features.shape[1],
             test_prop=.2,
             val_prop=.1,
             epochs=50,
             classifier='random forest')

# fit the model and outputs the training scores
loss, accuracy, f1,recall,precision,roc_auc,training_time = model.fit()
# prediction scores
loss,acc,f1,recall,precision,roc_auc = model.predict()
```
### 3.2. Datasets

```python
from Ghypeddings import Darknet

# Build a graph of 5000 nodes from the Darknet dataset
adj ,features ,labels = Darknet().build(n_nodes = 5000)

# The graph is already loaded automatically after executing the previous line of code
# This method saves time and helps comparing results
# it simply loads graphs built and saved previously
adj, features, labels = Darknet().load_samples()
```

## 5. References

[1]: [Nickel, Maximillian, and Douwe Kiela. "Poincaré embeddings for learning hierarchical representations." Advances in neural information processing systems 30 (2017).](https://proceedings.neurips.cc/paper_files/paper/2017/hash/59dfa2df42d9e3d41f5b02bfc32229dd-Abstract.html)

[2]: [Liu, Qi, Maximilian Nickel, and Douwe Kiela. "Hyperbolic graph neural networks." Advances in neural information processing systems 32 (2019).](https://proceedings.neurips.cc/paper/2019/hash/103303dd56a731e377d01f6a37badae3-Abstract.html)

[3]: [Chami, Ines, et al. "Hyperbolic graph convolutional neural networks." Advances in neural information processing systems 32 (2019).](https://proceedings.neurips.cc/paper_files/paper/2019/hash/0415740eaa4d9decbc8da001d3fd805f-Abstract.html)

[4]: [Mathieu, Emile, et al. "Continuous hierarchical representations with poincaré variational auto-encoders." Advances in neural information processing systems 32 (2019).](https://proceedings.neurips.cc/paper/2019/hash/0ec04cb3912c4f08874dd03716f80df1-Abstract.html)

[5]: [Dai, Jindou, et al. "A hyperbolic-to-hyperbolic graph convolutional network." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2021.](https://www.computer.org/csdl/proceedings-article/cvpr/2021/450900a154/1yeJgfbgw6Y)

[6]: [Park, Jiwoong, et al. "Unsupervised hyperbolic representation learning via message passing auto-encoders." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021.](https://ieeexplore.ieee.org/document/9577649)

[7]: Darknet dataset. Available at: [https://dl.acm.org/doi/10.1145/3442520.3442521]

[8]: CIC-DDoS2019 dataset. Available at: [https://www.unb.ca/cic/datasets/ddos-2019.html]

[9]: DGraphFin  dataset. Available at: [https://arxiv.org/abs/2207.03579]

[10]: Elliptic dataset. Available at: [https://medium.com/elliptic/the-elliptic-data-set-opening-up-machine-learning-on-the-blockchain-e0a343d99a14]

[11]: Cora dataset. Available at: [https://graphsandnetworks.com/the-cora-dataset]

[12]: YelpNYC dataset. Available at: [https://www.dgl.ai/dgl_docs/generated/dgl.data.FraudYelpDataset.html]