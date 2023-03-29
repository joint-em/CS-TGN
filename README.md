# CS-TGN
This repository contains the implementation of algorithms, and used datasets of paper ``CS-TGN: Community Search via Temporal Graph Neural Networks`` (ACM The Web Conference Companion 2023  (WWW)). 


#### Authors: [Farnoosh Hashemi](https://farnooshha.github.io//), [Ali Behrouz](https://abehrouz.github.io/), [Milad Rezaei Hajidehi](https://www.linkedin.com/in/milad-rezaei-hajidehi-63013773/)
#### [Link to the paper](https://doi.org/10.1145/3543873.3587654) ([Arxiv](https://arxiv.org/abs/2303.08964))
#### [Poster]()
#### [Brief video explanation]()



### Abstract
----------------  
A key graph mining primitive is extracting dense structures from graphs, and this has led to interesting notions such as k-cores which subsequently have been employed as building blocks for capturing the structure of complex networks and for designing efficient approximation algorithms for challenging problems such as finding the densest subgraph. In applications such as biological, social, and transportation networks, interactions between objects span multiple aspects. Multilayer (ML) networks have been proposed for accurately modeling such applications. In this paper, we present FirmCore, a new family of dense subgraphs in ML networks, and show that it satisfies many of the nice properties of k-cores in single-layer graphs. Unlike the state of the art core decomposition of ML graphs, FirmCores have a polynomial time algorithm, making them a powerful tool for understanding the structure of massive ML networks. We also extend FirmCore for directed ML graphs. We show that FirmCores and directed FirmCores can be used to obtain efficient approximation algorithms for finding the densest subgraphs of ML graphs and their directed counterparts. Our extensive experiments over several real ML graphs show that our FirmCore decomposition algorithm is significantly more efficient than known algorithms for core decompositions of ML graphs. Furthermore, it returns solutions of matching or better quality for the densest subgraph problem over (possibly directed) ML graphs.




#### Dependecies & Hardware
To run our codes you need to have these packages installed:

```
torch==1.13.0
torch-geometric==2.2.0
torch-scatter==2.1.0
torch-sparse==0.6.15
torch-spline-conv==1.2.1
torchmetrics==0.11.0
```

A single GPU should be enough to train models. For large datasets we capped at most 5GBs of GPU memory.  

#### Components
The core component is the models impelemented in `models/` directory. You can find the vanilla CSTGN and CSTGN without GRU as well as attention module. Modelwise, you may enhance the architecture for particular datasets by changing the attention snapshots, number of GCN layers and changing the hidden dimension of models. 

The `train.py` can be used to trained the RNN network of CSTGN. Please read the Dataset Structure for instruction for using it. 

The `dataset.py` has two classes `QueryFile` and `Snapshot` for representing the query/answer pairs and temporal view of graph at a specific time. 

The `util/` directory contains some codes to parse specific datasets and generate queries. 

#### Dataset Structure
The graphs and query/answer sets can be in any format but in order to use our train script and dataset utilities these should be in following format. 

##### File Structure
Our training script assume that the data stored in the following format:
```
data/
    dataset_name/
        graph_{snapshot_id}.txt
        graph_{snapshot_id}.txt
        queries_{snapshot_id}.txt
        queries_{snapshot_id}.txt
```

So for example for a a dataset named `foo` with 3 snapshots it would be like:

```
data/
    foo/
        graph_1.txt
        graph_2.txt
        graph_3.txt
        queries_1.txt
        queries_2.txt
        queries_3.txt
        test_queries.txt
        valid_queries.txt
```

##### Graph file and QueryFiles

The graph files must have the following format. 
```
node_id1 node_id2
..
..
```

The query files must have the following format. 

```
q1_node1,q1_node2,...
answer1_node1,answer1_node2,...
q2_node1,q2_node2,...
answer2_node1,answer2_node2
```

#### Training the Models & Execution

Finally the models could be trained and evaluated with the following command. 

```
python3 train.py <DATASET_NAME> <MAX_VERTICES> <START_SNAPSHOT_ID> <END_SNAPSHOT_ID> <THRESHOLD> <HIDDEN_DIM_SIZE> <EPOCHS> <LEARNING_RATE> <REGULARIZATION>
```

For example:
```
python3 train.py football 200 1 4 0.4 64 100 0.001 0.00001
```


### Reference
----------------  
```
@inproceedings{CS-TGN,
author = {Hashemi, Farnoosh and Behrouz, Ali and Rezaei Hajidehi, Milad},
title = {CS-TGN: Community Search via Temporal Graph Neural Networks},
year = {2023},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3543873.3587654},
doi = {10.1145/3543873.3587654},
booktitle = {Companion Proceedings of the Web Conference 2023},
numpages = {8},
location = {AUSTIN, TEXAS, USA},
series = {WWW '23}
}
```
