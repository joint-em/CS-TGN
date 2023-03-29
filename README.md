# CS-TGN
This repository contains the implementation of algorithms, and used datasets of paper ``CS-TGN: Community Search via Temporal Graph Neural Networks`` (ACM The Web Conference Companion 2023  (WWW)). 


#### Authors: [Farnoosh Hashemi](https://farnooshha.github.io//), [Ali Behrouz](https://abehrouz.github.io/), [Milad Rezaei Hajidehi](https://www.linkedin.com/in/milad-rezaei-hajidehi-63013773/)
#### [Link to the paper](https://doi.org/10.1145/3543873.3587654) ([Arxiv](https://arxiv.org/abs/2303.08964))
#### [Poster]()
#### [Brief video explanation]()



### Abstract
----------------  
Searching for local communities is an important research challenge that allows for personalized community discovery and supports advanced data analysis in various complex networks, such as the World Wide Web, social networks, and brain networks. The evolution of these networks over time has motivated several recent studies to identify local communities in temporal networks. Given any query nodes, Community Search aims to find a densely connected subgraph containing query nodes. However, existing community search approaches in temporal networks have two main limitations: (1) they adopt pre-defined subgraph patterns to model communities, which cannot find communities that do not conform to these patterns in real-world networks, and (2) they only use the aggregation of disjoint structural information to measure quality, missing the dynamic of connections and temporal properties. In this paper, we propose a query-driven Temporal Graph Convolutional Network (CS-TGN) that can capture flexible community structures by learning from the ground-truth communities in a data-driven manner. CS-TGN first combines the local query-dependent structure and the global graph embedding in each snapshot of the network and then uses a GRU cell with contextual attention to learn the dynamics of interactions and update node embeddings over time. We demonstrate how this model can be used for interactive community search in an online setting, allowing users to evaluate the found communities and provide feedback. Experiments on real-world temporal graphs with ground-truth communities validate the superior quality of the solutions obtained and the efficiency of our model in both temporal and interactive static settings.


  
    

### Dependecies & Hardware
----------------
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

  
  
### Components
----------------
The core component is the models impelemented in `models/` directory. You can find the vanilla CSTGN and CSTGN without GRU as well as attention module. Modelwise, you may enhance the architecture for particular datasets by changing the attention snapshots, number of GCN layers and changing the hidden dimension of models. 

The `train.py` can be used to trained the RNN network of CSTGN. Please read the Dataset Structure for instruction for using it. 

The `dataset.py` has two classes `QueryFile` and `Snapshot` for representing the query/answer pairs and temporal view of graph at a specific time. 

The `util/` directory contains some codes to parse specific datasets and generate queries. 
  
  
### Dataset Structure
----------------
The graphs and query/answer sets can be in any format but in order to use our train script and dataset utilities these should be in following format. 
  
  
### File Structure
----------------
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

  
  
### Graph file and QueryFiles
----------------
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
  
    
### Training the Models & Execution
----------------
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
