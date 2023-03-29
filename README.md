# CS-TGN

This repository contains the code and the instruction for running and reproducing results for Community Search with Temporal Graph Learning.

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

#### Contact & More Info
For more information, please read our paper at: ANONYMOUS LINK or contact anom@anom.com . 
