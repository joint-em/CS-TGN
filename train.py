
import random
import time
import sys
import copy
import torch
from torch_geometric.data import Data
import torch.nn.functional as F

from dataset import QueryFile, Snapshot

from models.cstgn import CSTGN
from f1score import compute_f1


EPOCH_COUNT = int(sys.argv[7])
MAX_VERTS = int(sys.argv[2])
LR = float(sys.argv[8])
L2 = float(sys.argv[9])
THRESHOLD = float(sys.argv[5])
HIDDEN_DIM = int(sys.argv[6])
START_SNAPSHOT = int(sys.argv[3])
END_SNAPSHOT = int(sys.argv[4])


DATASET = sys.argv[1]

def train_query(model, optimizer, q_index):
    optimizer.zero_grad()    
    hq = None
    hg = None
    hg_past = None
    hg_pastpast = None
    query_loss = 0
    
    for snapshot_id in range(0, END_SNAPSHOT - START_SNAPSHOT + 1):
        queries = temporal_queries[snapshot_id]
        data = temporal_snapshots[snapshot_id]
        query = queries.queries[q_index]
        answers = queries.answers[q_index]
        
        query = query.to(device)
        answers = answers.to(device)
        
        nex_hg_pastpast = hg_past        
        nex_hg_past = hg
        
        out, hg, hq = model(data, query, hg, hq, hg_past, hg_pastpast)
        
        hg_past = nex_hg_past
        hg_pastpast = nex_hg_pastpast
        
        loss = criterion(out, answers)
        query_loss += float(loss)
        loss.backward(retain_graph=True)
    
    optimizer.step()
    return query_loss

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = CSTGN(num_features=MAX_VERTS, hidden_dim=HIDDEN_DIM).to(device)

    print('Start reading and parsing the data (graph, queries)...')


    temporal_queries = [] 
    temporal_snapshots = []
    last_snapshot = None
    for i in range(START_SNAPSHOT, END_SNAPSHOT + 1):
        temporal_queries.append(QueryFile(MAX_VERTS, 'data/{}/queries_{}.txt'.format(DATASET,i)))
        graph = Snapshot(MAX_VERTS, 'data/{}/graph_{}.txt'.format(DATASET, i))
        data = Data(edge_index=graph.edge_index, x=graph.node_features).to(device=device)
        print("Snapshot {} with {} edge and {} nodes".format(i, len(graph.edge_list[0]), graph.node_count))
        data.validate(raise_on_error=True)
        last_snapshot = data
        temporal_snapshots.append(data)
    valid_queries = QueryFile(MAX_VERTS, 'data/{}/valid_queries.txt'.format(DATASET))

    print(len(temporal_queries), temporal_queries[0].q_count)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=L2)
    criterion = torch.nn.BCELoss()

    print('Done reading data, start training...')
    best_model = copy.deepcopy(model)
    best_val_acc = 0
    f1_list = []
    loss_list = []

    for epoch in range(EPOCH_COUNT):
        model.train()    
        epoch_loss = 0.0
        t0 = time.time()

        indices = [i for i in range(temporal_queries[-1].q_count)]
        random.shuffle(indices)
        
        for q_index in indices:
            epoch_loss += train_query(model, optimizer, q_index)

        print("Epoch {} loss: {}".format(epoch, epoch_loss/temporal_queries[-1].q_count),end='')
        print(" time: {}".format(time.time() - t0),end='')
        loss_list.append(epoch_loss/temporal_queries[-1].q_count)
        # Compute validation error
        model.eval()
        cur_f1_valid = compute_f1(valid_queries, last_snapshot, model, THRESHOLD)
        f1_list.append(cur_f1_valid)
        print(" valid f1: {}".format(cur_f1_valid),flush=True)
        if cur_f1_valid > best_val_acc:
            best_val_acc = cur_f1_valid
            best_model = copy.deepcopy(model)

    print(loss_list)
    print(f1_list)


    print("***** TEST *****")
    best_model.eval()
    test_queries = QueryFile(MAX_VERTS, 'data/{}/test_queries.txt'.format(DATASET))
    print("TEST F1: {}".format(compute_f1(test_queries, last_snapshot, best_model, THRESHOLD)))
    model.eval()
    test_queries = QueryFile(MAX_VERTS, 'data/{}/test_queries.txt'.format(DATASET))
    print("TEST F1: {}".format(compute_f1(test_queries, last_snapshot, best_model, THRESHOLD)))
