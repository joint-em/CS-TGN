
import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def list_to_one_hot_matrix(query_nodes, max_verts):
    query = [[0 for i in range(max_verts)] for i in range(max_verts)]
    for index in query_nodes:
        query[index][index] = 1
    return torch.Tensor(query)

class Snapshot:
    """
        Contains the state of the graph a
    """
    
    def __init__(self, max_verts, snapshot_dir, torchify=True):

        
        with open(snapshot_dir) as F:
            edges = F.readlines()
            nodes = set()
            edge_count = len(edges)
            self.edge_list = [[0 for i in range(edge_count * 2)] for i in range(2)]
            index = 0 
            for line in edges:
                line = line.strip('\n').split(' ')
                x = int(line[0])
                y = int(line[1])
                self.edge_list[0][index] = x
                self.edge_list[1][index] = y
                index += 1
                self.edge_list[0][index] = y
                self.edge_list[1][index] = x
                index += 1
                nodes.add(x)
                nodes.add(y)
        
        if not torchify:
            return
        nodes = list(nodes)
        self.max_verts = max_verts
        self.node_count = len(nodes)
        self.edge_index = torch.Tensor(self.edge_list).long()
        self.node_features = list_to_one_hot_matrix(nodes, max_verts=max_verts)


def list_to_index_vector(query_nodes, max_verts):
    query = [0 for i in range(max_verts)]
    for index in query_nodes:
        query[index] = 1
    return torch.Tensor(query)

class QueryFile:
    def __init__(self, max_verts, file_path, torchify=True):
        
        self.max_verts = max_verts
        
        self.path = file_path
        with open(file_path) as file:
            self.q_count = int(file.readline().strip('\n'))
            self.queries = []
            self.answers = []
            for i in range(self.q_count):
                query_nodes = list(map(int, file.readline().strip('\n').split(',')))
                answer_nodes = list(map(int, file.readline().strip('\n').split(',')))
                
                
                # print(query, answer)
                if torchify:
                    self.queries.append(list_to_one_hot_matrix(query_nodes, max_verts))
                    self.answers.append(list_to_index_vector(answer_nodes, max_verts))
            

    
    def __str__(self) -> str:
        return "Query object from {} with {} queries".format(self.path, len(self.queries))

