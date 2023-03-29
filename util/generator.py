import random
import sys

from confs import GENERATOR_MAX_Q_SIZE, SMALL_COMM_THRESHOLD

def com_file_to_query_file(com_file, output_file, query_per_com):
    
    final_lines = []
    with open(com_file) as F:

        for com_line in F:
            nodes = list(com_line.strip('\n').split(','))
            com_size = len(nodes)
            if len(nodes) < SMALL_COMM_THRESHOLD:
                continue
            
            for i in range(query_per_com):
                q_node_count = random.randint(1, min(com_size, 7))
                q_nodes = random.sample(nodes, q_node_count)
                final_lines.append(",".join(q_nodes) + '\n' + com_line)
    random.shuffle(final_lines)
    with open(output_file, "x") as F:
        F.write(str(len(final_lines)) + '\n')
        for line in final_lines:
            F.write(line)
        F.close()

if __name__ == '__main__':
    com_file = sys.argv[1]
    output_file = sys.argv[2]
    query_per_com = int(sys.argv[3])
    com_file_to_query_file(com_file, output_file, query_per_com)
