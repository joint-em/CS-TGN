
NODE_COUNT = 190

ROOT_DIR = 'data/ADHD/graphs/td/'

EDGE_INCIDENCE_THRESHOLD = 0.5

SCAN_FILES = [
    'KKI_1988015.txt',
    'NYU_1000804.txt',
    'Peking_2_1068505.txt',
    'KKI_1996183.txt',
    'NYU_1023964.txt',
    'Peking_2_1093743.txt',
    'KKI_2014113.txt',
    'NYU_1127915.txt',
    'Peking_2_1094669.txt',
    'KKI_2018106.txt',
    'NYU_1208795.txt',
    'Peking_2_1159908.txt',
    'KKI_2026113.txt',
    'NYU_1283494.txt',
    'Peking_2_1341865.txt',
    'KKI_2138826.txt',
    'NYU_1435954.txt',
    'Peking_2_1494102.txt',
    'KKI_2299519.txt',
    'NYU_1471736.txt',
    'Peking_2_1562298.txt',
    'KKI_2344857.txt',
    'NYU_1567356.txt',
    'Peking_2_1628610.txt',
    'KKI_2360428.txt',
    'NYU_1700637.txt',
    'Peking_2_1643780.txt',
    'KKI_2371032.txt',
    'NYU_1740607.txt',
    'Peking_2_1809715.txt',
    'KKI_2554127.txt',
    'NYU_1875084.txt',
    'Peking_2_1860323.txt',
    'KKI_2558999.txt',
    'NYU_1884448.txt',
    'Peking_2_1916266.txt',
    'KKI_2572285.txt',
    'NYU_1992284.txt',
    'Peking_2_2031422.txt',
    'KKI_2601925.txt',
    'NYU_1995121.txt',
    'Peking_2_2033178.txt',
    'KKI_2618929.txt',
    'NYU_2030383.txt',
    'KKI_2621228.txt',
    'Peking_2_2140063.txt',
    'Pittsburgh_0016058.txt',
    'Pittsburgh_0016059.txt',
    'Pittsburgh_0016060.txt',
    'Pittsburgh_0016061.txt',
]


if __name__ == '__main__':
    tot = 0
    s = set()
    agg_adj_mat = [[0. for i in range(NODE_COUNT)] for j in range(NODE_COUNT)]
    for file_path in SCAN_FILES:


        with open(ROOT_DIR + file_path) as F:
            lines = F.readlines()

            assert (len(lines) == NODE_COUNT)
            for i, line in enumerate(lines):
                adj_list = list(map(int, line.strip('\n').split(' ')))
                for j, v in enumerate(adj_list):
                    agg_adj_mat[i][j] += v
                    if agg_adj_mat[i][j] > 25:
                        s.add((i,j))

    for i,j in s:
        print(i, j)