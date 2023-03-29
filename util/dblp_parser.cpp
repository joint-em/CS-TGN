#include <bits/stdc++.h>

#define pb push_back

using namespace std;

const int max_n = 5 * 100000 + 10;
vector<int> adj[max_n];
unordered_map<string, int> name_to_id;
unordered_map<int, string> id_to_name;
int total_nodes, total_edges;

int main(int argc, char *argv[])
{
    string name_path = argv[1];
    string graph_path = argv[2];
    int thresh = atoi(argv[3]);

    cout << "READING FROM: " << name_path << " " << graph_path << endl;

    ifstream gin(graph_path);
    ifstream namin(name_path);
    ofstream selected_nodes_out("selected_nodes.txt");
    ofstream candidate_nodes_out("candidate_nodes.txt");

    if (gin.fail() || namin.fail())
    {
        throw runtime_error("File path are not correct");
    }

    int node_id;
    char delim;
    string person_name;
    cout << "ENDL" << endl;
    while (namin >> node_id)
    {
        namin.ignore();
        getline(namin, person_name);
        name_to_id[person_name] = node_id;
        id_to_name[node_id] = person_name;
        total_nodes++;
    }
    int x, y;
    while (gin >> x >> delim >> y)
    {
        total_edges++;
        adj[x].pb(y);
        adj[y].pb(x);
    }

    cout << total_nodes << " " << total_edges << endl;
    unordered_set<int> selected_nodes;
    unordered_set<int> candidate_nodes;
    while (candidate_nodes.size() < thresh)
    {
        int rand_id = rand() % total_nodes;
        if (adj[rand_id].size() > 10){
            selected_nodes.insert(rand_id);
            candidate_nodes.insert(rand_id);

            for (auto nei: adj[rand_id]){
                candidate_nodes.insert(nei);
                for (auto neinei: adj[nei]){
                    candidate_nodes.insert(neinei);
                }
            }
        }
    }

    cout << candidate_nodes.size() << " " << selected_nodes.size() << endl;

    for (auto sel_it:selected_nodes){
        selected_nodes_out << sel_it << "," << id_to_name[sel_it] << endl;
    }

    for (auto can_it:candidate_nodes){
        candidate_nodes_out << can_it << "," << id_to_name[can_it] << endl;
    }
}