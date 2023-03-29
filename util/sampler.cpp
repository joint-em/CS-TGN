#include <bits/stdc++.h>

using namespace std;

#define pb push_back

const int maxN = 1500000 + 10;
vector<int> v[maxN];
bitset<maxN> b;
int last_used_ind = 1;
map<int, int> mp;

inline void register_id(int x)
{
    if (b[x] == false)
    {
        b[x] = true;
        mp[x] = last_used_ind++;
    }
    return;
}

int main(int argc, char *argv[])
{
    string graph_path = argv[1];
    string com_path = argv[2];
    int thresh = atoi(argv[3]);

    cout << graph_path << " " << com_path << " " << thresh << endl;

    ifstream gin(graph_path);
    ifstream comin(com_path);
    ofstream gout("g.txt");
    ofstream comout("com.txt");
    if (gin.fail()){
        cout << "FATAL";
    }
    int x, y;
    int total_edge = 0;
    while (gin >> x >> y)
    {
        v[x].pb(y);
        // v[y].pb(x);
        total_edge++;
    }
    cout << total_edge << endl;

    string input_com;
    vector<int> sampled_nodes;
    vector < vector < int > > all_comp;
    int total_comm = 0;
    while (getline(comin, input_com) && total_comm < thresh)
    {
        istringstream iss(input_com);
        vector<int> cur_comp;
        int temp;
        while (iss >> temp)
        {
            cur_comp.push_back(temp);
        }

        if (cur_comp.size() > 5)
        {
            for (auto node : cur_comp)
            {
                sampled_nodes.pb(node);
            }
        }
        all_comp.pb(cur_comp);
        total_comm++;
    }
    cout << sampled_nodes.size() << endl;
    for (int node : sampled_nodes)
    {
        register_id(node);
        cout << node << " " << v[node].size() << endl;
        for (auto nei : v[node])
        {
            register_id(nei);
            for (auto neinei : v[nei])
            {
                register_id(neinei);
            }
        }
    }

    for (int i = 0; i < maxN; i++)
    {
        if (b[i] == false)
            continue;
        for (auto nei : v[i])
        {
            if (b[nei])
            {
                gout << mp[i] << " " << mp[nei] << endl;
            }
        }
    }
    for (auto comp:all_comp){
        for (auto old_id:comp){
            comout << mp[old_id] << ",";
        }
        comout << endl;
    }
    cout << last_used_ind << endl;
}