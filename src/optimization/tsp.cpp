#include<bits/stdc++.h>
using namespace std;
using namespace std::chrono;

double cost (vector <vector <double>> &m, vector <int> &perm) {
    double ans = 0;
    int prev = perm[0];
    for (int i = 0; i < perm.size()-1; i++) {
        ans += m[prev][perm[i+1]];
        prev = perm[i+1];
    }
    return ans;
}
int main() {
    
    mt19937 rng(random_device{}());
    uniform_real_distribution <double> edge(10, 10000);
    int n = 100;
    vector <vector <double>> mat(n, vector<double>(n));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j) continue;
            mat[i][j] = edge(rng);
        }
    }
    vector <int> perm(n);
    iota(perm.begin(), perm.end(), 0);
    double MAX = 10;
    double start_temp = 1e5;
    double end_temp = 0.001;
    double ratio = end_temp/start_temp;
    const double a = 0.9999;
    auto start = high_resolution_clock::now();
    uniform_real_distribution <double> proba(0, 1);
    double best_score = cost(mat, perm);
    uniform_int_distribution <int> swapp(0, n-1);
    while (true) {
        auto cur = high_resolution_clock::now();
        double elapsed = duration <double> (cur-start).count();
        if (elapsed > MAX) {
            break;
        }
        double progress = elapsed / MAX;
        double temp = t*a;
        int i = swapp(rng);
        int j = swapp(rng);
        while (i == j) {
            j = swapp(rng);
        }
        if (i > j) swap(i,j);
        reverse(perm.begin() + i, perm.begin()+j);
        double new_score = cost(mat, perm);
        if (new_score < best_score || proba(rng) < exp((best_score - new_score)/temp)) {
            best_score = new_score;
        } else {
            reverse(perm.begin() + i, perm.begin()+j);
        }

    }
    cout<<best_score;
}