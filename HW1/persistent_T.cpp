/* 
  Foundations of Parallel Computing II, Spring 2023.
  Instructor: Prof. Chao Yang @ Peking University.
  This is a serial implement of Bellman-Ford algorithm.
*/
#include <string>
#include <cassert>
#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>
#include <sys/time.h>
#include <omp.h>
#include <string.h>

using std::string;
using std::cout;
using std::endl;
using namespace std;
#define INF 1000000

/*
    矩阵大小N和邻接矩阵mat
    read_file: 从文件中输入邻接矩阵
    print_result: 把距离矩阵输出到文件中
*/
namespace utils
{
    int N; //number of vertices
    vector<vector<int> > mat; // the adjacency matrix
    vector<vector<int> > matT; //Transpose matrix
    void abort_with_error_message(string msg) {
        std::cerr << msg << endl;
        abort();
    }

    /*
    The input file will be in following format:
    - The first line is an integer N, the number of vertices in the graph.
    - The following lines are an N*N adjacency matrix mat. The entry mat[v][w] is the distance (weight) from vertex v to vertex w. All distances are integers.
    - If there is no edge joining vertex v and w, mat[v][w] will be 1000000 to represent infinity.
    - The vertices will be labeled by 0, 1, 2, …, N-1. We use vertex 0 as the source vertex.

    The output file will be in following format:
    - It consists the distances from vertex 0 to all vertices, in the increasing order of the vertex label (vertex 0, 1, 2, … and so on), one distance per line. 
    - If there are at least one negative cycle (the sum of the weights of the cycle is negative in the graph), the program will set variable has_negative_cycle to true and print "FOUND NEGATIVE CYCLE!" as there will be no shortest path.
    
    Run:
    $ ./serial <input file>
    */
    int read_file(string filename) {
        std::ifstream inputf(filename, std::ifstream::in);
        if (!inputf.good()) {
            abort_with_error_message("ERROR OCCURRED WHILE READING INPUT FILE");
        }
        inputf >> N;
        mat.resize(N);
        matT.resize(N);
        for (int i = 0; i < N; i++)
        {
            mat[i].resize(N);
            matT[i].resize(N);
        }
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
            {
                inputf >> mat[i][j];
                matT[j][i] = mat[i][j];
            }
        return 0;
    }

    int print_result(bool has_negative_cycle, int *dist) {
        std::ofstream outputf("output.txt", std::ofstream::out);
        if (!has_negative_cycle)
        {
            for (int i = 0; i < N; i++) {
                if (dist[i] > INF)
                    dist[i] = INF;
                outputf << dist[i] << '\n';
            }
            outputf.flush();
        } else {
            outputf << "FOUND NEGATIVE CYCLE!" << endl;
        }
        outputf.close();
        return 0;
    }
} //namespace utils


/*
    单源最短路径，以0为起点，通过邻接矩阵更新距离向量
    输入：邻接矩阵
    输出：距离向量
    
    算法：
        初始化：d[0]=0
        状态转移：d[x]=min(d[x],d[y]+w[y][x])
            迭代可以是同步的或异步的，但差距不可以超过一步。这给了我们并行的空间
        终止条件：在某次迭代后没有发生更新，或迭代了n-1轮（此时需要多迭代一轮，判定负权）。
*/

/**
 * Bellman-Ford algorithm. `has_shortest_path` will be set to false if negative cycle found
 */
void bellman_ford(int n, vector<vector<int> >&mat, vector<vector<int> > &matT, int *dist, bool *has_negative_cycle)
{
    //a flag to record if there is any distance change in this iteration
    bool has_change = 1;
    // int tid, nt, local_v;
    

    //bellman-ford edge relaxation
#pragma omp parallel shared(has_change)
{
    int nt = omp_get_num_threads();
    int tid = omp_get_thread_num();
    int chunksize = (n + nt - 1) / nt, begin = tid * chunksize, end = begin + chunksize;
    if (n < end)
    {
        end = n;
        chunksize = end - begin;
    }
    for (int i = begin; i < end; i++)
        dist[i] = INF;
    dist[0] = 0;


    // int* temp_matTv = (int*)malloc(sizeof(int) * n);
    // int* temp_dist = (int*) malloc(sizeof(int) * n);
    for (int i = 0; i < n - 1; i++)     // n - 1 iteration
    {
#pragma omp single
        has_change = false;

        // memcpy(temp_dist, dist, sizeof(int) * n);
        register bool local_has_change = false;
        for (int v = tid; v < n; v += nt)
        {
            // memcpy(temp_matTv, &(matT[v][0]), sizeof(int) * n);
            register int local_v = dist[v];
            for (int u = 0; u < n; u ++)
            {
                register int local_u = dist[u];
                register int weight = matT[v][u];
                if (weight < INF)       //test if u--v has an edge
                {
                    if (local_u + weight < local_v)
                    {
                        // has_change = true;
                        local_has_change = true;
                        local_v= local_u + weight;
                    }
                }
            }
            if (dist[v] != local_v)
                dist[v] = local_v;
        }
        // memcpy(dist + begin, temp_dist + begin, chunksize*sizeof(int));
        if (local_has_change)
            has_change = true;
        //if there is no change in this iteration, then we have finished
#pragma omp barrier
        if (!has_change)
            break;
#pragma omp barrier
    }
    // free(temp_matTv);
    // free(temp_dist);
}

    if (!has_change)
        return;
//do one more iteration to check negative cycles
#pragma omp parallel for
    for (int u = 0; u < n; u++)
    {
        if (*has_negative_cycle)
            continue;
        for (int v = 0; v < n; v++)
        {
            int weight = utils::mat[u][v];
            if (weight < INF)
                if (dist[u] + weight < dist[v])     // if we can relax one more step, then we find a negative cycle
                {   
                    *has_negative_cycle = true;
                    break;
                }
        }
    }

}

int main(int argc, char **argv) {
    if (argc <= 1)
        utils::abort_with_error_message("INPUT FILE WAS NOT FOUND!");
    string filename = argv[1];
    assert(utils::read_file(filename) == 0);

    //initialize results
    int *dist;
    dist = (int *) malloc(sizeof(int) * utils::N);

    bool has_negative_cycle = false;


    //time counter
    timeval start_wall_time_t, end_wall_time_t;
    float ms_wall;
    gettimeofday(&start_wall_time_t, nullptr);      //start timer

    bellman_ford(utils::N, utils::mat, utils::matT, dist, &has_negative_cycle);      //bellman ford algorithm

    gettimeofday(&end_wall_time_t, nullptr);        //end timer
    ms_wall = ((end_wall_time_t.tv_sec - start_wall_time_t.tv_sec) * 1000 * 1000
               + end_wall_time_t.tv_usec - start_wall_time_t.tv_usec) / 1000.0;

    std::cerr << "Time(s): " << ms_wall/1000.0 << endl;
    utils::print_result(has_negative_cycle, dist);

    free(dist);
    vector<vector<int> >().swap(utils::mat);
    return 0;
}