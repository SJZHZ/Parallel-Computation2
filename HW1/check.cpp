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

int read_file(string filename, vector<int>& V)
{
    std::ifstream inputf(filename, std::ifstream::in);
    if (!inputf.good()) {
        return 1;
    }
    if (inputf.get() == 'F')
        return 1;
    if (inputf.get() == EOF)
        return 1;
    int temp;
    while(inputf >> temp)
        V.push_back(temp);
    return 0;
}



int main(int argc, char **argv) {
    // if (argc <= 1)
    //     return 0;
    // string filename = argv[1];
    vector<int> standard, test;
    assert(read_file("output_std.txt", standard) == 0);
    assert(read_file("output.txt", test) == 0);

    int flag = 0;
    for (auto i = standard.begin(), j = test.begin(); i != standard.end() && j != test.end(); i++, j++)
    {
        if (*i != *j)
        {
            flag = 1;
            break;
        }
        // cout << *i << " == " << *j << endl;
    }
    if (flag)
        cout << "FAILED" << endl;
    else
        cout << "PASSED" << endl;
    return 0;
}