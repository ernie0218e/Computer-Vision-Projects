#ifndef USE_ARMA
#define USE_ARMA
#include <armadillo>
#endif

using namespace std;
using namespace arma;

struct Point{
    int x;
    int y;
};

struct Dataset{
    mat* data;
    vec* label;
};

struct Node{
    Point pt_dm1;
    Point pt_dm2;
    Dataset subset;
};

void optTree(mat& patches, vec& label, int classNum, int patchWidth, Node* resultNode, Dataset* subsets);