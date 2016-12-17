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
    uvec dataIndices;
    vec label;
};

struct Node{
    Point pt_dm1;
    Point pt_dm2;
    Dataset subset_left;
    Dataset subset_center;
    Dataset subset_right;
};

void optTree(mat& patches, vec& label, int classNum, int patchWidth, Node* resultNode);