#ifndef USE_ARMA
#define USE_ARMA
#include <armadillo>

using namespace std;
using namespace arma;

class Point{
public:
    int x;
    int y;
};

class Dataset{
public:
    mat* data;
    vec* label;
};

class Node{
public:
    Point pt_dm1;
    Point pt_dm2;
    Dataset subset;
};

class TreeNode {
public:
	Node * node;
	TreeNode ** childNodes;
};

struct Pair {
	int label;
	vec * lambda;
};

void optTree(mat& patches, vec& label, int classNum, int patchWidth, Node* resultNode, Dataset* subsets, int currentDepth=0);

#endif