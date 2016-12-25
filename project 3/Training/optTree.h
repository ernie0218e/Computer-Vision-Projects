#ifndef USE_ARMA
#define USE_ARMA
#include <armadillo>

using namespace std;
using namespace arma;

// store the info of point
class Point{
public:
    int x;
    int y;
};

// store image data and their label
// TO-DO - store image data costs too much memory
//		   we only need to store the index
class Dataset{
public:
    mat* data;
    vec* label;
};

// Store info of decision rule and subset of data
class Node{
public:
    Point pt_dm1;
    Point pt_dm2;
    Dataset subset;
};

// real node of tree
// store info in 'Node'
// connect to child nodes with 'TreeNode' pointers
class TreeNode {
public:
	Node * node;
	TreeNode ** childNodes;
};


// find the best way to divide date into three parts
// and the decision rule of given node
// parameters: mat& patches - given training image
//			   vec& label - training image label
//			   int classNum - number of classes
//		       int patchWidth - width of patch
//			   Node* resultNode - store the decision rule (pt_dm1, pt_dm2)
//			   Dataset* subsets - three subset of data (which are from 'patches')
//			   int currentDepth - the number of iterations for optimaization based on this value
void optTree(mat& patches, vec& label, int classNum, int patchWidth, Node* resultNode, Dataset* subsets, int currentDepth=0);

// Node* resultNode and Dataset* subsets may be redundant

#endif