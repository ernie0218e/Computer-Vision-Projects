#include "buildTree.h"
#include <cmath>
#include <cstdlib>
#include <armadillo>

using namespace std;
using namespace arma;

void fullTreebuilder(mat& patches, vec& label, int classNum, int patchWidth, int depth, Node ** finalTree)
{
	int treeSize = (pow(3, (depth + 1)) - 1) / 2.0;

	cout << "Tree size: " << treeSize << endl;

	Node * tree = new Node[treeSize];

	Dataset * subsets = new Dataset[3];
	for (int i = 0; i < 3; i++)
	{
		subsets[i].data = nullptr;
		subsets[i].label = nullptr;
	}

	optTree(patches, label, classNum, patchWidth, &tree[0], subsets);

	for (int k = 1; k <= 3; k++)
	{
		tree[k].subset = subsets[k - 1];
	}

	int count = (pow(3, (depth)) - 1) / 2.0;

	cout << "count: " << count << endl;

	for (int k = 1; k < count; k++)
	{
		cout << "k: " << k << endl;
		cout << "col_s: " << (*(tree[k].subset.data)).n_cols << endl;
		Dataset * subsets = new Dataset[3];
		for (int i = 0; i < 3; i++)
		{
			subsets[i].data = nullptr;
			subsets[i].label = nullptr;
		}
		optTree(*(tree[k].subset.data), *(tree[k].subset.label), classNum, patchWidth, &tree[k], subsets);

		delete tree[k].subset.data;
		delete tree[k].subset.label;

		for (int j = 1; j <= 3; j++)
		{
			tree[3 * k + j].subset = subsets[j - 1];
		}
	}

	finalTree = &tree;
}

int treebuilder(mat& patches, vec& label, int classNum, int patchWidth, int depth, int currentDepth, TreeNode * finalTree)
{
	int I = patches.n_cols;
	if (currentDepth >= depth || I < 20)
	{
		return 1;
	}
	else
	{
		cout << "currentDepth: " << currentDepth << endl;

		Dataset * subsets = new Dataset[3];
		for (int i = 0; i < 3; i++)
		{
			subsets[i].data = nullptr;
			subsets[i].label = nullptr;
		}

		if (finalTree == nullptr)
			cout << "ERROR" << endl;

		finalTree->node = new Node;
		
		optTree(patches, label, classNum, patchWidth, finalTree->node, subsets);

		finalTree->childNodes = new TreeNode * [3];

		for (int k = 0; k < 3; k++)
		{
			finalTree->childNodes[k] = new TreeNode;
			int val = treebuilder(*(subsets[k].data), *(subsets[k].label), classNum, patchWidth, depth,
				currentDepth + 1, finalTree->childNodes[k]);
			
			if (val)
			{
				finalTree->childNodes[k] = nullptr;
			}

			delete subsets[k].data;
			delete subsets[k].label;
		}

		delete []subsets;
		return 0;
	}
}