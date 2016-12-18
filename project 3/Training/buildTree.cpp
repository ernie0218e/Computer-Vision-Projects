#include "buildTree.h"
#include <cmath>
#include <cstdlib>
#include <fstream>
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

		int flag = 0;
		for (int k = 0; k < 3; k++)
		{
			if (finalTree->childNodes[k] != nullptr)
			{
				flag = 1;
				break;
			}
		}
		
		if (!flag)
		{
			finalTree->node->subset.data = new mat(patches);
			finalTree->node->subset.label = new vec(label);
		}


		delete []subsets;
		return 0;
	}
}

int travelTree(fstream& file, int classNum, int patchWidth, int depth, int currentDepth, TreeNode * finalTree)
{
	if (finalTree == nullptr)
	{
		return 1;
	}
	else
	{

		for (int k = 0; k < 3; k++)
		{
			int val = travelTree(file, classNum, patchWidth, depth, currentDepth + 1, finalTree->childNodes[k]);
		}

		int flag = 0;
		for (int k = 0; k < 3; k++)
		{
			if (finalTree->childNodes[k] != nullptr)
			{
				flag = 1;
				break;
			}
		}

		if (!flag)
		{
			// calculate posterior probability
			int test = 0;
			Dataset subset = finalTree->node->subset;
			vec sublabel = *(subset.label);

			double I = sublabel.n_rows;
			double max = 0;
			for (int c = 1; c <= classNum; c++)
			{
				double num = 0;
				for (int l = 0; l < sublabel.n_rows; l++)
				{
					if ((sublabel)(l) == c)
					{
						num = num + 1;
					}
				}
				double lambda = num / I;

				if (lambda > max)
					max = lambda;
			}
			cout << max << endl;
			file << "max: " << max << endl;
		}

		return 0;
	}
}

Pair testTree(vec& data, int classNum, int patchWidth, TreeNode * finalTree)
{
	if (finalTree == nullptr)
	{
		Pair returnedP = { -1, 0.0 };
		return returnedP;
	}
	else
	{

		Point dpt_1 = finalTree->node->pt_dm1;
		Point dpt_2 = finalTree->node->pt_dm2;

		dpt_1.x = dpt_1.x + (patchWidth + 1) / 2;
		dpt_1.y = dpt_1.y + (patchWidth + 1) / 2;
		dpt_2.x = dpt_2.x + (patchWidth + 1) / 2;
		dpt_2.y = dpt_2.y + (patchWidth + 1) / 2;

		int diff = data(dpt_1.x*patchWidth + dpt_1.y) - data(dpt_2.x*patchWidth + dpt_2.y);

		Pair returnedP = { -1, 0.0 };

		if (diff < -10)
		{
			returnedP = testTree(data, classNum, patchWidth, finalTree->childNodes[0]);
		}
		else if (abs(diff) <= 10)
		{
			returnedP = testTree(data, classNum, patchWidth, finalTree->childNodes[1]);
		}
		else
		{
			returnedP = testTree(data, classNum, patchWidth, finalTree->childNodes[2]);
		}

		if (returnedP.label == -1)
		{
			double I = finalTree->node->subset.label->n_rows;
			Pair maxP = { 0, 0.0 };
			for (int c = 1; c <= classNum; c++)
			{
				double num = 0;
				for (int l = 0; l < I; l++)
				{
					if ((*(finalTree->node->subset.label))(l) == c)
					{
						num = num + 1;
					}
				}
				double lambda = num / I;

				if (lambda > maxP.lambda)
				{
					maxP.lambda = lambda;
					maxP.label = c;
				}
			}
			returnedP = maxP;
		}
		return returnedP;
	}
}