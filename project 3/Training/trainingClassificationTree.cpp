#include <iostream>
#include <fstream>
#include <armadillo>
#include <cmath>
#include <cstdlib>
#include <string>
#include "optTree.h"
#include "buildTree.h"

using namespace std;
using namespace arma;

void testFullTree(mat& patches, vec& label, int classNum, int patchWidth, int depth)
{
	fstream file;

	int treeSize = (pow(3, (depth + 1)) - 1) / 2.0;
	int count = (pow(3, (depth)) - 1) / 2.0;

	Node * tree;
	fullTreebuilder(patches, label, classNum, patchWidth, depth, &tree);

	// calculate posterior probability
	int test = 0;
	file.open("Posterior.txt", ios::out);
	for (int k = count; k < treeSize; k++)
	{
		cout << "k: " << k << endl;
		Dataset subset = tree[k].subset;
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
		if (max >= 0.5)
			test++;
		file << "max: " << max << endl;
	}
	file << test << endl;


	double errorRate = 0;
	int I = patches.n_cols;
	for (int i = 0; i < I; i++)
	{
		vec data = patches.col(i);
		int k = 0;
		while (k < count)
		{
			Point dpt_1 = tree[k].pt_dm1;
			Point dpt_2 = tree[k].pt_dm2;

			dpt_1.x = dpt_1.x + (patchWidth + 1) / 2;
			dpt_1.y = dpt_1.y + (patchWidth + 1) / 2;
			dpt_2.x = dpt_2.x + (patchWidth + 1) / 2;
			dpt_2.y = dpt_2.y + (patchWidth + 1) / 2;

			int diff = data(dpt_1.x*patchWidth + dpt_1.y) - data(dpt_2.x*patchWidth + dpt_2.y);

			if (diff < -10)
			{
				k = 3 * k + 1;
			}
			else if (abs(diff) <= 10)
			{
				k = 3 * k + 2;
			}
			else
			{
				k = 3 * k + 3;
			}
		}

		double I = tree[k].subset.label->n_rows;
		double max = 0;
		int maxLabel = 0;
		for (int c = 1; c <= classNum; c++)
		{
			double num = 0;
			for (int l = 0; l < tree[k].subset.label->n_rows; l++)
			{
				if ((*(tree[k].subset.label))(l) == c)
				{
					num = num + 1;
				}
			}
			double lambda = num / I;

			if (lambda > max)
			{
				max = lambda;
				maxLabel = c;
			}
		}

		if (maxLabel != label(i))
		{
			errorRate++;
		}
	}
	file << "Error rate: " << errorRate / I << endl;
	file.close();
}

void testTree(mat& patches, vec& label, int classNum, int patchWidth, int depth)
{
	int treeNumber = 5;

	TreeNode * roots = new TreeNode[treeNumber];

	int I = patches.n_cols;

	int batchSize = I / treeNumber;
	int remainBatchSize = I % treeNumber;

	for (int t = 0; t < treeNumber - 1; t++)
	{
		mat tempPatch = patches.cols(t*batchSize, (t + 1)*batchSize - 1);
		vec tempLabel = label.rows(t*batchSize, (t + 1)*batchSize - 1);
		treebuilder(tempPatch, tempLabel, classNum, patchWidth, depth, 0, &roots[t]);
	}
	mat tempPatch = patches.cols((treeNumber - 1)*batchSize, treeNumber*batchSize + remainBatchSize - 1);
	vec tempLabel = label.rows((treeNumber - 1)*batchSize, treeNumber*batchSize + remainBatchSize - 1);
	treebuilder(tempPatch, tempLabel, classNum, patchWidth, depth, 0, &roots[treeNumber - 1]);

	fstream file;
	file.open("error.txt", ios::out);
	double errorRate = 0;
	for (int i = 0; i < I; i++)
	{
		vec data = patches.col(i);
		Pair * p = new Pair[treeNumber];
		vec maxLabel = zeros<vec>(classNum);
		for (int t = 0; t < treeNumber; t++)
		{
			p[t] = testTree(data, classNum, patchWidth, &roots[t]);
			maxLabel(p[t].label)++;
		}

		if ( (maxLabel.index_max() + 1) != label(i))
		{
			errorRate++;
		}
		delete[]p;
	}
	cout << "Error rate: " << errorRate / I << endl;
	file << "Error rate: " << errorRate / I << endl;
	file.close();

	/*
	fstream file;
	TreeNode root;
	treebuilder(patches, label, classNum, patchWidth, depth, 0, &root);
	
	file.open("Posterior.txt", ios::out);
	travelTree(file, classNum, patchWidth, depth, 0, &root);
	file.close();
	*/
}

int main()
{
	srand(time(NULL));

	string imagePatchFilename = "imagePatches.mat";
	string patchLabelFilename = "patchLabel.mat";

  
	mat patches;
	patches.load(imagePatchFilename, hdf5_binary);

	vec label;
	label.load(patchLabelFilename, hdf5_binary);

	int classNum =  200;
	int patchWidth = sqrt(patches.n_rows);

	int depth = 5;

	//testFullTree(patches, label, classNum, patchWidth, depth);
	testTree(patches, label, classNum, patchWidth, depth);
  
	return 0;
}