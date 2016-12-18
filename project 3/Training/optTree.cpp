#include "optTree.h"
#include <armadillo>
#include <cstdlib>
#include <cmath>
#include <limits>

using namespace std;
using namespace arma;


void optTree(mat& patches, vec& label, int classNum, int patchWidth, Node* resultNode, Dataset* subsets)
{
	if (patches.n_cols > 0)
	{
		const int D = patches.n_rows;
		int I = patches.n_cols;

		const double threshold = 10;
		double max_entropy = -numeric_limits<double>::infinity();

		const int times = 1000;

		for (int t = 0; t < times; t++)
		{
			int i = rand() % (D - 1);
			int j = i + 1 + rand() % (D - i - 1);

			rowvec temp_pixels_m1 = patches.row(i);
			rowvec temp_pixels_m2 = patches.row(j);

			// compute pixels difference
			vec intensityDiff = (temp_pixels_m1 - temp_pixels_m2).t();

			// decide which subset the data should belong to
			uvec temp_subset_left_idx = find(intensityDiff < -threshold);
			uvec temp_subset_center_idx = find(abs(intensityDiff) <= threshold);
			uvec temp_subset_right_idx = find(intensityDiff > threshold);

			// get label of data in each subset
			vec temp_subset_left_label = label.elem(temp_subset_left_idx);
			vec temp_subset_center_label = label.elem(temp_subset_center_idx);
			vec temp_subset_right_label = label.elem(temp_subset_right_idx);

			// calculate entropy of subset in left child
			double denom = temp_subset_left_label.n_rows;
			double entropy = 0;
			for (int c = 1; c <= classNum; c++)
			{
				double num = 0;
				for (int l = 0; l < temp_subset_left_label.n_rows; l++)
				{
					if (temp_subset_left_label(l) == c)
					{
						num = num + 1;
					}
				}
				double p = num / denom;
				if (p != 0)
					entropy = entropy - p*log2(p);
			}
			double total_entropy = (double)temp_subset_left_label.n_rows / I * entropy;

			// calculate entropy of subset in center child
			denom = temp_subset_center_label.n_rows;
			entropy = 0;
			for (int c = 1; c <= classNum; c++)
			{
				double num = 0;
				for (int l = 0; l < temp_subset_center_label.n_rows; l++)
				{
					if (temp_subset_center_label(l) == c)
					{
						num = num + 1;
					}
				}
				double p = num / denom;
				if (p != 0)
					entropy = entropy - p*log2(p);
			}
			total_entropy = total_entropy + (double)temp_subset_center_label.n_rows / I*entropy;

			// calculate entropy of subset in right child
			denom = temp_subset_right_label.n_rows;
			entropy = 0;
			for (int c = 1; c <= classNum; c++)
			{
				double num = 0;
				for (int l = 0; l < temp_subset_right_label.n_rows; l++)
				{
					if (temp_subset_right_label(l) == c)
					{
						num = num + 1;
					}
				}
				double p = num / denom;
				if (p != 0)
					entropy = entropy - p*log2(p);
			}
			total_entropy = total_entropy + (double)temp_subset_right_label.n_rows / I*entropy;


			if (-total_entropy > max_entropy)
			{
				max_entropy = -total_entropy;

				resultNode->pt_dm1.x = floor(i / patchWidth);
				resultNode->pt_dm1.y = i % patchWidth;

				resultNode->pt_dm2.x = floor(j / patchWidth);
				resultNode->pt_dm2.y = j % patchWidth;


				for (int temp = 0; temp < 3; temp++)
				{
					if (subsets[temp].data != nullptr)
						delete (subsets[temp].data);
					if (subsets[temp].label != nullptr)
						delete (subsets[temp].label);
				}

				subsets[0].data = new mat(patches.cols(temp_subset_left_idx));
				subsets[0].label = new vec(temp_subset_left_label);

				subsets[1].data = new mat(patches.cols(temp_subset_center_idx));
				subsets[1].label = new vec(temp_subset_center_label);

				subsets[2].data = new mat(patches.cols(temp_subset_right_idx));
				subsets[2].label = new vec(temp_subset_right_label);
			}
		}

		resultNode->pt_dm1.x = resultNode->pt_dm1.x - (patchWidth + 1) / 2;
		resultNode->pt_dm1.y = resultNode->pt_dm1.y - (patchWidth + 1) / 2;

		resultNode->pt_dm2.x = resultNode->pt_dm2.x - (patchWidth + 1) / 2;
		resultNode->pt_dm2.y = resultNode->pt_dm2.y - (patchWidth + 1) / 2;
		cout << "Max: " << max_entropy << endl;
	}
	else
	{
		cout << "The number of data isn't enough!!" << endl;
	}
}