% Filename: testTree.m
% Purpose: output the propability of each class based on given image
% Input: patch - image patch
%        nodeObj - root(node) of the tree
% Output: lambda - posterior probability of each class
function [lambda] = testTree(patch, nodeObj)

    % If we've reached the leaf
    if nodeObj.isLeaf == 1
        % return the probability of each class
        lambda = nodeObj.lambda;
    else
        patchCenter = (size(patch, 1) - 1)/2;
        
        % the decision rule is simply based on calculating the difference
		% of image intensity
        pt1 = nodeObj.pt_dm1 + (patchCenter + 1).*ones(2, 1);
        pt2 = nodeObj.pt_dm2 + (patchCenter + 1).*ones(2, 1);
        diff = patch(pt1(2, 1), pt1(1, 1)) - patch(pt2(2, 1), pt2(1, 1));
        
        % decide which child node we should go.
        if diff < -10
            lambda = testTree(patch, nodeObj.childNode_left);
        elseif abs(diff) <= 10
            lambda = testTree(patch, nodeObj.childNode_center);
        else
            lambda = testTree(patch, nodeObj.childNode_right);
        end
    end
end