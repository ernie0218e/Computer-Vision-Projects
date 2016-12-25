% Filename: reconstruct.m
% Purpose: reconstruct the Tree
% Input: fileId - file pointer (for reading data)
%        classNum - number of classes
% Output: nodeObj - can be viewed as the entry of this tree
function [] = reconstruct(fileID, classNum, nodeObj)
    % read current depth
    currentDepth = fscanf(fileID, '%d', 1);
    % If we've reached the leaf
    if currentDepth == -1
        % mark this node
        nodeObj.isLeaf = 1;
        % read the probability of each class
        nodeObj.lambda = fscanf(fileID, '%lf', classNum);
    else
        % read in the decision rule
        nodeObj.pt_dm1 = fscanf(fileID, '%d', 2);
        nodeObj.pt_dm2 = fscanf(fileID, '%d', 2);
        
        % recursively generate more child nodes
        % depth-first
        nodeObj.childNode_left = Node();
        reconstruct(fileID, classNum, nodeObj.childNode_left);
        nodeObj.childNode_center = Node();
        reconstruct(fileID, classNum, nodeObj.childNode_center);
        nodeObj.childNode_right = Node();
        reconstruct(fileID, classNum, nodeObj.childNode_right);
    end
end