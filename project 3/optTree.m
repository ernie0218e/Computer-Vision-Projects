%
% Filename: optTress.m
% Purpose: find function q(x) which can maxmize entropy of each subset at
%          given node
% Input:  
%         patches - dataset at given node (D x I)
%         label - label corresponded to given data (I x 1)
%         classNum - number of classes of given dataset
% Output:
%         pt_m1 - chosen point m1
%         pt_m2 - chosen point m2
%         subset_left - index of original dataset which will be saved in
%         left child node
%         subset_center - index of original dataset which will be saved in
%         center child node
%         subset_right - index of original dataset which will be saved in
%         right child node
%
function [pt_m1, pt_m2, subset_left, subset_center, subset_right] ...
    = optTree(patches, label, classNum)
    
    D = size(patches, 1);
    I = size(patches, 2);
    
    threshold = 10;
    
    for i = 1:D-1
        temp_pixels_m1 = patches(i, :);
        for j = (i + 1):D
            temp_pixels_m2 = patches(j, :);
            
            intensityDiff = temp_pixels_m1 - temp_pixels_m2;
            
        end
    end
    
end