%
% Filename: optTress.m
% Purpose: find function q(x) which can maxmize entropy of each subset at
%          given node
% Input:  
%         patches - dataset at given node (D x I)
%         label - label corresponded to given data (I x 1)
%         classNum - number of classes of given dataset
% Output:
%         pt_dm1 - displacement between chosen point m1 and image center (2 x 1)
%         pt_dm2 - displacement between chosen point m2 and image center (2 x 1)
%         subset_left - data and label which will be saved in
%         left child node (cell 2 x 1)
%         subset_center - data and label which will be saved in
%         center child node (cell 2 x 1)
%         subset_right - data and label which will be saved in
%         right child node (cell 2 x 1)
%
function [pt_dm1, pt_dm2, subset_left, subset_center, subset_right] ...
    = optTree(patches, label, classNum, patchWidth)
    
    D = size(patches, 1);
    I = size(patches, 2);
    
    threshold = 10;
    max_entropy = -Inf;
    
    subset_left = cell(2, 1);
    subset_center = cell(2, 1);
    subset_right = cell(2, 1);
    
    for i = 1:D-1
        temp_pixels_m1 = patches(i, :);
        for j = (i + 1):D
            temp_pixels_m2 = patches(j, :);
            
            % compute pixels difference
            intensityDiff = temp_pixels_m1 - temp_pixels_m2;
            
            % decide which subset the data should belong to
            temp_subset_left_idx = intensityDiff < -threshold;
            temp_subset_center_idx = abs(intensityDiff) <= threshold;
            temp_subset_right_idx = intensityDiff > threshold;
            
            % get label of data in each subset
            temp_subset_left_label = label(temp_subset_left_idx, :);
            temp_subset_center_label = label(temp_subset_center_idx, :);
            temp_subset_right_label = label(temp_subset_right_idx, :);
            
            % calculate entropy of subset in left child
            denom = size(temp_subset_left_label, 1);
            entropy = 0;
            for c = 1:classNum
                 num = size(temp_subset_left_label(temp_subset_left_label == c), 1);
                 p = num / denom;
                 if p ~= 0
                    entropy = entropy - p*log2(p);
                 end
            end
            total_entropy = size(temp_subset_left_label, 1)/I*entropy;
            
            % calculate entropy of subset in center child
            denom = size(temp_subset_center_label, 1);
            entropy = 0;
            for c = 1:classNum
                 num = size(temp_subset_center_label(temp_subset_center_label == c), 1);
                 p = num / denom;
                 if p ~= 0
                    entropy = entropy - p*log2(p);
                 end
            end
            total_entropy = total_entropy + size(temp_subset_center_label, 1)/I*entropy;
            
            % calculate entropy of subset in right child
            denom = size(temp_subset_right_label, 1);
            entropy = 0;
            for c = 1:classNum
                 num = size(temp_subset_right_label(temp_subset_right_label == c), 1);
                 p = num / denom;
                 if p ~= 0
                    entropy = entropy - p*log2(p);
                 end
            end
            total_entropy = total_entropy + size(temp_subset_right_label, 1)/I*entropy;
                        
            if -total_entropy > max_entropy
                max_entropy = -total_entropy;
                
                pt_dm1 = [floor(i/patchWidth) + 1; mod(i - 1, patchWidth) + 1];
                pt_dm2 = [floor(j/patchWidth) + 1; mod(j - 1, patchWidth) + 1];
                
                subset_left{1} = patches(:, temp_subset_left_idx);
                subset_center{1} = patches(:, temp_subset_center_idx);
                subset_right{1} = patches(:, temp_subset_right_idx);
                
                subset_left{2} = label(temp_subset_left_idx, :);
                subset_center{2} = label(temp_subset_center_idx, :);
                subset_right{2} = label(temp_subset_right_idx, :);
                
            end
            
        end
    end
    
    pt_dm1 = round(pt_dm1 - (patchWidth+1)/2*ones(2, 1));
    pt_dm2 = round(pt_dm2 - (patchWidth+1)/2*ones(2, 1));
    
end