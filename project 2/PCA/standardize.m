% Filename: standardize.m
% Purpose: standardize input data
% Input: data - (I x D)
% Output: std_data - (I x D)
function [std_data] = standardize(data)
    I = size(data, 1);
    
    % mean of each pixel (1 x D)
    mean = sum(data, 1) ./ I;
    
    % compute covariance matrix
    XC = data - repmat(mean, I, 1);
    C = XC' * XC ./ I;
    
    % compute standard deviation (D x 1)
    std = sqrt(diag(C));
    
    % let standard deviation equals to one if std==0
    std(std == 0) = 1;
    
    % std_data = (data - meanOfData)/std
    std_data = XC ./ repmat(std', I, 1);
    
end