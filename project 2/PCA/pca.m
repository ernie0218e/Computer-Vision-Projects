% Filename: pca.m
% Purpose: compute pca of data
% Input: x - Data (I x D)
%        K - new size of dimension of data
% Output: phi - Dimension-reduced data (I x K)
%         V - Coordinate transform matrix (D x K)
function [phi, V] = pca(x, K)
    
    I = size(x, 1);
    
    % data mean (D x 1)
    mean = sum(x, 1)' ./ I;
    
    % compute scatter matrix
    S = x - repmat(mean', I, 1);
    S = S' * S;
    
    
    [V, lambda] = eigs(S, K);
    
    phi = (V' * x')';
    
end