%==========================================================================%
%
% Input: W - World state which is in the form of a 'index' matrix (N x I)
%        x - all training data (I x D)
%        phi - parameters (N x D)
% Output: L - cost
%         g - gradient (N x D)
%
%==========================================================================%
function [L, g] = optMultiLogistic(W, x, phi)
    % initailize common const variables

    % I: number of data
    I = size(x, 1);
    
    % weight of weight decay term
    lambda = 0.0001;
    
    
    % calculate softmax of all data
    y = linearSoftMax(phi, x);
    
    % calculate sum of log probability (cost)
    L = -sum(log(y(logical(W))));
    
    % calculate gradient
    g = (y - W)*x;
    
    % nomalize cost and add weight decay term
    L = L ./ I + lambda/2 * sum(sum(phi.^2));
    
    % add weight decay term to gradient, too
    g = g ./ I + lambda*phi;
    
end