%==========================================================================%
%
% Input: W - World state which is in the form of a 'index' matrix (N x I)
%        x - training data (I x D)
%        act_p - previous activation (N x I)
%        phi_zero - bias (N x 1)
%        phi - parameters (N x 1)
%        zeta - parameters of nonlinear function (N x D)
% Output: L - cost
%         phi_zero_g - gradient of bias (N x 1)
%         phi_g - gradient of parameters (N x 1)
%         zeta_g - gradient of parameters of nonlinear function (N x D)
%         act - new activation (N x I)
%
%==========================================================================%
function [L, act, phi_zero_g, phi_g, zeta_g] = optNonLinearMultiLogistic(W, x, act_p, phi_zero, phi, zeta)
    % initailize common const variables
    % N: number of class
    N = size(phi, 1);
    
    % I: number of data
    I = size(x, 1);
    
    % D: data dimension
    D = size(x, 2);
  
    
    % weight of weight decay term
    lambda = 0.0001;
    
          
    % compute prediction y and new activation act
    [y, act] = nonLinearSoftMax(act_p, phi_zero, phi, zeta, x);

    % update log likelihood
    % calculate sum of log probability (cost)
    L = -sum(log(y(logical(W))));
    
    % calculate gradient
    % phi_zero_g (N x 1)
    phi_zero_g = sum((y - W), 2);
    % phi_g (N x 1)
    phi_g = sum((y - W).*atan(zeta*x'), 2);
    % zeta_g (N x D)
    zeta_g = (y - W).*repmat(phi, 1, I).*(1./(ones(N, I)+(zeta*x').^2))*x;
        
    
    L = L ./ I;

end