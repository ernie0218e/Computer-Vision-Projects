% 
% Input: act_p - activation calculate previously (N x I)
%        phi_zero -  bias (N x 1)
%        phi - parameters (N x 1)
%        zeta - parameters of nonlinear function (N x D)
%        x - single data (I x D)
% Output: lambda - output of softmax function (N x I)
%         act - output new activation (N x I)
%
function [lambda, act] = nonLinearSoftMax(act_p, phi_zero, phi, zeta, x)
    
    % get number of class N
    N = size(phi, 1);
    
    % get number of data
    I = size(x, 1);
    
    % caculate activation act (N x I)
    act = act_p + repmat(phi_zero, 1, I) + repmat(phi, 1, I).*atan(zeta*x');
    
    % find max value in each column of act
    % y (1 x I)
    y = max(act, [], 1);
    
    % den (1 x I)
    den = sum(exp(act - repmat(y, N, 1)), 1);
    
    lambda = exp(act - repmat(y, N, 1)) ./ repmat(den, N, 1);

    
end