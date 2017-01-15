% 
% Input: parameters phi (N x D)
%        data (I x D): I stands for number of data; D is the dimension
% Output: output of softmax function lambda (N x I)
%
function [lambda] = linearSoftMax(phi, x)
    
    % get number of class N
    N = size(phi, 1);
    
    
    % caculate activation act (N x I)
    act = phi*x';
    
    % find max value in each column of act
    % y (1 x I)
    y = max(act, [], 1);
    
    % den (1 x I)
    den = sum(exp(act - repmat(y, N, 1)), 1);
    
    lambda = exp(act - repmat(y, N, 1)) ./ repmat(den, N, 1);

%     M = bsxfun(@minus,phi*x,max(phi*x, [], 1));
%     M = exp(M);
%     lambda = bsxfun(@rdivide, M, sum(M));
    
end