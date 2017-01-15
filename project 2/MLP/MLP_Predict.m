% Purpose: compute output of NN given data
% Input: x - data (I x D)
%        W_xh - weight of net between input layer and hidden layer
%               (Ninp x Nhid)
%        W_hy - weight of net between hidden layer and output layer
%               (Nhid x Nout)
%        theta_h - bias of hidden layer (Nhid x 1)
%        theta_y - bias of output layer (Nout x 1)
% Output: Y - predicted targe value (Nout x I)
%
function [Y] = MLP_Predict(x, W_xh, W_hy, theta_h, theta_y)
    
    I = size(x, 1);
    
    % Caculate output of hidden layer
    % net_h:(Nhid x I) H:(Nhid x I)
    net_h = W_xh'*x' - repmat(theta_h, 1, I);
    H = 1 ./ (1 + exp(-net_h));

    % Caculate output of output layer
    % net_y:(Nout x I) Y:(Nout x I)
    net_y = W_hy'*H - repmat(theta_y, 1, I);
    Y = 1 ./ (1 + exp(-net_y));
    
end