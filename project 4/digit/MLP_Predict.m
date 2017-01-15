% Purpose: compute output of NN given data
% Input: x - data (I x D)
%        W_xh1 - weight of net between input layer and hidden layer 1
%               (Ninp x Nhid1)
%        W_h1h2 - weight of net between hidden layer 1 and hidden layer 2
%               (Nhid1 x Nhid2)
%        W_h2y - weight of net between hidden layer 2 and output layer
%               (Nhid2 x Nout)
%        theta_h1 - bias of hidden layer 1 (Nhid1 x 1)
%        theta_h2 - bias of hidden layer 2 (Nhid1 x 1)
% Output: Y - predicted targe value (Nout x I)
%
function [Y] = MLP_Predict(x, W_xh1, W_h1h2, W_h2y, theta_h1, theta_h2, theta_y)
    
    I = size(x, 1);
    
    % Caculate assumed output of hidden layer 1
    % net_h1:(Nhid1 x I) H1:(Nhid1 x I)
    net_h1 = W_xh1'*x' - repmat(theta_h1, 1, I);
    H1 = 1 ./ (1 + exp(-net_h1));

    % Caculate assumed output of hidden layer 2
    % net_h2:(Nhid2 x I) H2:(Nhid2 x I)
    net_h2 = W_h1h2'*H1 - repmat(theta_h2, 1, I);
    H2 = 1 ./ (1 + exp(-net_h2));

    % Caculate assumed output of output layer
    % net_y:(Nout x I) Y:(Nout x I)
    net_y = W_h2y'*H2 - repmat(theta_y, 1, I);
    Y = 1 ./ (1 + exp(-net_y));
    
end