%==========================================================================%
% Filename: multiLogistic.m
% Purpose: Train softmax regression model
% Input: w: World state (target value) (I x 1)
%        x: Training data (I x D)
%        N: number of class
%        eta: learning rate of gradient decent method
%        
% Output: parameters phi (N X D)
%==========================================================================%
function [phi] = multiLogistic(w, x, N, eta)
    
    I = size(x, 1);
    D = size(x, 2);
    
    % initialize parameters phi
    phi = 0.005*ones(N, D);
    
    count = 1;
    
    % set up world state matrix W
    W = zeros(N, I);
    for i = 1:I
        W(w(i), i) = 1;
    end
    
    while true
       
       % get cost and gradient base on current parameters and data
       [L, g] = optMultiLogistic(W, x, phi);
       
       % update parameters using gradient descent method
       phi = phi - eta*g;
       
       % algorithm termination condition
       if L < 0.3 || count >= 3000
           break;
       end
       display(L);
       
       % make learning rate adapt to cost
       if L < 1
           eta = eta * L;
       end
       
       count = count + 1;
       display(count);
    end

end