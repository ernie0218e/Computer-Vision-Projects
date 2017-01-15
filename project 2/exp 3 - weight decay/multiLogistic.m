%==========================================================================%
%
% Input: w: World state (I x 1)
%        x: Training data (I x D)
%        N: number of class
%        eta: learning rate of gradient decent method
%        
% Output: parameters phi (N X D)
%==========================================================================%
function [phi, e, t_e] = multiLogistic(w, x, N, eta, lambda, test_w, test_x)
    
    I = size(x, 1);
    D = size(x, 2);
    
    % initialize parameters phi with random var
    phi = 0.005*ones(N, D);
    
    count = 1;
    
    % set up world state matrix W
    W = zeros(N, I);
    for i = 1:I
        W(w(i), i) = 1;
    end
    
    times = 1000;
    e = zeros(times, 1);
    t_e = zeros(times, 1);
    
    while true
       
       [L, g] = optMultiLogistic(W, x, phi, lambda);
       
       phi = phi - eta*g;
       
       e(count) = testMulticlassLogistic(test_x, test_w, phi);
       t_e(count) = testMulticlassLogistic(x, w - 1, phi);

       if count >= times
           break;
       end
       display(L);
       
       if L < 1
           eta = eta * L;
       end
       
       count = count + 1;
       display(count);
    end

end