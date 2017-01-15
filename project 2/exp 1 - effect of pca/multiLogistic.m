%==========================================================================%
%
% Input: w: World state (I x 1)
%        x: Training data (I x D)
%        N: number of class
%        ita: learning rate of gradient decent method
%        
% Output: parameters phi (N X D)
%==========================================================================%
function [phi] = multiLogistic(w, x, N, ita)
    
    I = size(x, 1);
    D = size(x, 2);
    
    % initialize parameters phi with random var
    phi = 0.005*ones(N, D);
    pre_L = 0;
    
    count = 0;
    
    % set up world state matrix W
    W = zeros(N, I);
    for i = 1:I
        W(w(i), i) = 1;
    end
    
    while true
       
       [L, g] = optMultiLogistic(W, x, phi);
       
       phi = phi - ita*g;
       

       if count >= 1000
           break;
       end
       display(L);
       
       count = count + 1;
       display(count);
    end

end