%==========================================================================%
%
% Input: w: World state (I x 1)
%        x: Training data (I x D)
%        N: number of class
%        precision: the algorithm stops when the difference between
%                      the previous and the new likelihood is < precision.
%        ita: learning rate of gradient decent method
%        
% Output: parameters phi (N X D)
%==========================================================================%
function [phi] = multiLogisticBFGS(w, x, N, ita)
    
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
    
    Mat_D = zeros(D, D, N);
    
    for n = 1:N
        Mat_D(:, :, n) = eye(D, D);
    end
    
    d = zeros(N, D);
    
    [L, g] = optMultiLogistic(W, x, phi);
    
    pre_L = L;
    pre_g = g;
    
    searchCount = 4;
    
    
    itas = zeros(searchCount, 1);
    for i = 1:searchCount
        itas(i) = i*ita/searchCount;
    end
    
    itas_t = itas;
    
    while true
       
       L_min = Inf;
       for i = 1:searchCount
           
           for n = 1:N
               d(n, :) = -itas_t(i)*squeeze(Mat_D(:,:,n))*g(n, :)';
           end

           phi_t = phi + d;

           [L, g] = optMultiLogistic(W, x, phi_t);
           
           if L <= L_min
               L_min = L;
               g_min = g;
               phi_min = phi_t;
           end
           
       end
       
       L = L_min;
       g = g_min;
       phi = phi_min;
       

       if L > pre_L || count >= 200
           break;
       end
       display(L);

       
       y = g - pre_g;
       pre_g = g;   
       pre_L = L;
       
       for n = 1:N
           Mat_D(:,:,n) = (eye(D, D) - (d(n, :)'*y(n, :))./(d(n, :)*y(n, :)'))...
                            *squeeze(Mat_D(:,:,n))...
                            *(eye(D, D) - (y(n, :)'*d(n, :))./(y(n, :)*d(n, :)'))...
                            + (d(n, :)'*d(n, :))./(d(n, :)*y(n, :)');
       end
       
       if L < 1
           itas_t = itas .* L;
       end
       
       count = count + 1;
       display(count);
    end

end