% 
% Input: x - data (I x D)
%        w - world state (I x 1)
%        classNum - Number of class
%        eta - learning rate
%        alpha - weight of inertia
%        seg - number of segmentation of data
% Output: W_xh - weight of net between input layer and hidden layer
%                (Ninp x Nhid)
%         W_hy - weight of net between hidden layer and output layer
%                (Nhid x Nout)
%         theta_h - bias of hidden layer (Nhid x 1)
%         theta_y - bias of output layer (Nout x 1)
%
function  [W_xh, W_hy, theta_h, theta_y, e, t_e] = MLP(x, w, classNum, ...
                                                        test_x, test_w, eta, alpha, seg)
    

    Ninp = size(x, 2);
    Nout = classNum;
    Nhid = round((Ninp + Nout) / 2);
    
    I = size(x, 1);
    
    W_xh = 2*rand(Ninp, Nhid) - 1;
    W_hy = 2*rand(Nhid, Nout) - 1; 
    
    theta_h = 2*rand(Nhid, 1) - 1;
    theta_y = 2*rand(Nout, 1) - 1;
    
    eta_t = eta;
    
    count = 1;
    
    times = 500;
    e = zeros(times, 1);
    t_e = zeros(times, 1);
    
    % Construct target value based on world state w
    % T:(Nout x I)
    T = zeros(Nout, I);
    for i = 1:I
        T(w(i), i) = 1;
    end
    
    batchSize = round(I/seg);
    
    while true

        delta_W_hy = zeros(Nhid, Nout);
        delta_theta_y = zeros(Nout, 1);
        delta_W_xh = zeros(Ninp, Nhid);
        delta_theta_h = zeros(Nhid, 1);

        cost = 0;
        
        chose = randperm(I);
        
        for t = 1:seg
            xx = x(chose(batchSize*(t-1) + 1:batchSize*t), :);
            TT = T(:, chose(batchSize*(t-1) + 1:batchSize*t));
        
            % Caculate assumed output of hidden layer
            % net_h:(Nhid x I) H:(Nhid x I)
            net_h = W_xh'*xx' - repmat(theta_h, 1, batchSize);
            H = 1 ./ (1 + exp(-net_h));

            % Caculate assumed output of output layer
            % net_y:(Nout x I) Y:(Nout x I)
            net_y = W_hy'*H - repmat(theta_y, 1, batchSize);
            Y = 1 ./ (1 + exp(-net_y));

            % Caculate the difference of output layer
            % delta_y:(Nout x I)
            delta_y = Y.*(1-Y).*(TT - Y);

            % Caculate the difference of hidden layer
            % delta_h:(Nhid x I)
            delta_h = H.*(1-H).*(W_hy*delta_y);

            % Caculate correction in W and theta of each layer
            % hidden layer to output layer
            delta_W_hy = eta_t.*(delta_y*H')'./batchSize + alpha.*delta_W_hy;
            delta_theta_y = sum(-eta_t.*delta_y, 2)./batchSize + alpha.*delta_theta_y;

            % input layer to hidden layer
            delta_W_xh = eta_t.*(delta_h*xx)'./batchSize + alpha.*delta_W_xh;
            delta_theta_h = sum(-eta_t.*delta_h, 2)./batchSize + alpha.*delta_theta_h;

            % Update W and theta of each layer
            W_hy = W_hy + delta_W_hy;
            theta_y = theta_y + delta_theta_y;
            W_xh = W_xh + delta_W_xh;
            theta_h = theta_h + delta_theta_h;

            cost = cost + sum(0.5*sum((TT - Y).^2));
            
        end
        
        e(count) = testMLP(test_x, test_w, W_xh, W_hy, theta_h, theta_y);
        t_e(count) = testMLP(x, w-1, W_xh, W_hy, theta_h, theta_y);
        
        cost = sqrt(cost/I);
        if count >= times
            break;
        end

        if cost < 1
            eta_t = cost*eta;
        end
        
        display(cost);
        count = count + 1;
        display(count);
        
    end
end