%
% Input
function  [W_xh, W_hy, theta_h, theta_y] = MLP_GPU(x, w, classNum, precision)
    
    x = gpuArray(x);
    w = gpuArray(w);

    Ninp = size(x, 2);
    Nout = classNum;
    Nhid = round((Ninp + Nout) / 2);
    
    I = size(x, 1);
    
    W_xh = gpuArray(2*rand(Ninp, Nhid) - 1);
    W_hy = gpuArray(2*rand(Nhid, Nout) - 1); 
    
    theta_h = gpuArray(2*rand(Nhid, 1) - 1);
    theta_y = gpuArray(2*rand(Nout, 1) - 1);
    
    
    ita = 0.5;
    
    alpha = 0.2;
    
    pre_cost = 0;
    count = 0;
    
    while true
        delta_W_hy = gpuArray(zeros(Nhid, Nout));
        delta_theta_y = gpuArray(zeros(Nout, 1));
        delta_W_xh = gpuArray(zeros(Ninp, Nhid));
        delta_theta_h = gpuArray(zeros(Nhid, 1));

        cost = 0;
        
        % for each training data
        for i = 1:I

            % Caculate assumed output of hidden layer
            % net_h:(Nhid x 1) H:(Nhid x 1)
            net_h = W_xh'*x(i, :)' - theta_h;
            H = 1 ./ (1 + exp(-net_h));

            % Caculate assumed output of output layer
            % net_y:(Nout x 1) Y:(Nout x 1)
            net_y = W_hy'*H - theta_y;
            Y = 1 ./ (1 + exp(-net_y));

            % Construct target value based on world state w
            T = gpuArray(zeros(Nout, 1));
            T(w(i)) = 1;

            % Caculate the difference of output layer
            % delta_y:(Nout x 1)
            delta_y = Y.*(1-Y).*(T - Y);

            % Caculate the difference of hidden layer
            % delta_h:(Nhid x 1)
            delta_h = H.*(1-H).*(W_hy*delta_y);

            % Caculate correction in W and theta of each layer
            % hidden layer to output layer
            delta_W_hy = ita.*(delta_y*H')' + alpha.*delta_W_hy;
            delta_theta_y = -ita.*delta_y + alpha.*delta_theta_y;

            % input layer to hidden layer
            delta_W_xh = ita.*(delta_h*x(i, :))' + alpha.*delta_W_xh;
            delta_theta_h = -ita.*delta_h + alpha.*delta_theta_h;

            % Update W and theta of each layer
            W_hy = W_hy + delta_W_hy;
            theta_y = theta_y + delta_theta_y;
            W_xh = W_xh + delta_W_xh;
            theta_h = theta_h + delta_theta_h;

            cost = cost + sum((T - Y).^2);
        end
        
        cost = sqrt(cost/I);
        
%         if abs(pre_cost - cost) < precision
%             break;
%         else
%             pre_cost = cost;
%         end
        if cost < 0.1 || count >= 500
            break;
        end

        display(cost);
        count = count + 1;
        display(count);
        
    end
    
    W_xh = gather(W_xh);
    W_hy = gather(W_hy);
    theta_h = gather(theta_h);
    theta_y = gather(theta_y);

end