% 
% Purepose: Use backpropagation algorithm to train NN
% Input: x - data (I x D)
%        w - world state (I x 1)
%        classNum - Number of class
%        eta - learning rate
%        alpha - weight of inertia
%        iteration - number of iterations
% Output: W_xh1 - weight of net between input layer and hidden layer 1
%                (Ninp x Nhid1)
%         W_h1h2 - weight of net between hidden layer 1 and hidden layer 2
%                (Nhid1 x Nhid2)
%         W_h2y - weight of net between hidden layer 2 and output layer
%                (Nhid2 x Nout)
%         theta_h1 - bias of hidden layer 1 (Nhid1 x 1)
%         theta_h2 - bias of hidden layer 2 (Nhid1 x 1)
%         theta_y - bias of output layer (Nout x 1)
%
function  [cost, W_xh1, W_h1h2, W_h2y, theta_h1, theta_h2, theta_y] = MLP(x, w, classNum, eta, alpha, iteration)
    
    % store the root-mean-square error of each iteration
    cost = zeros(iteration, 1);
    
    % set number of nodes in input layer as the dimension of data
    Ninp = size(x, 2);
    
    % set number of nodes in output layer as number of class
    Nout = classNum;
    
    % set number of nodes in hidden layer as (Ninp + Nout) / 2
    Nhid1 = round((Ninp + Nout) / 2);
    Nhid2 = round((Nhid1 + Nout) / 2);
    
    I = size(x, 1);
    
    % initialize weight of network
    W_xh1 = rand(Ninp, Nhid1) - 0.5;
    W_h1h2 = rand(Nhid1, Nhid2) - 0.5;
    W_h2y = rand(Nhid2, Nout) - 0.5; 
    
    % initialize bias of network
    theta_h1 = rand(Nhid1, 1) - 0.5;
    theta_h2 = rand(Nhid2, 1) - 0.5;
    theta_y = rand(Nout, 1) - 0.5;
    
    eta_t = eta;
    
    count = 1;
    
    % Construct target value based on world state w
    % T:(Nout x I)
    T = zeros(Nout, I);
    for i = 1:I
        T(w(i), i) = 1;
    end
    
    delta_W_h2y = zeros(Nhid2, Nout);
    delta_theta_y = zeros(Nout, 1);
    
    delta_W_h1h2 = zeros(Nhid1, Nhid2);
    delta_theta_h2 = zeros(Nhid2, 1);
    
    delta_W_xh1 = zeros(Ninp, Nhid1);
    delta_theta_h1 = zeros(Nhid1, 1);

    
    while true
        
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

        % Caculate the difference of output layer
        % delta_y:(Nout x I)
        delta_y = Y.*(1-Y).*(T - Y);

        % Caculate the difference of hidden layer 2
        % delta_h2:(Nhid2 x I)
        delta_h2 = H2.*(1-H2).*(W_h2y*delta_y);
        
        % Caculate the difference of hidden layer 1
        % delta_h1:(Nhid1 x I)
        delta_h1 = H1.*(1-H1).*(W_h1h2*delta_h2);

        % Caculate correction in W and theta of each layer
        % hidden layer 2 to output layer
        delta_W_h2y = eta_t.*(delta_y*H2')'./I + alpha.*delta_W_h2y;
        delta_theta_y = sum(-eta_t.*delta_y, 2)./I + alpha.*delta_theta_y;
        
        % hidden layer 1 to hidden layer 2
        delta_W_h1h2 = eta_t.*(delta_h2*H1')'./I + alpha.*delta_W_h1h2;
        delta_theta_h2 = sum(-eta_t.*delta_h2, 2)./I + alpha.*delta_theta_h2;

        % input layer to hidden layer
        delta_W_xh1 = eta_t.*(delta_h1*x)'./I + alpha.*delta_W_xh1;
        delta_theta_h1 = sum(-eta_t.*delta_h1, 2)./I + alpha.*delta_theta_h1;

        % Update W and theta of each layer
        W_h2y = W_h2y + delta_W_h2y;
        theta_y = theta_y + delta_theta_y;
        W_xh1 = W_xh1 + delta_W_xh1;
        theta_h1 = theta_h1 + delta_theta_h1;
        W_h1h2 = W_h1h2 + delta_W_h1h2;
        theta_h2 = theta_h2 + delta_theta_h2;

        % calculate cost
        cost(count) = sum(sum((T - Y).^2));
        % normalize cost
        cost(count) = sqrt(cost(count)/I);
        
        
        % algorithm terminated condition
        if count >= iteration
            break;
        end
        
        
        display(cost(count));
        count = count + 1;
        display(count);
        
    end
end