% 
% Filename: MLP_training_mix.m
% Purpose: Train MultiLayer Perceptron (2-NN) model and test it
%
clear;

% open and read the data
load('data.mat');
load('label.mat');

% define constants
imageNum = size(data, 2);
imageSize = size(data, 1);
imageWidth = round(sqrt(imageSize));

% make testing data
f_test_size = 5000;
chose = randperm(imageNum, f_test_size);

test_data = data(:, chose);
test_label = label(chose, :);

data(:, chose) = [];
label(chose, :) = [];

disp('End of read Data');


% number of class
classNum = 36;

% learning rate
eta = 1;

% number of iterations
iteration = 10000;

% label and data
w = label;
x = data';

% weight of Inertia
alpha = 0.5;

% train model and get parameters
[cost, W_xh1, W_h1h2, W_h2y, theta_h1, theta_h2, theta_y] = MLP(x, w, classNum, eta, alpha, iteration);
save('cost.mat', 'cost');

% compute test error
error = zeros(classNum, 1);
% get output of neural network given test data
Y = MLP_Predict(test_data', W_xh1, W_h1h2, W_h2y, theta_h1, theta_h2, theta_y);
for i = 1:f_test_size
    % find max value from the output of neural network
    % the label correspond to the max value is predicted result
    [maxVal, maxLabel] = max(Y(:, i));
    
    if maxLabel ~= test_label(i)
        error(test_label(i)) = error(test_label(i)) + 1;
    end
end

e = sum(error)/f_test_size;

% compute training error
train_error = zeros(classNum, 1);
% get output of neural network given training data
Y = MLP_Predict(data', W_xh1, W_h1h2, W_h2y, theta_h1, theta_h2, theta_y);
for i = 1:size(data, 2)
     % find max value from the output of neural network
    % the label correspond to the max value is predicted result
    [maxVal, maxLabel] = max(Y(:, i));
    
    if maxLabel ~= label(i)
        train_error(label(i)) = train_error(label(i)) + 1;
    end
end

train_e = sum(train_error)/size(data, 2);

% save parameters of neural network for predicting
parameters = cell(6, 1);
parameters{1, 1} = W_xh1;
parameters{2, 1} = W_h1h2;
parameters{3, 1} = W_h2y;
parameters{4, 1} = theta_h1;
parameters{5, 1} = theta_h2;
parameters{6, 1} = theta_y;

save('parameters.mat', 'parameters');