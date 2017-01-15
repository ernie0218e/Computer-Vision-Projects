% Filename: MLP_mnist.m
% Purpose: Train MultiLayer Perceptron (2-NN) model and test it
clear;

% open and read the data
load('data_PCA_100.mat');


imageNum = 60000;
ratio = 1;
imageSize = round((sqrt(784)*ratio)^2);
imageWidth = round(sqrt(imageSize));

% get the training data
data = (train_data(1:imageNum, 2:size(train_data, 2)));
label = train_data(1:imageNum, 1);

% get the testing data
f_test_size = size(test_data, 1);
f_test_data = (test_data(1:f_test_size, 2:size(test_data, 2)));
f_test_label = test_data(1:f_test_size, 1);

disp('End of read Data');


% number of class
classNum = 10;

% learning rate
eta = 0.5;

w = label + 1;
x = data;

% weight of Inertia
alpha = 0.2;

% number of mini-batch
seg = 1000;

% train model and get parameters
[W_xh, W_hy, theta_h, theta_y] = MLP(x, w, classNum, eta, alpha, seg);

% compute test error
error = zeros(10, 1);
% get output of neural network given test data
Y = MLP_Predict(f_test_data, W_xh, W_hy, theta_h, theta_y);
for i = 1:f_test_size
    % find max value from the output of neural network
    % the label correspond to the max value is predicted result
    [maxVal, maxLabel] = max(Y(:, i));
    maxLabel = maxLabel - 1;
    
    if maxLabel ~= f_test_label(i)
        error(f_test_label(i) + 1) = error(f_test_label(i) + 1) + 1;
    end
end

e = sum(error)/f_test_size;

% compute training error
train_error = zeros(10, 1);
% get output of neural network given training data
Y = MLP_Predict(data, W_xh, W_hy, theta_h, theta_y);
for i = 1:imageNum
     % find max value from the output of neural network
    % the label correspond to the max value is predicted result
    [maxVal, maxLabel] = max(Y(:, i));
    maxLabel = maxLabel - 1;
    
    if maxLabel ~= label(i)
        train_error(label(i) + 1) = train_error(label(i) + 1) + 1;
    end
end

train_e = sum(train_error)/imageNum;