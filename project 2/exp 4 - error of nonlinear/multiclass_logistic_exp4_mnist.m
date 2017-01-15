clear;

imageNum = 60000;
imageSize = 784;
imageWidth = round(sqrt(imageSize));

loadFileName = 'data_PCA_200.mat';
    
% open and read the data
load(loadFileName);

theta = [10 20;20 10;50 1];

errorSR = cell(size(theta, 1), 1);
tr_errorSR = cell(size(theta, 1), 1);

for i = 1:size(theta, 1);

    % get the training data
    data = (train_data(1:imageNum, 2:size(train_data, 2)));
    label = train_data(1:imageNum, 1);

    % get the testing data
    f_test_size = size(test_data, 1);
    f_test_data = (test_data(1:f_test_size, 2:size(test_data, 2)));
    f_test_label = test_data(1:f_test_size, 1);
    disp('End of read Data');


    classNum = 10;
    eta = 0.0001;
    w = label + 1;
    x = data;
    K = theta(i, 1);
    multiplier = theta(i, 2);

    [phi_zero, phi, zeta, errorSR{i, 1}, tr_errorSR{i, 1}] = ...
        nonLinearMultiLogistic(w, x, classNum, K, eta, multiplier, f_test_label, f_test_data);
end

figure(1);
for i = 1:size(errorSR, 1)
    plot(errorSR{i});
    hold on;
end
title('Test error');
grid on;
xlabel('iterations');
ylabel('error rate');
legend('K=10 multiplier=20','K=20 multiplier=10','K=50 multiplier=1');
        
figure(2);
for i = 1:size(tr_errorSR, 1)
    plot(tr_errorSR{i});
    hold on;
end
title('Training error');
grid on;
xlabel('iterations');
ylabel('error rate');
legend('K=10 multiplier=20','K=20 multiplier=10','K=50 multiplier=1');

