clear;

imageNum = 60000;
imageSize = 784;
imageWidth = round(sqrt(imageSize));

loadFileName = 'data_PCA_200.mat';
    
% open and read the data
load(loadFileName);

Nhids = [100;200;300;400;500;600;700];

error = zeros(500, size(Nhids, 1));
tr_error = zeros(500, size(Nhids, 1));

for i = 1:size(Nhids, 1)

    % get the training data
    data = (train_data(1:imageNum, 2:size(train_data, 2)));
    label = train_data(1:imageNum, 1);

    % get the testing data
    f_test_size = size(test_data, 1);
    f_test_data = (test_data(1:f_test_size, 2:size(test_data, 2)));
    f_test_label = test_data(1:f_test_size, 1);
    disp('End of read Data');


    % train 1 hidden layer NN model
    classNum = 10;
    eta = 0.5;
    alpha = 0.2;
    seg = 100;
    w = label + 1;
    x = data;

    [W_xh, W_hy, theta_h, theta_y, error(:, i), tr_error(:, i)]...
        = MLP(x, w, classNum,...
            f_test_data, f_test_label, eta, alpha, seg, Nhids(i));

end


figure(1);
for i = 1:size(error, 2)
    plot(error(error(:,i) > 0,i));
    hold on;
end
title('Test error');
grid on;
xlabel('iterations');
ylabel('error rate');
legend('Nhid=100','Nhid=200','Nhid=300','Nhid=400',...
    'Nhid=500','Nhid=600','Nhid=700');
        
figure(2);
for i = 1:size(tr_error, 2)
    plot(tr_error(tr_error(:,i) > 0,i));
    hold on;
end
title('Training error');
grid on;
xlabel('iterations');
ylabel('error rate');
legend('Nhid=100','Nhid=200','Nhid=300','Nhid=400',...
    'Nhid=500','Nhid=600','Nhid=700');