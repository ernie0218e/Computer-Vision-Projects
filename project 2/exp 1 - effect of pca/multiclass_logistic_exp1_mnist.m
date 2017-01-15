clear;

imageNum = 60000;
imageSize = 784;
imageWidth = round(sqrt(imageSize));

e = zeros(15, 4);
train_e = zeros(15, 4);

useNN = 1;
useNonlinear = 1;
useSR = 1;
useSRBFGS = 1;

for c = 11:15
    
    if c ~= 15
        fileNamePrefix = 'data_PCA_';
        loadFileName = strcat(fileNamePrefix, int2str(50*c), '.mat');
    else
        loadFileName = 'data_std.mat';
    end
    
    % open and read the data
    load(loadFileName);


    % get the training data
    data = (train_data(1:imageNum, 2:size(train_data, 2)));
    label = train_data(1:imageNum, 1);

    % get the testing data
    f_test_size = size(test_data, 1);
    f_test_data = (test_data(1:f_test_size, 2:size(test_data, 2)));
    f_test_label = test_data(1:f_test_size, 1);
    disp('End of read Data');
    
    if useNonlinear == 1
        % train nonlinear softmax regression model
        classNum = 10;
        eta = 0.0001;
        K = 10;
        w = label + 1;
        x = data;
        [phi_zero, phi, zeta] = nonLinearMultiLogistic(w, x, classNum, K, eta);

        % get test error and training error
        e(c, 3) = testNonlinearMulticlassLogistic(data, label, ...
                                                    phi_zero, phi, zeta, K);
        train_e(c, 3) = testNonlinearMulticlassLogistic(f_test_data, f_test_label, ...
                                                    phi_zero, phi, zeta, K);
    end
    
    if useNN == 1
        % train 1 hidden layer NN model
        classNum = 10;
        eta = 0.5;
        alpha = 0.2;
        seg = 100;
        w = label + 1;
        x = data;

        [W_xh, W_hy, theta_h, theta_y] = MLP(x, w, classNum, eta, alpha, seg);

        % get test error and training error
        e(c, 4) = testMLP(data, label, W_xh, W_hy, theta_h, theta_y);
        train_e(c, 4) = testMLP(f_test_data, f_test_label, W_xh, W_hy, theta_h, theta_y);
    end
    
    
    % add ones before data
    data = [ones(imageNum, 1) data];
    f_test_data = [ones(f_test_size, 1) f_test_data];
    
    if useSR == 1
        % train softmax regression model
        classNum = 10;
        ita = 0.5;
        w = label + 1;
        x = data;

        [phi] = multiLogistic(w, x, classNum, ita);

        % get test error and training error
        e(c, 1) = testMulticlassLogistic(f_test_data, f_test_label, phi);
        train_e(c, 1) = testMulticlassLogistic(data, label, phi);
    end
    
    if useSRBFGS == 1
        % train softmax regression (BFGS) model
        classNum = 10;
        ita = 0.01;
        w = label + 1;
        x = data;

        [phi] = multiLogisticBFGS(w, x, classNum, ita);

        % get test error and training error
        e(c, 2) = testMulticlassLogistic(f_test_data, f_test_label, phi);
        train_e(c, 2) = testMulticlassLogistic(data, label, phi);
    end
    
    
    data = [];
    f_test_data = [];
end

save('pca_error.mat', 'e', 'train_e');