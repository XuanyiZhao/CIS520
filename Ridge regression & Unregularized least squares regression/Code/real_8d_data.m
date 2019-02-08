% Read data from 10% and 100% then allocate them to training data and test data
num_train = 721;
num_test = 309;
X_train_10 = load('../Data-set-2/Train-subsets/X_train_10%.txt');
X_train_10(:,9) = ones(round(num_train * 0.1),1);
X_train_100 = load('../Data-set-2/Train-subsets/X_train_100%.txt');
X_train_100(:,9) = ones(num_train,1);
Y_train_10 = load('../Data-set-2/Train-subsets/y_train_10%.txt');
Y_train_100 = load('../Data-set-2/Train-subsets/y_train_100%.txt');
X_test = load('../Data-set-2/X_test.txt');
X_test(:,9) = ones(num_test,1);
Y_test = load('../Data-set-2/y_test.txt');
%Training on the 10% data set
[w1_b1] = (X_train_10' * X_train_10)^-1 * X_train_10' * Y_train_10;
predicted_y_train10 = X_train_10 * w1_b1;
predicted_y_test10 = X_test * w1_b1;
training_error_10 = mean_squared_error(Y_train_10, predicted_y_train10);
test_error_10 = mean_squared_error(Y_test, predicted_y_test10);
%Training on the 100% data set
[w10_b10] = (X_train_100' * X_train_100)^-1 * X_train_100' * Y_train_100;
predicted_y_train100 = X_train_100 * w10_b10;
predicted_y_test100 = X_test * w10_b10;
training_error_100 = mean_squared_error(Y_train_100, predicted_y_train100);
test_error_100 = mean_squared_error(Y_test, predicted_y_test100);
