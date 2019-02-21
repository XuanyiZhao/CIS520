% Load training data and test data
X_train_full = load('../Data-set-1/X_train.txt');
y_train_full = load('../Data-set-1/y_train.txt');
X_test_full = load('../Data-set-1/X_test.txt');
y_test_full = load('../Data-set-1/y_test.txt');

init_b1 = load('../Data-set-1/InitParams/b1.txt');
init_b2 = load('../Data-set-1/InitParams/b2.txt');
init_w1 = load('../Data-set-1/InitParams/W1.txt');
init_w2 = load('../Data-set-1/InitParams/W2.txt');
tic
[w1_2, w2_2, b1_2, b2_2] = NN_ReLU_model(X_train_full,y_train_full, ...
    init_w1, init_w2, init_b1, init_b2, 20000, 0.1);
y_pred_train = NN_ReLU_classifier(X_train_full, w1_2, w2_2, b1_2, b2_2);
training_error = mean_squared_error(y_pred_train, y_train_full);

y_pred_test = NN_ReLU_classifier(X_test_full, w1_2, w2_2, b1_2, b2_2);
test_error = mean_squared_error(y_pred_test, y_test_full);
toc

figure
hold on
y_pred_plot = NN_ReLU_classifier(linspace(0,1)', w1_2, w2_2, b1_2, b2_2);
plot(X_test_full,y_test_full,'or',linspace(0,1),y_pred_plot,'b','linewidth',1)
xlabel('Input instances');
ylabel('Predicted value/True value');
legend('True value','Predicted value');
hold off