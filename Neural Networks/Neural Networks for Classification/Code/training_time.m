% Load training data and test data
X_train_full = load('../Spam-Dataset/X_train.txt');
y_train_full = load('../Spam-Dataset/y_train.txt');
X_test_full = load('../Spam-Dataset/X_test.txt');
y_test_full = load('../Spam-Dataset/y_test.txt');

init_b1 = load('../Spam-Dataset/InitParams/relu/15/b1.txt');
init_b2 = load('../Spam-Dataset/InitParams/relu/15/b2.txt');
init_w1 = load('../Spam-Dataset/InitParams/relu/15/W1.txt');
init_w2 = load('../Spam-Dataset/InitParams/relu/15/W2.txt');

tic
[w1_2,w2_2, b1_2, b2_2] = NN_ReLU_model(X_train_full,y_train_full, ...
    init_w1, init_w2, init_b1, init_b2, 8000, 0.1);
y_pred_train = NN_ReLU_classifier(X_train_full, w1_2, w2_2, b1_2, b2_2);
training_error = classification_error(y_pred_train, y_train_full);

y_pred_test = NN_ReLU_classifier(X_test_full, w1_2, w2_2, b1_2, b2_2);
test_error = classification_error(y_pred_test, y_test_full);
toc