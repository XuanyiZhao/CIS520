% One-hidden-layer NN algorithm with a logistic sigmoid activation function
function [w1, w2, b1, b2] = ...
    NN_sigmoid_model(traindata, trainlabels, w1, w2, b1, b2, num_iter, eta)

m = size(traindata, 1);

trainlabels = (trainlabels + 1) / 2;

for i = 1:num_iter
    z1 = w1 * traindata' + repmat(b1, 1, m);
    a1 = sigmoid(z1);
    f = w2 * a1 + b2;
    deri_b1 = (1 / m) * ((sigmoid(f) - trainlabels') * (derivative_sigmoid(z1) .* w2')');
    deri_b2 = (1 / m) * sum(sigmoid(f) - trainlabels');
    deri_w1 = (1 / m) * ((sigmoid(f) - trainlabels') .* (derivative_sigmoid(z1) .* w2') * traindata);
    deri_w2 = (1 / m) * (sigmoid(f) - trainlabels') * a1';
    
    w2 = w2 - eta * deri_w2;
    b2 = b2 - eta * deri_b2;
    
    w1 = w1 - eta * deri_w1;
    b1 = b1 - eta * deri_b1';
end
end