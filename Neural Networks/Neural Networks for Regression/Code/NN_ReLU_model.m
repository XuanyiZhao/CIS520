% One-hidden-layer NN algorithm with a rectified linear unit function
function [w1, w2, b1, b2] = ...
    NN_ReLU_model(traindata, trainlabels, w1, w2, b1, b2, num_iter, eta)

m = size(traindata, 1);

for i = 1:num_iter
    z1 = w1 * traindata' + repmat(b1, 1, m);
    a1 = ReLU(z1);
    f = w2 * a1 + b2;
    deri_b1 = (2 / m) * ((f- trainlabels') * (derivative_ReLU(z1) .* w2')');
    deri_b2 = (2 / m) * sum(f - trainlabels');
    deri_w1 = (2 / m) * ((f - trainlabels') .* (derivative_ReLU(z1) .* w2') * traindata);
    deri_w2 = (2 / m) * (f - trainlabels') * a1';
    
    w2 = w2 - eta * deri_w2;
    b2 = b2 - eta * deri_b2;
    
    w1 = w1 - eta * deri_w1;
    b1 = b1 - eta * deri_b1';
end
end