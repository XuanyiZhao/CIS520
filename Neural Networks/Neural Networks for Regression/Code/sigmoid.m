function sigmoid_f = sigmoid(f)
% logistic sigmoid activation function
sigmoid_f = 1 ./ (1 + exp(-f));
end