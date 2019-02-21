function deriv_sigmoid_f = derivative_sigmoid(f)
% The 1st-order derivative of logistic sigmoid activation function
deriv_sigmoid_f = exp(-f) ./ ((1 + exp(-f)) .^ 2);
end