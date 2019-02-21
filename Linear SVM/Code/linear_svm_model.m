function [w, b] = linear_svm_model(traindata, trainlabels, C)
% X = quadprog(H,f,A,b,Aeq,beq,LB,UB)
H = (trainlabels * trainlabels') .* (traindata * traindata');
m = length(trainlabels);
f = ones(m, 1);
Aeq = trainlabels';
beq = 0;
LB = zeros(m, 1);
UB = ones(m, 1) * C;
a = quadprog(H, -f, [], [], Aeq, beq, LB, UB);
a(a < 0.000001 * C) = 0;
a(a > (C - C * 0.000001)) = C;
w = a' * (trainlabels .* traindata);
sv1 = find(0.000001 * C < a < (C - 0.000001 * C));
diff = trainlabels - traindata * w';
b = sum(diff(sv1)) / length(sv1);
end