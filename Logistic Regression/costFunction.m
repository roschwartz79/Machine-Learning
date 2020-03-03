function [J , grad] = costFunction(theta, X_train, y_train)

% OUTPUTS
% J = value of cost function computer from param vector theta
% grad = partial derivative of the cost w.r.t. each param vector in theta

% INPUTS
% theta = param vector 
% X_train = Feature vector
% y_train = correct classification for each sample

m = length(y_train);

% Calculate the hypothesis which we get from our signmoid function
h_x = sigmoid(X_train * theta');

%1/m* sum(-y*log(h_x) - (1-y)*log(1-h_x))
total_error = sum(-y_train.*log10(h_x) - (1-y_train).*log10(1-h_x));

% Return J 
J = total_error/(m);

% gradient
grad = zeros(1,3);
% for each theta val -> 1/m*sum((h_x - y_train)*x_train(:,i))
for i = 1:length(theta)
     grad(1,i) = sum((h_x - y_train)'*X_train(:,i));
end

% divide by m and return the vector
grad = grad ./ length(theta);
end