function J = computeCost(X, y, theta)

%1/2m* sum(h(x(i)) - y(i))^2

% m = Number of training examples
% X -> the INPUT variable/features
% y -> OUTPUT variable or TARGET variable

m = length(X);

total_error = 0.0;

% Vector X multiplied by our theta paramter vector
h_x = X*theta;

%1/2m* sum(h(x(i)) - y(i))^2
total_error = sum((h_x - y).^2);  

% Return J 
J = total_error/(2*m);

end