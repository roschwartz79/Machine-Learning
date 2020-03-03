function y_predict = weightedKNN(X_train, y_train, X_test, sigma)

% INPUTS
% X_train -> m x n feature matrix where m is the training instances and n
% is the number of features
% y_train -> m x 1 label vector 
% sigma ->scalar denoting the bandwidth of the gaussian weighing function

% OUTPUTS
% y_predict -> d x 1 vector that has the predicted labels for testing
y_predict = zeros(25,1);

% weight on each sample
% Wi = exp((abs(x-xi)^2/(sigma^2))

%weight = exp((-1*abs(pdist2(X_test(1,:),X_train(1,:)))^2)/(sigma^2))

% Loop through each point in the test matrix
for i = 1:size(X_test)
    weight1 = 0;
    weight2 = 0;
    weight3 = 0;
    votes1 = 0;
    votes2 = 0;
    votes3 = 0;
   % Loop through each point in the training matrix, assign a weight
  for j = 1:size(X_train)
      weight_at_point = exp((-1*abs(pdist2(X_test(i,:),X_train(j,:)))^2)/(sigma^2));
      
      % Add the weights up
      if y_train(j) == 1
         weight1 = weight1 + weight_at_point; 
         votes1 = votes1 + 1;
      end
      if y_train(j) == 2
         weight2 = weight2 + weight_at_point;
         votes2 = votes2 + 1;
      end
      if y_train(j) == 3
         weight3 = weight3 + weight_at_point;    
         votes3 = votes3 + 1;
      end
  end

  %Pick what the prediction is based on the weights 
  if weight1 > weight2 && weight1 > weight3
      y_predict(i) = 1; 
  end
  if weight2 > weight1 && weight2 > weight3
      y_predict(i) = 2; 
  end
  if weight3 > weight1 && weight3 > weight2
      y_predict(i) = 3; 
  end

end


end