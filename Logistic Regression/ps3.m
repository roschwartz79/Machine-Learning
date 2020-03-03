% Problem Set 2, ECE 1395
% Written by Rob Schwartz
clc
close all

% Separate the file by spaces into a matrix
fileID = fopen('input/hw3_data1.txt');
rawData = textscan(fileID,'%f %f %f','Delimiter',',');
fclose(fileID);

% Set up the data 
X_raw1 = cell2mat(rawData(:,1));
X_raw2 = cell2mat(rawData(:,2));
X = ones(length(X_raw1), 3);
X(:,2) = X_raw1;
X(:,3) = X_raw2;
y = cell2mat(rawData(:,3));


j = 1;
k = 1;
for i = 1:length(y)
    
    if y(i,1) == 0
        temp(j,1) = X(i,2);
        temp(j,2) = X(i,3);
        j = j+1;
    end
    if y(i) == 1
        temp1(k,1) = X(i,2);
        temp1(k,2) = X(i,3);
        k = k + 1;
    end
end

figure(1);
scatter(temp(:,1),temp(:,2),'y','filled');
hold on;
scatter(temp1(:,1),temp1(:,2),'k','+');
legend({'Not Admitted','Admitted'});
xlabel("Exam 1 score");
ylabel("Exam 2 score");
hold off;

% Part c: sigmoid function
z = [-10:10];
gz = sigmoid(z);

figure(2);
plot(z, gz);
title("z vs gz");
xlabel("z");
ylabel("gz");

% Part d: Cost Function
theta = [0, 0, 0];

[J, grad] = costFunction(theta, X, y);

% Part e: logistic regression
fprintf("Running logistic regression");
options = optimset('GradObj', 'on', 'MaxIter', 400);

[theta,cost] = fminunc(@(theta)(costFunction(theta, X, y)), theta, options);


% Part f: 
x1 = [30:100];
x2 = -1* theta(2)/theta(3)*x1 - theta(1)/theta(2);

figure(3);
scatter(temp(:,1),temp(:,2),'y','filled');
hold on;
scatter(temp1(:,1),temp1(:,2),'k','+');
plot(x1,x2);
title("Plot of exam scores with the decision boundary");
legend({'Not Admitted','Admitted','Decision Boundary'});
xlabel("Exam 1 score");
ylabel("Exam 2 score");
hold off;

% Part g:
test1score = 45;
test2score = 85;

admission_prob = theta(1) + theta(2)*test1score + theta(3)*test2score;
admission_prob = sigmoid(admission_prob);
fprintf("The admission probability is: %s", num2str(admission_prob));



        
        
