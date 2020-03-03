% Problem set 5
% Written by Rob Schwartz

% Set up the images
moveImages;

% Read in the images from training directory
T = zeros(10304,320);
temp = zeros(10304, 1);
testingImages = zeros(10304,80);

% What index in t we are at
tIndex = 1;

% Loop through all the training images 
for i = 1:40
    for j = 1:8
        readString = "input/train/s" + i + j + ".pgm";
        A = imread(char(readString));
        temp = A(:);
        T(:,tIndex) = temp;
        tIndex = tIndex + 1; 
    end
end

tIndex = 1;
% Loop through all the testing images 
for i = 1:40
    for j = 1:2
        readString = "input/test/s" + i + "/image" + j + ".pgm";
        A = imread(char(readString));
        temp = A(:);
        testingImages(:,tIndex) = temp;
        tIndex = tIndex + 1; 
    end
end

fprintf("finished creating T\n");

% Display the grey level image
figure(1);
imshow(T,[]);

%Compute the average face vector m
meanFaceVector = mean(T')';
image2display = reshape(meanFaceVector, 112,92);
figure(2);
imshow(image2display, []);

% Find the centered data matrix 
centeredData = bsxfun(@minus, T, meanFaceVector);

% covariance matrix C = AA'
C = centeredData * centeredData';

% display the image
figure(3);
imshow(C,[]);

fprintf("Getting eigenvalues\n");
% get the eigenvalues 
vals = eig(C);

fprintf("Descending");
eigenVals = sort(vals,'descend');

fprintf("finished getting eigenvals\n");

kVector = zeros(1,320);
for k = 1:320
    sumVals = 0;
    for i = 1:k
         sumVals = sumVals + eigenVals(i);
    end
    kVector(k) = sumVals;
end
fprintf("kVector Done");

vVector = zeros(1,320);

for i = 1:320
   vVector(i) = kVector(i)/kVector(320); 
end

fprintf("vVector done");

fprintf("plotting\n");

figure(4);
plot(vVector);
title("k vs v(k)");
xlabel("k");
ylabel("v(k)");
fprintf("The number of eigenvectors to keep .95 of the data is k = 163\n");

% Save the 163 dominant eigenvectors
[U,~] = eigs(C,163);

figure(6);
fprintf("First 8 eigenfaces");
for i = 1:8
    subplot(4,2,i);
    imshow(reshape(U(:,i),112, 92),[]);
end

% 2 - Feature extraction for face recognition
% project all images in training folder into reduced eigenspace U

% w = U'*(I-m)

W_training = U' * (T - meanFaceVector);
W_training = W_training';

% Get W_testing 
W_testing = U' * (testingImages - meanFaceVector);
W_testing = W_testing';

% 3 - Face Recognition
% Train KNN for 1, 3, 5, 7, 9 and 11 
trained_model_vector = {6,1};
index = 1;

labelVector = zeros(320,1);
label = 1;
for i = 1:8:320
   labelVector(i:i+7,1) = label; 
   label = label + 1;
end

for k = 1:2:11
   % each fold
   trained_model_vector(index,1) = {fitcknn(W_training,labelVector,'NumNeighbors',k)};
   index = index + 1;
end

% 6 classifiers
predictionVector = zeros(6,80);
% For each value of k
for i = 1:6
    % For each picture make a prediction
    for j = 1:80
        predictionVector(i,j) = predict(trained_model_vector{i,1},W_testing(j,:));
    end
end

percentageVector = zeros(6,1);
% calculate accuracy of the knn classifier
for i = 1:6
    misclassified = 0;
    % go every 2 bc 2 pics for each person
    for j = 1:2:80 
        % check pic 1
        if predictionVector(i,j) ~= j
            misclassified = misclassified + 1;
        end
        % check pic 2
        if predictionVector(i,j + 1) ~= j
            misclassified = misclassified + 1;
        end
        percentageVector(i) = misclassified/80;
    end
end

% Make the table 
fprintf("\n\nTable for KNN classifier Accuracy:\n");
fprintf("1         3          5          7          9          11\n");
fprintf("%f  %f   %f   %f   %f   %f",percentageVector(1:6));
fprintf("\n\n");

% Train an SVM classifier
% Use fitcsvm and predict 
% linear, 3rd order, then gaussian rbf kernels

%%%%%%%----START OF LINEAR SVM----%%%%%%

% Set all labels to class 2, set each set of 8 to 1 when it is their
% classifier to be trained
startingIndex = 1;
svm_model_cells = cell(3,40);
prediction_cells = cell(3,40);

% Train the linear classifiers
index1 = 1;
for i = 1:8:320
   svmLabels(1:320) = 2;
   svmLabels(i:i+7) = 1;
   svm_model_cells(1,index1) = {fitcsvm(W_training, svmLabels)};
   index1 = index1 + 1;
end

%SVMModel = fitcsvm(W_training, test);
%[label, score] = predict(SVMModel, W_testing(20,:));

% rows = test picture, columns = SVM classifiers
predictionSVMVector = zeros(80,40);
% get the predictions for the 80 test files
for i = 1:80
    % predict with each SVM model
    for j = 1:40
        % predict the ith pic with the jth svm classifier
        [label1,score1] = predict(svm_model_cells{1,j},W_testing(i,:));
        predictionSVMVector(i,j) = score1(1);
    end
end

% Determine what picture the test pic is
picArray = zeros(1,80);
for i = 1:80
   picArray(i) = maxScore(predictionSVMVector(i,:)); 
end
if picArray(1,1) ~= 1.0
    fprintf("Why");
end
% Calculate accuracy
index1 = 1;
index2 = 2;
misclassified = 0;
for i = 1:40
    if picArray(index1) ~= i
       misclassified = misclassified + 1;
    end
    if picArray(index2) ~= i
       misclassified = misclassified + 1; 
    end
    index1 = index1 + 2;
    index2 = index2 + 2;
end

accuracyLinearSVM = (80-misclassified)/80;
fprintf("Accuracy of the Linear SVM Model is %f",accuracyLinearSVM);

%%%%%%%----END OF LINEAR SVM----%%%%%%

%%%%%%%----START OF 3RD ORDER POLYNOMIAL SVM----%%%%%%
% Set all labels to class 2, set each set of 8 to 1 when it is their
% classifier to be trained
startingIndex3 = 1;
svm_model_cells3 = cell(1,40);

% Train the 3rd order classifiers
index1 = 1;
for i = 1:8:320
   svmLabels(1:320) = 2;
   svmLabels(i:i+7) = 1;
   svm_model_cells3(1,index1) = {fitcsvm(W_training, svmLabels','Standardize',true, 'KernelFunction', 'polynomial')};
   index1 = index1 + 1;
end

%SVMModel = fitcsvm(W_training, test);
%[label, score] = predict(svm_model_cells3{1,1}, W_testing(20,:));

% rows = test picture, columns = SVM classifiers
predictionSVMVector3 = zeros(80,40);
% get the predictions for the 80 test files
for i = 1:80
    % predict with each SVM model
    for j = 1:40
        % predict the ith pic with the jth svm classifier
        [label1,score1] = predict(svm_model_cells3{1,j},W_testing(i,:));
        predictionSVMVector3(i,j) = score1(1);
    end
end

% Determine what picture the test pic is
picArray3 = zeros(1,80);
for i = 1:80
   picArray3(i) = maxScore(predictionSVMVector3(i,:)); 
end

% Calculate accuracy
index1 = 1;
index2 = 2;
misclassified = 0;
for i = 1:40
    if picArray3(index1) ~= i
       misclassified = misclassified + 1;
    end
    if picArray3(index2) ~= i
       misclassified = misclassified + 1; 
    end
    index1 = index1 + 2;
    index2 = index2 + 2;
end

accuracy3rdSVM = (80-misclassified)/80;
fprintf("\n\nAccuracy of the 3rd Order Polynomial SVM Model is %f",accuracy3rdSVM);

%%%%%%%----END OF 3RD ORDER POLYNOMIAL SVM----%%%%%%


%%%%%%%----START OF GAUSSIAN SVM----%%%%%%
% Set all labels to class 2, set each set of 8 to 1 when it is their
% classifier to be trained
startingIndex3 = 1;
svm_model_cellsG = cell(1,40);

% Train the 3rd order classifiers
index1 = 1;
for i = 1:8:320
   svmLabels(1:320) = 2;
   svmLabels(i:i+7) = 1;
   svm_model_cellsG(1,index1) = {fitcsvm(W_training, svmLabels', 'KernelFunction', 'gaussian')};
   index1 = index1 + 1;
end

%SVMModel = fitcsvm(W_training, test);
[label, score] = predict(svm_model_cellsG{1,1}, W_testing(20,:));

% rows = test picture, columns = SVM classifiers
predictionSVMVectorG = zeros(80,40);
% get the predictions for the 80 test files
for i = 1:80
    % predict with each SVM model
    for j = 1:40
        % predict the ith pic with the jth svm classifier
        [label1,score1] = predict(svm_model_cellsG{1,j},W_testing(i,:));
        predictionSVMVectorG(i,j) = score1(1);
    end
end

% Determine what picture the test pic is
picArrayG = zeros(1,80);
for i = 1:80
   picArrayG(i) = maxScore(predictionSVMVectorG(i,:)); 
end

% Calculate accuracy
index1 = 1;
index2 = 2;
misclassified = 0;
for i = 1:40
    if picArrayG(index1) ~= i
       misclassified = misclassified + 1;
    end
    if picArrayG(index2) ~= i
       misclassified = misclassified + 1; 
    end
    index1 = index1 + 2;
    index2 = index2 + 2;
end

accuracyGSVM = (80-misclassified)/80;
fprintf("\n\nAccuracy of the Gaussian SVM Model is %f",accuracyGSVM);

%%%%%%%----END OF GAUSSIAN SVM----%%%%%%


