% configure LibSVM
addpath('/usr/local/MATLAB/R2018b/libsvm-3.23/matlab/');

% data
train_data = load('../bigbangtheory/train.mat');
test_data = load('../bigbangtheory/test.mat');
img_path = @(id)sprintf('../bigbangtheory/%06d.jpg',id);
labels = {'living_room','kitchen','hallway', 'pennys_living_room', 'cafeteria', 'cheesecake_factory','laundry_room','comic_bookstore'};

% compute bag-of-words
HW5_BoW.main();
load('bows_data.mat','trD','tstD','trLbs');

%{
trainpred = csvread('trainpred.csv');
trainpred = trainpred(:,2)';
trainpred = normalize(trainpred,'range');
trD = [trD; trainpred];

testpred = csvread('sample_submission.csv');
testpred = testpred(:,2)';
testpred = normalize(testpred,'range');
tstD = [tstD; testpred];
%}
[~, num_train] = size(trD);
[~, num_test] = size(tstD);

% 5 Fold split
nFold = 5;
indices = cell(nFold,1);
for class_label = 1:8
    index = find(trLbs == class_label);
    num_samples = length(index);
    order=1:num_samples;
    for j=1:nFold
       indices{j}=[indices{j};index(order(((j-1)*ceil(num_samples/nFold)+1):min(end,j*ceil(num_samples/nFold))))];
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SVM with RBF kernel
% question 3.4 - part 2 - 
% default values for C and gamma
fprintf('######## question 3.4 - part 2 - ########\n');
cross_val_accuracy = [];
C_values = [1];
gamma_values = [0.001];
for C_i = 1:length(C_values)
    for gamma_i = 1:length(gamma_values)
        predicted_label={};
        accuracy={};
        prob_estimates={};
        for idx=1:nFold
            train_indices = unique(cat(1,indices{setdiff(1:nFold,idx)}));
            val_indices = unique(cat(1,indices{idx}));
            model = svmtrain(trLbs(train_indices), trD(:,train_indices)', sprintf('-s 0 -t 2 -g %f -c %f -q',gamma_values(gamma_i),C_values(gamma_i)));
            [predicted_label{idx}, accuracy{idx}, prob_estimates{idx}] = svmpredict(trLbs(val_indices), trD(:,val_indices)', model);
        end
        cross_val_accuracy(C_i,gamma_i)=sum(cellfun(@(x)x(1),accuracy).*cellfun(@(x)length(x),predicted_label))/num_train;
    end
end
cross_val_accuracy

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SVM with RBF kernel
% question 3.4 - part 3 - 
% Tune for C and gamma
fprintf('######## question 3.4 - part 3 - ########\n');
cross_val_accuracy = [];
C_values = [1000];
gamma_values = [10];
for C_i = 1:length(C_values)
    for gamma_i = 1:length(gamma_values)
        predicted_label={};
        accuracy={};
        prob_estimates={};
        for idx=1:nFold
            train_indices = unique(cat(1,indices{setdiff(1:nFold,idx)}));
            val_indices = unique(cat(1,indices{idx}));
            model = svmtrain(trLbs(train_indices), trD(:,train_indices)', sprintf('-s 0 -t 2 -g %f -c %f -q',gamma_values(gamma_i),C_values(gamma_i)));
            [predicted_label{idx}, accuracy{idx}, prob_estimates{idx}] = svmpredict(trLbs(val_indices), trD(:,val_indices)', model);
        end
        cross_val_accuracy(C_i,gamma_i)=sum(cellfun(@(x)x(1),accuracy).*cellfun(@(x)length(x),predicted_label))/num_train;
    end
end
cross_val_accuracy


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SVM with RBF kernel
% question 3.4 - part 4 - 
% SVM with exponential X2 kernel
fprintf('######## question 3.4 - part 4 - ########\n');
cross_val_accuracy=[];
C_values=[1000];
gamma_values=[50];
for C_i=1:length(C_values)
    for gamma_i=1:length(gamma_values)
        % compute exponential K-square kernel
        [trainK, testK] = cmpExpX2Kernel(trD, tstD, gamma_values(gamma_i));
        predicted_label={};
        accuracy={};
        prob_estimates={};
        for idx=1:nFold
            train_indices = unique(cat(1,indices{setdiff(1:nFold,idx)}));
            val_indices = unique(cat(1,indices{idx}));
            model = svmtrain(trLbs(train_indices), [(1:length(train_indices))',trainK(train_indices,train_indices)'], sprintf('-q -s 0 -t 4 -c %f',C_values(C_i)));
            [predicted_label{idx}, accuracy{idx}, prob_estimates{idx}] = svmpredict(trLbs(val_indices), [(1:length(val_indices))',trainK(train_indices, val_indices)'], model);
        end
        cross_val_accuracy(C_i,gamma_i)=sum(cellfun(@(x)x(1),accuracy).*cellfun(@(x)length(x),predicted_label))/num_train;
    end        
end
cross_val_accuracy


% test score
C=1000;
model = svmtrain(trLbs, [(1:num_train)',trainK'], sprintf('-q -s 0 -t 4 -c %f',C));
[ytrain_pred, accuracy_train, prob_estimates_train] = svmpredict(trLbs, [(1:num_train)',trainK'], model);
csvwrite('trainpred.csv',[train_data.imIds', ytrain_pred]);
[predicted_label, accuracy, prob_estimates] = svmpredict(ones(1,num_test)', [(1:num_test)',testK'], model);
% write predicition into a csv file
csvwrite('sample_submission.csv',[test_data.imIds', predicted_label]);
