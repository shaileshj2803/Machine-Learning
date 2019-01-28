function [trainK, testK] = cmpExpX2Kernel(trD, tstD, gamma)

[~, num_train] = size(trD);
[~, num_test] = size(tstD);

trainK = zeros(num_train, num_train);
for i=1:num_train
    for j=(i+1):num_train
        trainK(i,j)=sum((trD(:,i) - trD(:,j)).^2./(trD(:,i) + trD(:,j) + eps));
    end
end
if ~exist('gamma','var') 
    gamma = mean(trainK(trainK~=0)); 
end
trainK = trainK + trainK';
trainK = exp(-trainK/gamma);

testK = zeros(num_train,num_test);
for i=1:num_train
    for j=1:num_test
        testK(i,j) = sum((trD(:,i) - tstD(:,j)).^2./(trD(:,i) + tstD(:,j) + eps));
    end
end
testK = exp(-testK/gamma);
end