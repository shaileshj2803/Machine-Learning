load q2_2_data.mat;

trD = double(trD);
trLb = double(trLb);
valD = double(valD);
valLb = double(valLb);
tstD = double(tstD);

%perm = randperm(size(trD,2));
%valD = valD(:,perm);
%valD = valLb(:, perm);
%trD = [trD, valD(:,[1:1000])];
%trLb = [trLb; valLb(:,[1000:end])];

%train
[weights, biases]=train_multiclass_svm(trD, trLb);

% predict on validation
[d,n] = size(valD);
pred_prob = zeros(10, n);
for i=1:10
   wi=weights(i,:);
   bi=biases(i,:);
   pred_prob(i,:) = wi*valD + bi;
end
[~, pred]=max(pred_prob);
disp(size(pred));

% accuracy for validation data
correct_predicted = 0;
for i = 1:length(valLb)
    if pred(i)==valLb(i)
        correct_predicted=correct_predicted+1;
    end
end
accuracy_value = correct_predicted*100/length(valLb);
save('weights_baseline.mat', 'weights', 'biases');

% predict for test data
[d,n] = size(tstD);
pred_prob = zeros(10, n);
for i=1:10
   wi=weights(i,:);
   bi=biases(i,:);
   pred_prob(i,:) = wi*tstD + bi;
end
[~, pred]=max(pred_prob);
disp(size(pred));
csvwrite('SampleSubmission.csv',pred');
