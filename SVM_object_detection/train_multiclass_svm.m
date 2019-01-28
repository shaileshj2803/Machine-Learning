function [weights, biases]=train_multiclass_svm(x, y)
    weights = zeros(10, 4096);
    biases = zeros(10, 1);
    for i=1:10
        class_i = y;
        class_i(class_i==i)=55;
        class_i(class_i~=55)=-1;
        class_i(class_i==55)=1;
        [w, b, ~, ~]=SVM_Quadprog(x, class_i, 1);
        fprintf('doing it for %d', i);
        
        weights(i,:)=w;
        biases(i,:) =b;
        disp(size(weights));
        
    end
    fprintf('\nTrain Model Completed\n');
end


