load q2_1_data.mat;

C_values=[0.1,10];

for C=C_values
    [weights, bias, objective_function, alpha] = SVM_Quadprog(trD, trLb, C);

    val_prediction = sign(valD'*weights + bias);

    confusion_matrix = confusionmat(val_prediction, valLb);

    val_accuracy = compute_accuracy(valLb, val_prediction); 

    support_vectors = compute_sv(weights, bias, valD);
    
    fprintf('C : %.2f \n', C);
    fprintf('Accuracy : %.3f \n', val_accuracy);
    fprintf('Objective : %.3f \n', objective_function);
    fprintf('Num of support vectors : %.2f \n', support_vectors);
    fprintf('Confusion Matrix : \n');
    disp(confusion_matrix);
end

function accuracy_value = compute_accuracy(actual, predicted)
    correct_predicted = 0;
    for i = 1:length(actual)
        if predicted(i)==actual(i)
            correct_predicted=correct_predicted+1;
        end
    end
    accuracy_value = correct_predicted*100/length(actual);
end

function support_vectors = compute_sv(weights, bias, x)
    prediction = x' * weights + bias;
    support = prediction <= 1 & prediction >= -1;
    support_vectors = nnz(support);
end
