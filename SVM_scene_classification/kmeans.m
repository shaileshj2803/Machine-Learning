function [center_points, assignment, iteration]=kmeans(X, k, max_iteration, center_ids)

[d,n]=size(X);
if isempty(center_ids)
    center_ids = randperm(n,k);
end
center_points = X(:,center_ids);

% initialize assignment
assignment = nan(n,1);

% repeat until convergence or iteration limit
for iteration = 1:max_iteration
    % re-assigment
    distance = pdist2(X',center_points','euclidean');
    [~,assignment_1]=min(distance,[],2);
    difference = sum(assignment_1~=assignment);
    assignment = assignment_1;
    fprintf('Iter %03d: Assignment Difference = %d\n',iteration,difference);
    if difference==0
        break;
    end
    % re-center
    for i=1:k
        idx = find(assignment==i);
        center_points(:,i)=mean(X(:,idx),2);
    end
end

         