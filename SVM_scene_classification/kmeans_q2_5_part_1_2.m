X = load('../digit/digit.txt');
Y = load('../digit/labels.txt');
X = X';
[d,n] = size(X);

for k=[2,4,6]
    % kmeans
    center_ids=1:k;
    [center_points, assignment, iteration] = kmeans(X, k, 20, center_ids);

    % within-group sum-of-squares
    ss=nan(1,k);
    for i=1:k
        idx=find(assignment==i);
        one=X(:,idx)';
        two=center_points(:,i)';
        ss(i)= sum(pdist2(one, two, 'euclidean').^2);
    end
    sum_ss=sum(ss);

    % pair-counting measure
    same_cluster=(pdist2(assignment,assignment,@(x,y) x-y))==0;
    same_class=(pdist2(Y,Y,@(x,y) x-y))==0;
    
    same_cluster_same_class = sum(sum(same_class & same_cluster))-n;
    diff_cluster_same_class = sum(sum(same_class & ~same_cluster));
    same_cluster_diff_class = sum(sum(~same_class & same_cluster))-n;
    diff_cluster_diff_class = sum(sum(~same_class & ~same_cluster));
    
    p1 = same_cluster_same_class / (same_cluster_same_class + diff_cluster_same_class);
    p2 = diff_cluster_diff_class / (diff_cluster_diff_class + same_cluster_diff_class);
    p3 = (p1+p2)/2;

    % report results
    fprintf('\nk = %d:\n',k);
    fprintf('Kmeans converges at iter = %d;\n',iteration);
    fprintf('The total within sum of squares is %.2f;\n',sum_ss);
    fprintf('p1 = %.2f%%, p2 = %.2f%%, p3 = %.2f%%.\n\n',p1*100, p2*100, p3*100);
end
