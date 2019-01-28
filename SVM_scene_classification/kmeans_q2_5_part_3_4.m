X = load('../digit/digit.txt');
Y = load('../digit/labels.txt');
X = X';
[d,n] = size(X);

sum_ss=[];
p1=[];
p2=[];
for k=1:10
    for epoch = 1:10
        % kmeans
        [center_points, assignment, iteration] = kmeans(X, k, 20, []);

        % within-group sum-of-squares
        ss = nan(1,k);
        for i=1:k
            idx = find(assignment==i);
            one = X(:,idx)';
            two = center_points(:,i)';
            ss(i) = sum(pdist2(one, two, 'euclidean').^2);
        end
        sum_ss(k,epoch)=sum(ss);

        % pair-counting measure
        same_cluster=(pdist2(assignment,assignment,@(x,y) x-y))==0;
        same_class=(pdist2(Y,Y,@(x,y) x-y))==0;
        
        same_cluster_same_class = sum(sum(same_class & same_cluster))-n;
        diff_cluster_same_class = sum(sum(same_class & ~same_cluster));
        same_cluster_diff_class = sum(sum(~same_class & same_cluster))-n;
        diff_cluster_diff_class = sum(sum(~same_class & ~same_cluster));
        
        p1(k,epoch) = same_cluster_same_class / (same_cluster_same_class + diff_cluster_same_class);
        p2(k,epoch) = diff_cluster_diff_class / (diff_cluster_diff_class + same_cluster_diff_class);
    end
end
p3=(p1+p2)/2;

figure(1);
plot(mean(sum_ss,2),'*-');
xlabel('K'); legend('SS_{sum}');

figure(2);
plot(squeeze(mean(cat(3,p1,p2,p3),2)),'*-')
xlabel('K'); legend('p_1','p_2','p_3');
