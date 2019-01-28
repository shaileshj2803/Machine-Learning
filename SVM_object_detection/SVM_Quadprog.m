function [weights, bias, objective, alpha] = SVM_Quadprog(x, y, C)
    uniq = unique(y);
    disp(uniq);
    [d, n]=size(x);
    
    A=[];
    b=[];
    Aeq=y';
    beq=0;
    
    lb=zeros(n,1);
    ub=C*ones(n,1);
    
    k_term = x'*x;
    H = diag(y)*k_term*diag(y);
    f=-1*ones(n,1);

    
    [alpha, obj_val] = quadprog(H,f,A,b,Aeq,beq,lb,ub,[]);
    
    objective=-1*obj_val;
    weights = x*(y.*alpha);
    
    [~, idx]=max(min(alpha,C-alpha));
    bias = y(idx)- k_term(idx,:)*diag(y)*alpha;

end