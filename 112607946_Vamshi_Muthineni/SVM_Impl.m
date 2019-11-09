
classdef SVM_Impl

  methods(Static)
    function [A,b,Aeq,beq,lb,ub,f,H,alpha,obj_val] = svm_quad_prog(X,Y,C)

    [~,n] = size(X);
    A = [];
    b = [];
    Aeq = Y';
    beq = 0;
    lb = zeros(1,n);
    ub = ones(1,n) * C;
    f = ones(n,1) * -1;
    H = (Y*Y').*(X'*X);
    options = optimset('display','off');
    [alpha,obj_val] = quadprog(H,f,A,b,Aeq,beq,lb,ub,[],options);
%     disp(alpha)
    end
    
    function str=log2str(a)
        if a
            str='true';
        else
            str='false';
        end
    end
    
    function [w,b,alpha,obj_val, no_support_vectors] = get_primal_variables(X,Y,C)
        [A,b,Aeq,beq,lb,ub,f,H,alpha,obj_val] = SVM_Impl.svm_quad_prog(X,Y,C);
        [m,n] = size(X);

         w = zeros(m,1);
         for j=1:n
             w = w + ((alpha(j)*Y(j))*X(:,j));
         end
         
         b = 0;
         j = 0;
         no_support_vectors = 0;
         for i=1:n
             if alpha(i)>0.01
                no_support_vectors = no_support_vectors + 1;
             end
             areEssentiallyEqual = SVM_Impl.log2str(abs(single(alpha(i))-single(C)) < single(eps(C)));
             if(areEssentiallyEqual)
                 b = b + (Y(i)- w'*(X(:,i)));
                 j = j+1;
             end
         end
         b = b/j;
    end
    
    function [obj_val, accuracy, no_support_vectors] = train_svm(X,Y,C,X_val,Y_val)
        [w,b,alpha,obj_val,no_support_vectors] = SVM_Impl.get_primal_variables(X,Y,C);
        
        [m,n] = size(Y_val);
        Y_pred = zeros(m,n);
        correct_preds = 0;
        
        for i=1:m
            Y_pred(i) = w'*(X_val(:,i)) + b;
            if Y_pred(i)>=0
                Y_pred(i) = 1;
            else
                Y_pred(i) = -1;
            end
            if Y_pred(i) == Y_val(i)
                correct_preds = correct_preds + 1;
            end
        end
        accuracy = correct_preds/m;
        SVM_Impl.displayConfusion(Y_val,Y_pred);
        fprintf("\n")
    end
    
    function [w_all,b_all] = multiClassPredictor(X,Y,C)
        [m,n] = size(X);
        w_all = [];
        b_all = [];
        for i=1:4
            Y_temp = zeros(n, 1);
            Y_temp(Y==i) = 1;
            Y_temp(Y~=i) = -1;
            [w,b,alpha,obj_val] = SVM_Impl.get_primal_variables(X,Y_temp,C);
            w_all = [w_all;w'];
            b_all = [b_all;b];
        end
    end
    
    function [Y_pred] = predictMultiClasses(X,Y,X_to_predict,C)
        [w_all,b_all] = SVM_Impl.multiClassPredictor(X,Y,C);
        [m,n] = size(X_to_predict);
        Y_pred_all = zeros(4, n);
        Y_pred = zeros(n, 1);
        
        for j=1:4
            for i=1:n
                temp = (w_all(j,:)*(X_to_predict(:,i))) + b_all(j);
                Y_pred(i) = temp;
            end
            Y_pred_all(j,:) = Y_pred';
        end
        
        [maxes, Y_pred] = max(Y_pred_all);
%         disp(Y_pred);      
    end
    
    function [accuracy] = calculateAccuracy(Y,Y_pred)
        correct_preds = 0;
        [n,m] = size(Y);
        for i=1:n
            if Y_pred(i) == Y(i)
                correct_preds = correct_preds + 1;
            end
        end
        accuracy = correct_preds/n;
    end
    
    function [] = displayConfusion(Y,Y_pred)
        t_t = 0;
        t_f = 0;
        f_t = 0;
        f_f = 0;
        [m,n] = size(Y);
        for i=1:m
            if Y_pred(i)==Y(i) && Y_pred(i) == 1
                t_t = t_t + 1;
            end
            if Y_pred(i)~=Y(i) && Y_pred(i) == -1
                t_f = t_f + 1;
            end
            if Y_pred(i)~=Y(i) && Y_pred(i) == 1
                f_t = f_t + 1;
            end
            if Y_pred(i)==Y(i) && Y_pred(i) == -1
                f_f = f_f + 1;
            end
        end
        fprintf("\n")
        fprintf("%d   %d \n", t_t, t_f);
        fprintf("%d   %d", f_t, f_f);
    end
    
    function [] = objective_value(alpha,X,Y)
        [m,n] = size(Y);
        alpha_sum = 0;
        for i=1:m
            alpha_sum = alpha_sum + alpha(i);
        end
        other_sum = 0;
        for i=1:m
            for j=1:m
                other_sum = other_sum + alpha(i)*alpha(j)*Y(i)*Y(j)*(X(i)'*X(j));
            end
        end
        obj_val = alpha_sum - (0.5)*(other_sum);
        fprintf("\n Objective Value: %f \n",obj_val);
    end
    
  end
end
