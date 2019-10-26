
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
    
    function [w,b] = get_primal_variables(X,Y,C)
        [A,b,Aeq,beq,lb,ub,f,H,alpha,obj_val] = SVM_Impl.svm_quad_prog(X,Y,C);
        [m,n] = size(X);

         w = zeros(m,1);
         for j=1:n
             w = w + ((alpha(j)*Y(j))*X(:,j));
         end
         
         b = 0;
         j = 0;
         for i=1:n
             if(alpha(i)>0 && alpha(i)<C)
                 b = b + (Y(i)- w'*(X(:,i)));
                 j = j+1;
             end
         end
         b = b/j;
         fprintf("\nObjective value from quadprog %f \n", obj_val);
         SVM_Impl.objective_value(alpha,X,Y);
%          disp(b)
         
%          disp(size(w))
    end
    
    function [] = train_svm(X,Y,C,X_val,Y_val)
        [w,b] = SVM_Impl.get_primal_variables(X,Y,C);
        
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
        fprintf("accuracy %f", correct_preds/m);
        SVM_Impl.displayConfusion(Y_val,Y_pred);
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
