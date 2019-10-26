
classdef SVM_Impl

  methods(Static)
    function [A,b,Aeq,beq,lb,ub,f,H] = svm_quad_prog(X,Y,C)

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
    alpha = quadprog(H,f,A,b,Aeq,beq,lb,ub,[],options);
    disp(alpha)
    end
  end
end
