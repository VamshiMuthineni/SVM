function [] = main()
data = load('q2_1_data.mat');
X = data.trD;
Y = data.trLb;
X_val = data.valD;
Y_val = data.valLb;
%disp(size(Y));

alpha = SVM_Impl.svm_quad_prog(X,Y,0.1);
%disp(alpha);
%disp(size(X));
[w,b] = SVM_Impl.get_primal_variables(X,Y,0.1);
SVM_Impl.train_svm(X,Y,0.1,X_val,Y_val);

[w,b] = SVM_Impl.get_primal_variables(X,Y,10);
SVM_Impl.train_svm(X,Y,10,X_val,Y_val);

end

