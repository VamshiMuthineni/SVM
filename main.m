function [] = main()
data = load('q2_1_data.mat');
X = data.trD;
Y = data.trLb;
disp(size(Y));

alpha = SVM_Impl.svm_quad_prog(X,Y,0.1);
disp(alpha);
end
