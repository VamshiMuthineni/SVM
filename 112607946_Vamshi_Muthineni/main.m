function [] = main()

quest = 2;
if quest == 2
    data = load('q2_1_data.mat');
    X = data.trD;
    Y = data.trLb;
    X_val = data.valD;
    Y_val = data.valLb;
    %disp(size(Y));

    [obj_val, accuracy, no_support_vectors] = SVM_Impl.train_svm(X,Y,0.1,X_val,Y_val);
    fprintf("\n obj val is %f , accuracy is %f, no of support vectors are %d", obj_val, accuracy, no_support_vectors);
    
    [obj_val, accuracy, no_support_vectors] = SVM_Impl.train_svm(X,Y,10,X_val,Y_val);
    fprintf("\n obj val is %f , accuracy is %f, no of support vectors are %d", obj_val, accuracy, no_support_vectors);
    
    %disp(alpha);
    %disp(size(X));
%     [w,b] = SVM_Impl.get_primal_variables(X,Y,0.1);
%     SVM_Impl.train_svm(X,Y,0.1,X_val,Y_val);
% 
%     [w,b] = SVM_Impl.get_primal_variables(X,Y,10);
%     SVM_Impl.train_svm(X,Y,10,X_val,Y_val);

    X_T = csvread('/Users/vamshimuthineni/Vamshi/ML/hw4/cse512hw4/Train_Features.csv',0,1);
    X_T=X_T';
    X_V = csvread('/Users/vamshimuthineni/Vamshi/ML/hw4/cse512hw4/Val_Features.csv',0,1);
    X_V=X_V';
    X_Test = csvread('/Users/vamshimuthineni/Vamshi/ML/hw4/cse512hw4/Test_Features.csv',0,1);
    X_Test=X_Test';
    Y_T = csvread('/Users/vamshimuthineni/Vamshi/ML/hw4/cse512hw4/Train_Labels.csv',1,1);
    Y_V = csvread('/Users/vamshimuthineni/Vamshi/ML/hw4/cse512hw4/Val_Labels.csv',1,1);
    fulldata = horzcat(X_T,X_V);
    fulllabels = vertcat(Y_T,Y_V);

%     disp(size(X_T));
%     disp(size(Y_T));
%     disp(size(X_V));
%     disp(size(Y_V));
%     disp(size(fulldata));
%     disp(size(fulllabels));
    
%     X_T_2 = fulldata.*fulldata;
%     X_T_3 = fulldata.*fulldata;
%     fulldata = horzcat(fulldata,X_T_2);
%     fulldata = horzcat(fulldata,X_T_2);
    
    
    X_T = HW4_Utils.l2Norm(double(fulldata));
    X_V = HW4_Utils.l2Norm(double(X_V));
    X_Test = HW4_Utils.l2Norm(double(X_Test));    
    
    Y_T = fulllabels;
    
%     values = [1,2,0.1,10,100,200,1000];
    values = [1.25];
    for k=1:size(values,2)
        [Y_pred] = SVM_Impl.predictMultiClasses(X_T,Y_T, X_T, values(k));
        [accuracy] = SVM_Impl.calculateAccuracy(Y_T,Y_pred);
        fprintf("train data accuracy %f at C: %f\n", accuracy, values(k));

        [Y_pred] = SVM_Impl.predictMultiClasses(X_T,Y_T, X_V,values(k));
        [accuracy] = SVM_Impl.calculateAccuracy(Y_V,Y_pred);
        fprintf("val data accuracy %f at C: %f\n", accuracy, values(k));

        [Y_pred] = SVM_Impl.predictMultiClasses(X_T,Y_T, X_Test, values(k));

        Y_pred = Y_pred';
%         disp(Y_pred);
    end
    
end

if quest == 4
    SVM_Obj_Detec.computeAP();
    

if quest == 3
%     SVM_Obj_Detec.computeAP();
    SVM_Obj_Detec.hardNegMinAlgo2();
end
end

