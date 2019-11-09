classdef SVM_Obj_Detec
    
    methods(Static)
        
        function [] = computeAP()
            [trD, trLb, valD, valLb, trRegs, valRegs] = HW4_Utils.getPosAndRandomNeg();
            [w,b] = SVM_Impl.get_primal_variables(trD,trLb,2);
            
            HW4_Utils.genRsltFile(w, b, "val", "/Users/vamshimuthineni/Vamshi/ML/hw4/cse512hw4/Objval");
            disp(HW4_Utils.cmpAP("/Users/vamshimuthineni/Vamshi/ML/hw4/cse512hw4/Objval", "val"));
        end
        
         
        function [] = hardNegMinAlgo2()
            [trD, trLb, valD, valLb, trRegs, valRegs] = HW4_Utils.getPosAndRandomNeg();
            [w,b,alpha,obj_val] = SVM_Impl.get_primal_variables(trD,trLb,15);
            [n,m] = size(trLb);
            posD = [];
            negD = [];
            support_negD = [];
            j = 1;
            k = 1;
            for i=1:n
                if trLb(i)==1
                    posD(:,j) = trD(:,i);
                    j = j+1;
                elseif(alpha(i)>0.01)
                    support_negD(:,k) = trD(:,i);
                    k = k+1;
                end
            end
            train_obj_values = [];
            val_aps = [];
            iters = [];
            for iter = 1:10
                total_trD = [posD,support_negD];
                k = 1000 - size(total_trD,2);
                [B, lb, imRegs] = HW4_Utils.extractHardNegatives('train',k,w,b);
                total_trD = [total_trD,B];
                
                neg_trLb = -1*ones(size(support_negD,2)+size(B,2),1);                
                pos_trLb = ones(size(posD,2),1);
                total_trLb = [pos_trLb;neg_trLb];   %appends in the bottom
                w=[];
                b=0;
                obj_val = 0;
                ap = 0;
                filenam = strcat("abc"+num2str(iter));
                [w,b,alpha,obj_val] = SVM_Impl.get_primal_variables(total_trD,total_trLb,15);
                iters(iter) = iter;
                train_obj_values(iter) = -1 * obj_val;
                HW4_Utils.genRsltFile(w, b, "val", filenam);
                [ap, prec, rec] = HW4_Utils.cmpAP(filenam, "val");
                val_aps(iter) = ap;
                fprintf("\n iteration %d, ap: %f, obj val: %f, total values: %d",iter, ap, obj_val, size(total_trD,2));
                
                support_negD = [];
                j = 1;
                for i =1:size(total_trD,2)
                    if alpha(i)>0.01 && total_trLb(i) == -1
                        support_negD(:,j) = total_trD(:,i);
                        j = j+1;
                    end
                end
                total_trLb=[];
                total_trD=[];
            end
            
%             plot(iters,val_aps,'-bs','LineWidth',2);
%             title('Average Precision of Validation Data with Iterations');
%             xlabel('Iterations');
%             ylabel('Average Precision');
            
            plot(iters,train_obj_values,'-bs','LineWidth',2);
            title('Objective values of train Data with Iterations');
            xlabel('Iterations');
            ylabel('Objective values');
            
            
            HW4_Utils.genRsltFile(w, b, "test", "/Users/vamshimuthineni/Vamshi/ML/hw4/cse512hw4/112607946.mat");
            fprintf("\n iteration %d, ap: %f",iter, ap);
        end
        
    end
end

