run('/usr/local/MATLAB/R2018b/vlfeat/toolbox/vl_setup');
[trD, trLb, valD, valLb, trRegs, valRegs] = HW4_Utils.getPosAndRandomNeg();
C=1;
[weight, bias, objective, alpha] = SVM_Quadprog(trD, trLb, C);
HW4_Utils.genRsltFile(weight, bias, 'val', 'val_result_q_3_4_1.mat');
avg_precision = HW4_Utils.cmpAP('val_result_q_3_4_1.mat','val');
% avg_precision = 0.6815