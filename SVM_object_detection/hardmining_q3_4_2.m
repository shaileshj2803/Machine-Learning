run('/usr/local/MATLAB/R2018b/vlfeat/toolbox/vl_setup');

C = 10;
[trD, trLb, valD, valLb, trRegs, valRegs] = HW4_Utils.getPosAndRandomNeg();
[d, n] = size(trD);

PosD = [trD(:,trLb==1)];
NegD = [trD(:,trLb==-1)];
[~, num_pos] = size(PosD);
[~, num_neg] = size(NegD);
PosD = double(PosD);
NegD = double(NegD);
x_train = [PosD, NegD];
y_train = [ones(num_pos,1); -1*ones(num_neg,1)];
[weight, bias, objective, alpha] = SVM_Quadprog(x_train, y_train, C);

train_objectives = [];
val_avg_precisions = [];

max_iterations = 7
for iteration = 1:max_iterations
    disp(iteration);
    HW4_Utils.genRsltFile(weight, bias, 'train', 'train_q342.mat');
    [avg_precision, precision, recall] = HW4_Utils.cmpAP('train_q342.mat','train');
    load('train_q342.mat','rects');
    load('trainAnno.mat','ubAnno');
    count = length(rects);
    
    total_features = cell(1,count);
    for i = 1:count
        image_i = imread(sprintf('%s/%sIms/%04d.jpg', HW4_Utils.dataDir, "train", i));
        rects_i = rects{i};
        feature_i = zeros(size(trD,1),size(rects_i,2));
        for j = 1:size(rects_i,2)
            current_rect = rects_i(:,j);
            bounding_box = zeros(1,4);
            [width, height, channels] = size(image_i);
            bounding_box(1) = int16(max(0,current_rect(1)));
            bounding_box(2) = int16(max(0,current_rect(2)));
            bounding_box(3) = int16(min(height,current_rect(3)));
            bounding_box(4) = int16(min(width,current_rect(4)));
                        
            imReg = image_i(bounding_box(2):bounding_box(4),bounding_box(1):bounding_box(3),:);
            imReg = imresize(imReg, HW4_Utils.normImSz);
 
            feature_i(:,j) = HW4_Utils.cmpFeat(rgb2gray(imReg));
        end
        total_features{i} = feature_i;
    end
    total_features = cat(2,total_features{:});

    NegD_without_A = NegD(:,alpha(end - num_neg+1: end)>0.0005);

    % from HW4_Utils.cmpAP()
    nIm = length(ubAnno);
    if length(rects) ~= nIm
        error('result and annotation files mismatch. Are you using the right dataset?');
    end
    [detScores, isTruePos] = deal(cell(1, nIm));
    for i=1:nIm
        rects_i = rects{i};
        detScores{i} = rects_i(5,:);
        ubs_i = ubAnno{i}; 
        isTruePos_i = -ones(1, size(rects_i, 2));
        for j=1:size(ubs_i,2)
            ub = ubs_i(:,j);
            overlap = HW4_Utils.rectOverlap(rects_i, ub);
            isTruePos_i(overlap >= 0.3) = 1;
        end
        isTruePos{i} = isTruePos_i;
    end
    detScores = cat(2, detScores{:});
    isTruePos = cat(2, isTruePos{:});

    
    [~, idx]=sort(detScores,'descend');
    negative_examples = find(isTruePos(idx)==-1);
    count_negative_examples = length(negative_examples);

    hard_negative_idx = idx(negative_examples(1:min(count_negative_examples,1000)));

    B = HW4_Utils.l2Norm(total_features(:,hard_negative_idx));

    NegD=[NegD_without_A,B];

    
    

    [~, num_pos] = size(PosD);
    [~, num_neg] = size(NegD);
    PosD = double(PosD);
    NegD = double(NegD);
    x_train = [PosD, NegD];
    y_train = [ones(num_pos,1); -1*ones(num_neg,1)];
    [weight, bias, objective, alpha] = SVM_Quadprog(x_train, y_train, C);

    train_objectives(iteration)=objective;

    HW4_Utils.genRsltFile(weight, bias, 'val', 'val_q342.mat');
    avg_precision = HW4_Utils.cmpAP('val_q342.mat','val');
    val_avg_precisions(iteration) = avg_precision;
    
end


iteration = 1:max_iterations
figure(1)
train_objectives_fig = plot(iteration, train_objectives)
xlabel('iteration')
ylabel('objective_value')
saveas(train_objectives_fig,'train_objectives_fig.png')



figure(2)
val_avg_precision_fig = plot(iteration,val_avg_precisions)
xlabel('iteration')
ylabel('Validation Avg Precisions')
saveas(val_avg_precision_fig,'val_avg_precisions.png')

% save model
save('train_objectives.mat','train_objectives')
save('val_avg_precisions.mat','val_avg_precisions');

save('Weight.mat','weight');
save('bias.mat','bias');
HW4_Utils.genRsltFile(weight, bias, "test", "112046765");

