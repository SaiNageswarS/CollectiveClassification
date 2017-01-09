 %linkData = load('Cornell/citationLinks');
 contentData = load('WebKB2/WebKB_indexed.content');
 %contentData = load('Cornell/contentData.mat');
 %contentData = contentData.contentData;
 c = 5;
 
 lastCol = size(contentData, 2);
 classLabels = contentData(:, lastCol);

%% balancing classes. Discarding class 4&5 instances
% contentData(classLabels==4, :) = [];
% classLabels = contentData(:, lastCol);  %reassignment to align classLabel indices
% contentData(classLabels==5, :) = [];
% classLabels = contentData(:, lastCol);

%% 

contentData(:, lastCol) = [];   %removing class labels
n = size(contentData, 1);

fixLabels = randomLabelMask(c, 0.30, classLabels);
numUnknowns = length(find(fixLabels==-1))  %print number of unknowns

%% Preprocessing linkData

% l(:,1) = linkData(:,2);
% l(:,2) = linkData(:,1);
% l(:,3) = 1;
% linkData = l;  clear l;

%% classify using only content
  clear H;
  H{1} = contentData;
  clusterLabels = HypergraphMRCC.predict(H, fixLabels, [1]);

%% measure f1 and accuracy of results
[accuracy macroF1]=evalClassification(clusterLabels, classLabels, fixLabels, c);