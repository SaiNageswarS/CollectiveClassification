%% citeseer
% linkData = load('citeseer/citationLinksHypergraph2'); %citationLinksHypergraph2 has self links removed and citationLinksHypergraph has self links
% linkData = spconvert(linkData);
% 
% contentData = load('citeseer/citeseer.content');
% c = 6;

%% cora
linkData = load('cora/citationLinksHypergraph'); %citationLinksHypergraph2 has self links removed and citationLinksHypergraph has self links
linkData = spconvert(linkData);

contentData = load('cora/cora2.content');
c = 7;

%% WebKB Cornell
%linkData = load('Cornell/citationLinksHypergraph');
%contentData = load('Cornell/cornell_p.content');
%c=5;

%% processing corpus to hypergraph
lastCol = size(contentData, 2);
classLabels = contentData(:, lastCol);
contentData(:, lastCol) = [];   %removing class labels
 
n = size(contentData, 1);
fixLabels = randomLabelMask(c, 0.50, classLabels);

%% code to call collective classification and evaluate
numUnknowns = length(find(fixLabels==-1))  %print number of unknowns

alpha = [0.75 0.25];
clear H;
H{1} = contentData;
H{2} = linkData;
clusterLabels = HypergraphMRCC.predict(H, fixLabels, alpha);
%clusterLabels = HypergraphSVM.predict(H, fixLabels, alpha);
%clusterLabels = HypergraphDRandomWalk.predict(H, fixLabels, alpha);
[accuracy macroF1]=evalClassification(clusterLabels, classLabels, fixLabels, c);
