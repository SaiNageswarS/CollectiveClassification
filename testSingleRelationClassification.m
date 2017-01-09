%% citeseer
%linkData = load('citeseer/citationLinks'); %citationLinksHypergraph2 has self links removed and citationLinksHypergraph has self links
%contentData = load('citeseer/citeseer.content');
%c = 6;

%% cora
%linkData = load('cora/citationLinksHypergraph'); %citationLinksHypergraph2 has self links removed and citationLinksHypergraph has self links
addpath('/home/SaiNageswar/Matlab/libsvm-3.17/matlab');

contentData = load('cora/cora2.content');
c = 7;
lastCol = size(contentData, 2);
classLabels = contentData(:, lastCol);
contentData(:, lastCol) = [];   %removing class labels
fixLabels = randomLabelMask(c, 0.50, classLabels);
 
numUnknowns = length(find(fixLabels==-1))  %print number of unknowns

%nb = NaiveBayes.fit(contentData(fixLabels~=-1, :), fixLabels(fixLabels~=-1), 'Distribution', 'mn');
nb = libsvmtrain(fixLabels(fixLabels~=-1), contentData(fixLabels~=-1, :), '-c 1 -g 0.04 -b 1');
%nb = ClassificationTree.fit(contentData(fixLabels~=-1, :), fixLabels(fixLabels~=-1));
%predicted = nb.predict(contentData(fixLabels==-1, :));
[predicted, acc, ~] = libsvmpredict(classLabels(fixLabels==-1), contentData(fixLabels==-1, :), nb);
%updating labels
predictedLabels = fixLabels;
predictedLabels(fixLabels==-1) = predicted;
clusterLabels = predictedLabels;
acc
[accuracy macroF1]=evalClassification(clusterLabels, classLabels, fixLabels, c);