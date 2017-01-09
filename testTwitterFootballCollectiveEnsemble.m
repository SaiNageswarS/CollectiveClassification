classLabels = load('TwitterPoliticsUK/classLabels');
classLabels(classLabels==5) = 4;
n = length(classLabels);
c = max(classLabels);

%fixLabels = randomLabelMask(c, 0.50, classLabels);
numUnknowns = length(find(fixLabels==-1))  %print number of unknowns

followedByGraph = load('TwitterPoliticsUK/followedBy.mtx');
linkMat{1} = followedByGraph;

followsGraph = load('TwitterPoliticsUK/follows.mtx');
linkMat{2} = followsGraph;

mentionsGraph = load('TwitterPoliticsUK/mentions.mtx');
linkMat{3} = mentionsGraph;

mentionedGraph = load('TwitterPoliticsUK/mentionedBy.mtx');
linkMat{4} = mentionedGraph;

retweetedGraph = load('TwitterPoliticsUK/retweets.mtx');
linkMat{5} = retweetedGraph;

retweetedByGraph = load('TwitterPoliticsUK/retweetedBy.mtx');
linkMat{6} = retweetedByGraph;

listHypergraph = load('TwitterPoliticsUK/listsMergedHypergraph.mtx');
listHypergraph = spconvert(listHypergraph);

tweetsHypergraph = load('TwitterPoliticsUK/tweetsHypergraph.mtx');
tweetsHypergraph = spconvert(tweetsHypergraph);
contentData = [listHypergraph tweetsHypergraph];

dataMat.linkMat = linkMat;
dataMat.contentMat = contentData;
dataMat.n = n;
dataMat.c = c;

clusterLabels = CollectiveEnsemble(dataMat, fixLabels);

[accuracy macroF1]=evalClassification(clusterLabels, classLabels, fixLabels, c);



