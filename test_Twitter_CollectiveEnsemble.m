classLabels = load('TwitterOlympics/pp_olympics.classes');
n = 464;
c = max(classLabels);

%fixLabels = randomLabelMask(c, 0.40, classLabels);
numUnknowns = length(find(fixLabels==-1))  %print number of unknowns

followedByGraph = load('TwitterOlympics/pp_olympics-followedby.mtx');
linkMat{1} = followedByGraph;

followsGraph = load('TwitterOlympics/pp_olympics-follows.mtx');
linkMat{2} = followsGraph;

mentionsGraph = load('TwitterOlympics/pp_olympics-mentions.mtx');
linkMat{3} = mentionsGraph;

mentionedGraph = load('TwitterOlympics/pp_olympics-mentionedby.mtx');
linkMat{4} = mentionedGraph;

retweetedGraph = load('TwitterOlympics/pp_olympics-retweets.mtx');
linkMat{5} = retweetedGraph;

retweetedByGraph = load('TwitterOlympics/pp_olympics-retweetedby.mtx');
linkMat{6} = retweetedByGraph;

listHypergraph = load('TwitterOlympics/pp_olympics-listmergedHypergraph.mtx');
listHypergraph = spconvert(listHypergraph);

tweetsHypergraph = load('TwitterOlympics/pp_olympics-tweetsHypergraph.mtx');
tweetsHypergraph = spconvert(tweetsHypergraph);
contentData = [listHypergraph tweetsHypergraph];

dataMat.linkMat = linkMat;
dataMat.contentMat = contentData;
dataMat.n = n;
dataMat.c = c;

clusterLabels = CollectiveEnsemble(dataMat, fixLabels);

[accuracy macroF1]=evalClassification(clusterLabels, classLabels, fixLabels, c);



