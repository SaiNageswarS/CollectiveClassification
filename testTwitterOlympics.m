classLabels = load('TwitterOlympics/pp_olympics.classes');
n = 464;
c = max(classLabels);

fixLabels = randomLabelMask(c, 0.50, classLabels);
numUnknowns = length(find(fixLabels==-1))  %print number of unknowns

followedByGraph = load('TwitterOlympics/pp_olympics-followedby.mtx');
G{1} = spconvert(followedByGraph);

followsGraph = load('TwitterOlympics/pp_olympics-follows.mtx');
G{2} = spconvert(followsGraph);

mentionsGraph = load('TwitterOlympics/pp_olympics-mentions.mtx');
G{3} = spconvert(mentionsGraph);

mentionedGraph = load('TwitterOlympics/pp_olympics-mentionedby.mtx');
G{4} = spconvert(mentionedGraph);

retweetedGraph = load('TwitterOlympics/pp_olympics-retweets.mtx');
G{5} = spconvert(retweetedGraph);

retweetedByGraph = load('TwitterOlympics/pp_olympics-retweetedby.mtx');
G{6} = spconvert(retweetedByGraph);

listHypergraph = load('TwitterOlympics/pp_olympics-listmergedHypergraph.mtx');
H{1} = spconvert(listHypergraph);

tweetsHypergraph = load('TwitterOlympics/pp_olympics-tweetsHypergraph.mtx');
H{2} = spconvert(tweetsHypergraph);

alpha = [0.125 0.125 0.125 0.125 0.125 0.125 0 0.125];

clusterLabels = HypergraphMRCC.predict2(H, G, fixLabels, alpha);
[accuracy macroF1]=evalClassification(clusterLabels, classLabels, fixLabels, c);



