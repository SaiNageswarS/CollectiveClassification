classLabels = load('TwitterPoliticsUK/classLabels');
classLabels(classLabels==5) = 4;
n = length(classLabels);
c = max(classLabels);

fixLabels = randomLabelMask(c, 0.50, classLabels);
numUnknowns = length(find(fixLabels==-1))  %print number of unknowns

followedByGraph = load('TwitterPoliticsUK/followedBy.mtx');
G{1} = spconvert(followedByGraph);

followsGraph = load('TwitterPoliticsUK/follows.mtx');
G{2} = spconvert(followsGraph);

mentionsGraph = load('TwitterPoliticsUK/mentions.mtx');
G{3} = spconvert(mentionsGraph);

mentionedGraph = load('TwitterPoliticsUK/mentionedBy.mtx');
G{4} = spconvert(mentionedGraph);

retweetedGraph = load('TwitterPoliticsUK/retweets.mtx');
G{5} = spconvert(retweetedGraph);

retweetedByGraph = load('TwitterPoliticsUK/retweetedBy.mtx');
G{6} = spconvert(retweetedByGraph);

listHypergraph = load('TwitterPoliticsUK/listsMergedHypergraph.mtx');
H{1} = spconvert(listHypergraph);

tweetsHypergraph = load('TwitterPoliticsUK/tweetsHypergraph.mtx');
H{2} = spconvert(tweetsHypergraph);

alpha = [0.125 0.125 0.125 0.125 0.125 0.125 0 0.125];

clusterLabels = HypergraphMRCC.predict2(H, G, fixLabels, alpha);
[accuracy macroF1]=evalClassification(clusterLabels, classLabels, fixLabels, c);



