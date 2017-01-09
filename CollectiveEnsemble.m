function predictedLabels = CollectiveEnsemble(dataMat, fixLabels)
% dataMat.linkMat - links between web pages due to hyperlinks. format : adjacency
%               list. i.e. vi vj in each line. It is a 3D matrix with
%               linkMat(:, :, 1) representing first graph
% dataMat.contentMat - attribute vector of words containing 1/0 vector for
%               presence/absence of word in page
% dataMat.n - number of nodes
% dataMat.c - number of classes

contentMat = dataMat.contentMat;
%linkMat = dataMat.linkMat;
n = dataMat.n;
c = dataMat.c;
a = size(contentMat, 2);

contentMat = [contentMat zeros(n, c)];

for k=1:length(dataMat.linkMat)
    linkMat = dataMat.linkMat{k};
    for i=1:n
        %finding neighbors of i in linkMat
        neighborLabels = fixLabels(linkMat(linkMat(:, 1)==i, 2));
        for j=1:c
            contentMat(i, a+j) = length(find(neighborLabels==j));
        end
    end

    nb{k} = NaiveBayes.fit(contentMat(fixLabels~=-1, :), fixLabels(fixLabels~=-1), 'Distribution', 'mn');
    %predicted = nb.predict(contentMat(fixLabels==-1, :));
    %updating labels
    %predictedLabels = fixLabels;
    %predictedLabels(fixLabels==-1) = predicted;
end

predictedLabels = fixLabels;
predictedAcrossModels = zeros(n, size(dataMat.linkMat, 3));

for iterCnt = 1:10
    for k=1:length(dataMat.linkMat)
        linkMat = dataMat.linkMat{k};
        for i=1:n
            %finding neighbors of i in linkMat
            neighborLabels = predictedLabels(linkMat(linkMat(:, 1)==i, 2));
            for j=1:c
                contentMat(i, a+j) = length(find(neighborLabels==j));
            end
        end

        %nb = NaiveBayes.fit(contentMat(fixLabels~=-1, :), fixLabels(fixLabels~=-1), 'Distribution', 'mn');
        predicted = nb{k}.predict(contentMat(fixLabels==-1, :));
        pLabels = fixLabels;
        pLabels(fixLabels==-1) = predicted;

        numChanges = length(find(predictedLabels ~= pLabels));
        fprintf('Num of changes #%d : %d\n', iterCnt, numChanges);

%         if numChanges==0
%             break;
%         end
        %predictedLabels = pLabels;
        predictedAcrossModels(:, k) = pLabels;
    end
    
    predictedLabels = mode(predictedAcrossModels, 2);
end


