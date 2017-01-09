classdef HypergraphMRCC
    methods (Static)
        
        %% stochastic matrix for graph
        function [theta, Pi] = computeStochasticMatrix_graph(G)
            %G is an adjacency matrix
            outDegree = sum(G, 2);
            invOutDegree = outDegree.^(-1);
            Dv = diagonalize(outDegree);
            invDv = diagonalize(invOutDegree);
            %beta = 0.9;
            %n = size(G, 1);

            %theta = beta*invDv*G + (1-beta)*ones(n,n)*(1/n);
            %Pi = beta*Dv/trace(Dv) + (1-beta)*speye(n,n)*(1/n);
            theta = invDv*G;
            theta = full(theta);
            Pi = outDegree/trace(Dv);
            %theta = (Pi^0.5*theta*Pi^-0.5 + Pi^-0.5*theta'*Pi^0.5)/2;
        end
        
        %% stochastic matrix for hypergraph
        function [theta, Pi] = computeStochaticMatrix_hypergraph(H, W)
            dv = sum(H*W, 2); %vertex degree vector
            invdv = dv.^(-1);
            Dv = diagonalize(dv);
            invDv = diagonalize(invdv);
            de = sum(H, 1)-1; %edge degree vector
            De = diagonalize(de); %edge degree diagonal matrix

            theta = invDv*H*W*De^(-1)*H';  % stochastic matrix
            theta(eye(size(theta))==1) = 0;
            Pi = dv/trace(Dv);
        end
        %% Compute weights to maximize purity
        function W = computeWeights(H, fixLabels)
            display('Computing weights of hyperedges');
            e = size(H, 2);
            W = zeros(e,3);

            for i = 1:e
                labelk = fixLabels(H(:,i)==1);
                labelk = labelk(labelk~=-1);    %considering only known instances
                
                %if isempty(labelk)
                %    fprintf('%d\n', i);
                %end
                [yPos, freq] = mode(labelk);   %max occuring label and its frequency
                yNegLengthk = length(labelk) - length(find(labelk==yPos));   %number of instances with other labels
                numNegativeInstances = length(find(fixLabels~=yPos)) - length(find(fixLabels==-1));

                hellingerSimilarity = ( (freq/length(find(fixLabels==yPos)))^0.5 - (yNegLengthk/numNegativeInstances)^0.5 )^2;
                %hellingerSimilarity = ( (freq/length(labelk))^0.5 - (yNegLengthk/length(labelk))^0.5 )^2;

                W(i,:) = [i i hellingerSimilarity];
            end

            index = ~isnan(W(:,3));
            W(isnan(W)) = mean(W(index,3));
            W = spconvert(W);
        end
        
        %% Combine Pi
        function [Pi_mix, Theta_mix] = combineRandomWalk(pi, theta, alpha)
            % pi is nXd matrix where n is the number of instances and d is
            % the number of views
            
            if length(alpha)==1
                Pi_mix = pi;
                Theta_mix = theta{1};
                return;
            end
            
            n = size(pi, 1);
            Pi_mix = sparse(n, 1);

            for i=1:length(alpha)
                Pi_mix = Pi_mix + alpha(i)*pi(:, i);
            end
            
            beta = zeros(size(pi));
            
            for i = 1:length(alpha)
                beta(:, i) = alpha(i)*(pi(:, i)./Pi_mix);
            end
             
            clear pi;
            
            Theta_mix = sparse(n, n);
            for i=1:length(alpha)
                Theta_mix = Theta_mix + diagonalize(beta(:, i))*theta{i};
            end
            
%             Theta_mix = sparse(Theta_mix);
        end
        
         %% Transductive inference
        function clusterLabels = transductiveInference(theta, Pi, fixLabels, v)
            
            c = max(fixLabels);
            n = size(theta, 1);

            classSize = zeros(c,1);
            for i=1:c
                classSize(i) = length(find(fixLabels==i));
                if (classSize(i) > 4)
                    classSize(i) = 0.75*classSize(i);
                end
            end

           display('Begin transductive inference');
           if (size(Pi, 2) == 1)
                Pi = Pi';
           else
                Pi = sum(Pi, 1);    %row vector
           end
           
           clusterLabels = fixLabels;
           %v = find(fixLabels==-1);

           for k=1:10   %Iterative semi-supervised learning
                for i = 1:length(v)
                    f = zeros(c, 1);
                                        
                    for j = 1:c
                        tempLabels = fixLabels;
                        tempLabels(tempLabels~=j) = -1;
                        tempLabels(tempLabels==j) = 1;
                        tempLabels(fixLabels==-1) = 0;  %Effective in first iteration of k only

                        % Consider only k-Nearest neighbors instead of all
                        % neighbors
                        prob = theta(:, v(i))';

                        [~, ind] = sort(prob, 'descend');
                        %numNeighbors = floor(0.1*length(find(prob)));
                        numNeighbors = length(find(prob));
                        numNeighbors(numNeighbors>classSize(j)) = floor(classSize(j));
                        
                        prob(ind(numNeighbors:n)) = 0;          %discarding least probable neighbors
                        prob = sparse(prob);

                        f(j, 1) = (Pi.*prob)*tempLabels;
                    end

                    [~, clusterLabels(v(i))] = max(f);
                end
                numChanges = length(find((clusterLabels - fixLabels)~=0));
                fprintf('Num of changes #%d : %d\n', k, numChanges);
                if numChanges==0
                   break;
                end
                fixLabels = clusterLabels;
           end
        end
           
        %% learn weights 
        function clusterLabels = predict(H, fixLabels, alpha)
            %H is a cell array with H{k} representing
            %kth hypergraph
            %fixLabels contain class labels of instances. fixLabels = -1
            %for instances whose class labels are not known.

            numInstances = size(H{1}, 1);
            numGraphs = length(H);
            
            Pi = zeros(numInstances, numGraphs);
            unknowns = find(fixLabels==-1);
            
            %for iter = 1:1
                for i = numGraphs:-1:1
                    W = HypergraphMRCC.computeWeights(H{i}, fixLabels);
                    [theta{i}, Pi(:, i)] = HypergraphMRCC.computeStochaticMatrix_hypergraph(H{i}, W);
                    H{i} = [];
                end

                [Pi_mix, Theta_mix] = HypergraphMRCC.combineRandomWalk(Pi, theta, alpha);
                clear theta Pi;
                clusterLabels = HypergraphMRCC.transductiveInference(Theta_mix, Pi_mix, fixLabels, unknowns);  %ignore this warning.. don't change to suggestion
                %numChanges = length(find(clusterLabels ~= fixLabels));
                %fprintf('Num of changes #%d : %d\n', iter, numChanges);
                %if numChanges==0
                %    break;
                %end
                
                %fixLabels = clusterLabels;
           %end
        end
        
        function clusterLabels = predict2(H, G, fixLabels, alpha)
            %H is a cell array with H{k} representing
            %kth hypergraph
            %fixLabels contain class labels of instances. fixLabels = -1
            %for instances whose class labels are not known.

            numInstances = size(H{1}, 1);
            numHGraphs = length(H);
            numGGraphs = length(G);
            
            Pi = zeros(numInstances, numHGraphs+numGGraphs);
            unknowns = find(fixLabels==-1);
            
            %for iter = 1:1
                for i = numHGraphs:-1:1
                    W = HypergraphMRCC.computeWeights(H{i}, fixLabels);
                    [theta{i}, Pi(:, i)] = HypergraphMRCC.computeStochaticMatrix_hypergraph(H{i}, W);
                    H{i} = [];  %releasing memory
                end
                
                for i = numGGraphs:-1:1
                    [theta{numHGraphs+i}, Pi(:, numHGraphs+i)] = HypergraphMRCC.computeStochasticMatrix_graph(G{i});
                    G{i} = [];
                end

                [Pi_mix, Theta_mix] = HypergraphMRCC.combineRandomWalk(Pi, theta, alpha);
                clear theta Pi;
                clusterLabels = HypergraphMRCC.transductiveInference(Theta_mix, Pi_mix, fixLabels, unknowns);  %ignore this warning.. don't change to suggestion
                %numChanges = length(find(clusterLabels ~= fixLabels));
                %fprintf('Num of changes #%d : %d\n', iter, numChanges);
                %if numChanges==0
                %    break;
                %end
                
                %fixLabels = clusterLabels;
           %end
        end
        
    end
end