classdef HypergraphDRandomWalk
    properties
        alphaMap
        betaMap
        L
        theta
        fixLabels
        numClasses
        numInstances
    end
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
    end
    
    methods
        
         %% Constructor
        function model = HypergraphDRandomWalk(fixLabels, L)
            model.numInstances = length(fixLabels);
            model.numClasses = max(fixLabels);
            model.fixLabels = fixLabels;
            model.theta = [];
            model.L = L;
            model.alphaMap = NaN(model.numInstances, L, model.numClasses);
            model.betaMap = NaN(model.numInstances, L, model.numClasses);
        end
        
        %% Compute forward variable 
        function [alpha, model]=computeAlpha(model, t, y, q)
            if ~isnan(model.alphaMap(q, t, y))
                alpha = model.alphaMap(q, t, y);
                return;
            end
            
            if t==1
                Ly = find(model.fixLabels==y);
                alpha = 0;
                for j=1:length(Ly)
                    q1 = Ly(j);
                    p = model.theta(q1, q);
                    
                    alpha = alpha + p/(length(Ly));
                end
                model.alphaMap(q, t, y) = alpha;
                return;
            end
            
            Lny = find(model.fixLabels~=y);
            alpha = 0;
            for j = 1:length(Lny)
                q1 = Lny(j);
                [alphaprev, model] = model.computeAlpha(t-1, y, q1);
                alpha = alpha + alphaprev*model.theta(q1, q);
            end
            model.alphaMap(q, t, y) = alpha;
        end
        
        %% Compute backward variable
        
        function [beta, model]=computeBeta(model, t, y, q)
            if ~isnan(model.betaMap(q, t, y))
                beta = model.betaMap(q, t, y);
                return;
            end
            
            if t==1
                Ly = find(model.fixLabels==y);
                beta = 0;
                for j=1:length(Ly)
                    q1 = Ly(j);
                    beta = beta + model.theta(q, q1);
                end
                model.betaMap(q, t, y) = beta;
                return;
            end
            
            Lny = find(model.fixLabels~=y);
            beta = 0;
            for j = 1:length(Lny)
                q1 = Lny(j);
                [betaprev, model] = model.computeBeta(t-1, y, q1);
                beta = beta + betaprev*model.theta(q, q1);
            end
            model.betaMap(q, t, y) = beta;
        end
        
         %% Transductive inference using D-Walks
        function clusterLabels = performDWalk(model)
            unknowns = find(model.fixLabels==-1);
            c = max(model.fixLabels);
            Bl = zeros(length(model.fixLabels), c);      %betweeness values
            
            %compute betweeness for each unknown instance
            denom = zeros(c, 1);
            
            for y=1:c
                Ly = find(model.fixLabels==y);
                for l=1:model.L
                    for j=1:length(Ly)
                        q1 = Ly(j);
                        [alpha, model] = model.computeAlpha(l, y, q1);
                        denom(y, 1) = denom(y, 1) + alpha;
                    end
                    fprintf('Computed denominator for y=%d, l=%d\n', y, l);
                end
            end
            
            for i=1:length(unknowns)
                q = unknowns(i);
                for y = 1:c
                    tot1 = 0;
                    
                    for l=1:model.L
                        tot2 = 0;
                        for t=1:l-1
                            [alpha, model] = model.computeAlpha(t, y, q);
                            [beta, model] = model.computeBeta(l-t, y, q);
                            tot2 = tot2 +  alpha * beta;
                        end
                        tot1 = tot1+tot2;
                    end
                    
                    Bl(q, y) = tot1/denom(y, 1);
                end
            end
            
            norm = sum(Bl, 2);
            
            for i=1:c
                Bl(:, i) = Bl(:, i)./norm; %likelihood
                Bl(:, i) = Bl(:, i).*(length(model.fixLabels==i)/length(model.fixLabels)); %posterior probability
            end
            clear norm;
            
            [~, clusterLabels] = max(Bl, [], 2);    %argmax
            clusterLabels(model.fixLabels~=-1) = model.fixLabels(model.fixLabels~=-1);
        end
           
        %% learn weights 
        function clusterLabels = predict(model, H, G, alpha)
            %H is a cell array with H{k} representing
            %kth hypergraph
            %fixLabels contain class labels of instances. fixLabels = -1
            %for instances whose class labels are not known.
            numHGraphs = length(H);
            numGGraphs = length(G);
            numGraphs = numHGraphs + numGGraphs;
            
            Pi = zeros(model.numInstances, numGraphs);
                        
            
            for i = numHGraphs:-1:1
                W = HypergraphDRandomWalk.computeWeights(H{i}, model.fixLabels);
                [thetaH{i}, Pi(:, i)] = HypergraphDRandomWalk.computeStochaticMatrix_hypergraph(H{i}, W);
                H{i} = [];
            end
            
            for i = numGGraphs:-1:1
                [thetaH{numHGraphs+i}, Pi(:, numHGraphs+i)] = HypergraphMRCC.computeStochasticMatrix_graph(G{i});
                G{i} = [];
            end

            [~, model.theta] = HypergraphDRandomWalk.combineRandomWalk(Pi, thetaH, alpha);
            clear thetaH Pi;
            clusterLabels = model.performDWalk(); 
            
        end        
    end
end