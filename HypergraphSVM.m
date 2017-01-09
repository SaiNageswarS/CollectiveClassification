classdef HypergraphSVM
    methods (Static)
        
        %% stochastic matrix for graph
        function [theta, Pi] = computeStochasticMatrix_graph(G)
            %G is an adjacency matrix
            outDegree = sum(G, 2);
            invOutDegree = outDegree.^(-0.5);
            Dv = diagonalize(outDegree);
            invDv = diagonalize(invOutDegree);
            %beta = 0.9;
            %n = size(G, 1);

            %theta = beta*invDv*G + (1-beta)*ones(n,n)*(1/n);
            %Pi = beta*Dv/trace(Dv) + (1-beta)*speye(n,n)*(1/n);
            theta = invDv*G*invDv;
            theta = full(theta);
            Pi = outDegree/trace(Dv);
            %theta = (Pi^0.5*theta*Pi^-0.5 + Pi^-0.5*theta'*Pi^0.5)/2;
        end
        
        %% stochastic matrix for hypergraph
        function [theta, Pi] = computeStochaticMatrix_hypergraph(H, W)
            dv = sum(H*W, 2); %vertex degree vector
            invdv = dv.^(-0.5);
            Dv = diagonalize(dv);
            invDv = diagonalize(invdv);
            de = sum(H, 1)-1; %edge degree vector
            De = diagonalize(de); %edge degree diagonal matrix

            theta = invDv*H*W*De^(-1)*H'*invDv;  % stochastic matrix
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
          
        %% learn weights 
        function clusterLabels = predict(H, fixLabels, alpha)
            %H is a cell array with H{k} representing
            %kth hypergraph
            %fixLabels contain class labels of instances. fixLabels = -1
            %for instances whose class labels are not known.

            numInstances = size(H{1}, 1);
            numGraphs = length(H);
            
            Pi = zeros(numInstances, numGraphs);
                     
            for i = numGraphs:-1:1
                W = HypergraphSVM.computeWeights(H{i}, fixLabels);
                [theta{i}, Pi(:, i)] = HypergraphSVM.computeStochaticMatrix_hypergraph(H{i}, W);
                H{i} = [];
            end

            [PiMix, thetaMix] = HypergraphSVM.combineRandomWalk(Pi, theta, alpha);
            clear theta Pi;
            
            PiMix = diagonalize(PiMix);
%             laplacian = eye(size(thetaMix)) - ((PiMix*thetaMix + thetaMix'*PiMix)/2)*PiMix^(-1);
%             
%             d = eigs(laplacian, 1);
%             kernel = (eye(size(laplacian)) + (1/d)*laplacian)^(-1);
            kernel = (thetaMix + thetaMix')/2;
            kernel(eye(size(kernel))==1) =1;
            
%             minEig = eigs(kernel, 1, 'sm');
%             fprintf('Smallest eigen value of kernel : %d\n', minEig);
%             
%             if minEig<0
%                 display('Error : Kernel is not PSD');
%             end
                
            
            display('Calculated Kernel');
            numTrain = length(find(fixLabels~=-1));
            numTest = length(find(fixLabels==-1));
            
            addpath('/home/SaiNageswar/Matlab/libsvm-3.17/matlab');
            model = libsvmtrain(fixLabels(fixLabels~=-1), [(1:numTrain)' , kernel(fixLabels~=-1, fixLabels~=-1)], '-t 4');  %kernel matrix of train-train
            predicted = libsvmpredict(ones(numTest, 1), [(1:numTest)' , kernel(fixLabels==-1, fixLabels~=-1)], model); %kernel matrix of test-train
            
            predictedLabels = fixLabels;
            predictedLabels(fixLabels==-1) = predicted;
            clusterLabels = predictedLabels;
        end
        
    end
end