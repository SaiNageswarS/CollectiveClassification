macF1Multi = zeros(10, 4);

for p=1:10
    %testTwitterOlympics;
    %testCollectiveClassification_WebKB;
    %test3Sources;
    %testTwitterFootball;
    testCollective_Citeseer;
    macF1Multi(p, 1) = numUnknowns;
    macF1Multi(p, 2) = macroF1;
    
    
    %testCollectiveClassification_ICA;
    %macF1Multi(p, 3) = macroF1;
    
    temp1;
    %testTwitterFootballCollectiveEnsemble;
    macF1Multi(p, 3) = macroF1;
    
    %testSingleRelationClassification;
    %macF1Multi(p, 5) = macroF1;
    
end

for i=1:10
    fprintf('%d, %d, %d\n', macF1Multi(i, 1), macF1Multi(i, 2), macF1Multi(i, 3));
    %fprintf('%d, %d, %d\n', macF1Multi(i, 1), macF1Multi(i, 2), macF1Multi(i, 3));
end

fprintf(' Mean : \n');
fprintf('%d, %d, %d\n', mean(macF1Multi(:, 1)), mean(macF1Multi(:, 2)), mean(macF1Multi(:, 3)));