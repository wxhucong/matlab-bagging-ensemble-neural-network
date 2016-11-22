function [Cost_AllTrain, Acc_AllTrain, Cost_AllTest, Acc_AllTest] = ensembleEvaluate(numNetwork, fileID, trainInputs, trainOutputs, weightsEnsemble, biasesEnsemble, curEpoch, cost_function, activationF_hidden, activationF_output, testInputs, testOutputs)
    biases_AllTrainEnsemble = cell(1,numNetwork);
    %biases_AllValEnsemble = cell(1,numNetwork);
    biases_AllTestEnsemble = cell(1,numNetwork);
    for i = 1:numNetwork
        biases_AllTrainEnsemble{i} = matrixizeBiases(biasesEnsemble{i}, length(trainInputs));
        %biases_AllValEnsemble{i} = matrixizeBiases(biasesEnsemble{i}, length(valInputs));
        biases_AllTestEnsemble{i} = matrixizeBiases(biasesEnsemble{i}, length(testInputs));
    end
    
    [Cost_AllTrain, correctCount_AllTrain] = ensembleGetCostAndCorrect(numNetwork, trainInputs, weightsEnsemble, biases_AllTrainEnsemble, trainOutputs, cost_function, activationF_hidden, activationF_output);
    %[Cost_AllVal, correctCount_AllVal] = ensembleGetCostAndCorrect(numNetwork, valInputs, weightsEnsemble, biases_AllValEnsemble, valOutputs, cost_function, activationF_hidden, activationF_output);
    [Cost_AllTest, correctCount_AllTest] = ensembleGetCostAndCorrect(numNetwork, testInputs, weightsEnsemble, biases_AllTestEnsemble, testOutputs, cost_function, activationF_hidden, activationF_output);
    
    Acc_AllTrain = correctCount_AllTrain/length(trainOutputs);
    %Acc_AllVal = correctCount_AllVal/length(valOutputs);
    Acc_AllTest = correctCount_AllTest/length(testOutputs);
    
    %fprintf(fileID, 'Epoch %d | Cost: %.3f | Correct: %d/%d | Acc: %.2f || Cost: %.3f | Correct: %d/%d | Acc: %.2f || Cost: %.3f | Correct: %d/%d | Acc: %.2f\n', ...
    fprintf(fileID, 'Epoch %d | Cost: %.3f | Correct: %d/%d | Acc: %.4f || Cost: %.3f | Correct: %d/%d | Acc: %.4f\n', ...
        curEpoch, Cost_AllTrain, correctCount_AllTrain, length(trainOutputs), Acc_AllTrain, ...
        Cost_AllTest, correctCount_AllTest, length(testOutputs), Acc_AllTest);
        %Cost_AllVal, correctCount_AllVal, length(valOutputs), Acc_AllVal, ...
    
    %sprintf('Epoch %d | Cost: %.3f | Correct: %d/%d | Acc: %.2f || Cost: %.3f | Correct: %d/%d | Acc: %.2f || Cost: %.3f | Correct: %d/%d | Acc: %.2f\n', ...
    sprintf('Epoch %d | Cost: %.3f | Correct: %d/%d | Acc: %.4f || Cost: %.3f | Correct: %d/%d | Acc: %.4f\n', ...
        curEpoch, Cost_AllTrain, correctCount_AllTrain, length(trainOutputs), Acc_AllTrain, ...
        Cost_AllTest, correctCount_AllTest, length(testOutputs), Acc_AllTest)
        %Cost_AllVal, correctCount_AllVal, length(valOutputs), Acc_AllVal, ...
end

