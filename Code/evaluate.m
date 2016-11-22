function [Cost_AllTrain, Acc_AllTrain, Cost_AllVal, Acc_AllVal, Cost_AllTest, Acc_AllTest] = evaluate(fileID, trainInputs, trainOutputs, weights, biases, curEpoch, cost_function, activationF_hidden, activationF_output, valInputs, valOutputs, testInputs, testOutputs)
    biases_AllTrain = matrixizeBiases(biases, length(trainInputs));
    biases_AllVal = matrixizeBiases(biases, length(valInputs));
    biases_AllTest = matrixizeBiases(biases, length(testInputs));
    
    [Cost_AllTrain, correctCount_AllTrain] = getCostAndCorrect(trainInputs, weights, biases_AllTrain, trainOutputs, cost_function, activationF_hidden, activationF_output);
    [Cost_AllVal, correctCount_AllVal] = getCostAndCorrect(valInputs, weights, biases_AllVal, valOutputs, cost_function, activationF_hidden, activationF_output);
    [Cost_AllTest, correctCount_AllTest] = getCostAndCorrect(testInputs, weights, biases_AllTest, testOutputs, cost_function, activationF_hidden, activationF_output);
    Acc_AllTrain = correctCount_AllTrain/length(trainOutputs);
    Acc_AllVal = correctCount_AllVal/length(valOutputs);
    Acc_AllTest = correctCount_AllTest/length(testOutputs);
    
    fprintf(fileID, 'Epoch %d | Cost: %.3f | Correct: %d/%d | Acc: %.2f || Cost: %.3f | Correct: %d/%d | Acc: %.2f || Cost: %.3f | Correct: %d/%d | Acc: %.2f\n', ...
        curEpoch, Cost_AllTrain, correctCount_AllTrain, length(trainOutputs), Acc_AllTrain, ...
        Cost_AllVal, correctCount_AllVal, length(valOutputs), Acc_AllVal, ...
        Cost_AllTest, correctCount_AllTest, length(testOutputs), Acc_AllTest);
    
    sprintf('Epoch %d | Cost: %.3f | Correct: %d/%d | Acc: %.2f || Cost: %.3f | Correct: %d/%d | Acc: %.2f || Cost: %.3f | Correct: %d/%d | Acc: %.2f\n', ...
        curEpoch, Cost_AllTrain, correctCount_AllTrain, length(trainOutputs), Acc_AllTrain, ...
        Cost_AllVal, correctCount_AllVal, length(valOutputs), Acc_AllVal, ...
        Cost_AllTest, correctCount_AllTest, length(testOutputs), Acc_AllTest)
end

