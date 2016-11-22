function [trainInputs, trainOutputs, testInputs, testOutputs] = splitData(data, inputs, outputs, split)
    [trainInd, valInd, testInd] = dividerand(length(data),split(1), 0, split(2));
    %[trainInd,valInd,testInd] = dividerand(length(data),split(1),split(2),split(3));
    trainInputs = inputs(:,trainInd);
    trainOutputs = outputs(:,trainInd);
    %valInputs = inputs(:,valInd);
    %valOutputs = outputs(:,valInd);
    testInputs = inputs(:,testInd);
    testOutputs = outputs(:,testInd);
end

