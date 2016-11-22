function [isPerfectMatch] = evaluateForEachBatch(inputs, outputs, weights, biases)
    biases_AllTrain = cell(1, length(biases));
    for i = 1:length(biases)
        bias = repmat(biases{i},1, length(inputs));
        biases_AllTrain{i} = bias;
    end
    [ weightedInputs_AllTrain, activations_AllTrain ] = Feedforward( inputs, weights, biases_AllTrain );
    predictions = round(activations_AllTrain{end});
    correctCount = 0;
    for i = 1:length(outputs)
        if(predictions(:,i) == outputs(:,i))
            correctCount = correctCount+1;
        end
    end
    if(correctCount == length(outputs))
        isPerfectMatch = true;
    else
        isPerfectMatch = false;
    end
end

