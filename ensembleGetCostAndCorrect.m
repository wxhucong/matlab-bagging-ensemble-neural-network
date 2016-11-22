function [avgCost, votedCorrectCount] = ensembleGetCostAndCorrect(numNetwork, inputs, weightsEnsemble, biases_matrixEnsemble, outputs, cost_function, activationF_hidden, activationF_output)
    costEnsemble = cell(1,numNetwork);
    outputsEnsemble = cell(1,numNetwork);
    for k = 1:numNetwork
        weights = weightsEnsemble{k};
        biases_matrix = biases_matrixEnsemble{k};
        % set dropoutWeight as identity matrix, eye(length(weights{1}(:,1)))
        [ weightedInputs, activations ] = Feedforward( inputs, weights, biases_matrix, activationF_hidden, activationF_output, eye(length(weights{1}(:,1))));

        if strcmp(cost_function, 'log')
            costEnsemble{k} = 0;
            predictMatrix = activations{end};
            [row, col] = size(outputs);
            for i = 1:col
                for j = 1:row
                    if outputs(j,i) == 1
                        costEnsemble{k} = costEnsemble{k} - log(predictMatrix(j,i));
                        break
                    end
                end
            end
            costEnsemble{k} = 1.0/length(outputs)*costEnsemble{k};
        elseif strcmp(cost_function, 'cross')
            costEnsemble{k} = -1.0/length(outputs) * sum(sum(outputs .* log(activations{end}) + (1-outputs) .* log(1-activations{end})));
        else %strcmp(cost_function, 'quad')
            costEnsemble{k} = 1.0/2.0/length(outputs) * sum(sum((activations{end} - outputs) .^ 2));
        end

        outputsEnsemble{k} = activations{end};
    end
    
    % average the costEnsemble
    costSum = 0;
    for i = 1:numNetwork
        costSum = costSum + costEnsemble{i};
    end
    avgCost = costSum / numNetwork;
    % do weighted vote in outputsEnsemble
    sumOutputs = outputsEnsemble{1};
    for i = 2:numNetwork
        sumOutputs = sumOutputs + outputsEnsemble{i};
    end
    avgOuputs = sumOutputs ./ numNetwork;
    predictions = round(avgOuputs);
    
    votedCorrectCount = 0;
    for i = 1:length(outputs)
        if(predictions(:,i) == outputs(:,i))
            votedCorrectCount = votedCorrectCount+1;
        end
    end
    
end
