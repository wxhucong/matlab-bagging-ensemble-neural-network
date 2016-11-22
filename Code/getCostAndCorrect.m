function [Cost, correctCount] = getCostAndCorrect(inputs, weights, biases_matrix, outputs, cost_function, activationF_hidden, activationF_output)
    % set dropoutWeight as identity matrix, eye(length(weights{1}(:,1)))
    [ weightedInputs, activations ] = Feedforward( inputs, weights, biases_matrix, activationF_hidden, activationF_output, eye(length(weights{1}(:,1))));
    
    if strcmp(cost_function, 'log')
        Cost = 0;
        predictMatrix = activations{end};
        [row, col] = size(outputs);
        for i = 1:col
            for j = 1:row
                if outputs(j,i) == 1
                    Cost = Cost - log(predictMatrix(j,i));
                    break
                end
            end
        end
        Cost = 1.0/length(outputs)*Cost;
    elseif strcmp(cost_function, 'cross')
        Cost = -1.0/length(outputs) * sum(sum(outputs .* log(activations{end}) + (1-outputs) .* log(1-activations{end})));
    else %strcmp(cost_function, 'quad')
        Cost = 1.0/2.0/length(outputs) * sum(sum((activations{end} - outputs) .^ 2));
    end
    
    predictions = round(activations{end});
    correctCount = 0;
    for i = 1:length(outputs)
        if(predictions(:,i) == outputs(:,i))
            correctCount = correctCount+1;
        end
    end
    
end

