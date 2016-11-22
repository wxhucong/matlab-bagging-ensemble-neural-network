function [outWeights, outBiases] = SGD(fileID, trainInputs, trainOutputs, weights, biases, numEpochs, batchSize, cost_function, eta, lmbda, mu_momentum, scheduleLearningRateFactor, activationF_hidden, activationF_derivative_hidden, activationF_output, activationF_derivative_output, firstHiddenNeuronsCount, dropout, valInputs, valOutputs, testInputs, testOutputs)
    % varaible initialization
    curEpoch = 1;
    trainSize = length(trainInputs);
    % costVal_worseCount, bestCostVal are monitor for the early stopping
    costVal_worseCount = 0;
    % this is not perfect to hard code bestCostVal, because Cost is
    % possible to be grater than 1
    bestCostVal = 10000000;
    previousWeightDelta = cell(1, length(weights));
    for i = 1:length(weights)
        weightDelata = zeros(size(weights{i}));
        previousWeightDelta{i} = weightDelata;
    end
    % plot related para
    EpochsList = [];
    CostList_AllTrain = [];
    AccList_AllTrain = [];
    CostList_AllVal = [];
    AccList_AllVal = [];
    CostList_AllTest = [];
    AccList_AllTest = [];
    
    % loop over numEpochs
    for i = 1:numEpochs
        % data shuffle
        reOrderIndex = randperm(size(trainInputs,2));
        reInputs = trainInputs(:,reOrderIndex);
        reOutputs = trainOutputs(:,reOrderIndex);

        % loop over mini batch
        for j = 1:batchSize:length(trainInputs)
            if(j+batchSize-1 > length(trainInputs))
                minibatch_x = reInputs(:,j:end);
                minibatch_y = reOutputs(:,j:end);
            else
                minibatch_x = reInputs(:,j:j+batchSize-1);
                minibatch_y = reOutputs(:,j:j+batchSize-1);
            end

            % matrixize the bias after getting each mini batch size
            biases_minibatch = matrixizeBiases(biases, length(minibatch_x(1,:)));
            
            % the dropout layer after the 1st hidden layer 
            % need to install adds on package dividerand
            dropoutIndex = dividerand(firstHiddenNeuronsCount, dropout, 1-dropout);
            dropoutVec = ones(1,firstHiddenNeuronsCount) ./ (1-dropout);
            dropoutVec(dropoutIndex) = 0;
            dropoutWeight = diag(dropoutVec);

            % feedforward
            [ weightedInputs, activations ] = Feedforward( minibatch_x, weights, biases_minibatch, activationF_hidden, activationF_output, dropoutWeight);
            
            % Backpropagate the error
            %[errors_minibatch] = Backpropagate(minibatch_y, weightedInputs, activations, weights, cost_function, activationF_hidden, activationF_derivative_hidden, activationF_output, activationF_derivative_output);
            [errors_minibatch] = NesterovBackpropagate(minibatch_y, weightedInputs, activations, weights, cost_function, activationF_hidden, activationF_derivative_hidden, activationF_output, activationF_derivative_output, mu_momentum, previousWeightDelta, dropoutWeight);

            % Momentum Gradient Descent (remember: we are update our weight and biases at Matrix level, so weightedInputs, activations, errors_minibatch will not sum together)
            [weights, biases, previousWeightDelta] = updateNetwork(errors_minibatch, eta, lmbda, activations, weights, biases, batchSize, mu_momentum, previousWeightDelta, trainSize);

        end
        %evaluate on epoch level
        [Cost_AllTrain, Acc_AllTrain, Cost_AllVal, Acc_AllVal, Cost_AllTest, Acc_AllTest] = evaluate(fileID, trainInputs, trainOutputs, weights, biases, curEpoch, cost_function, activationF_hidden, activationF_output, ...
            valInputs, valOutputs, testInputs, testOutputs);
        % append to plot
        EpochsList(end+1) = i;
        CostList_AllTrain(end+1) = Cost_AllTrain;
        AccList_AllTrain(end+1) = Acc_AllTrain;
        CostList_AllVal(end+1) = Cost_AllVal;
        AccList_AllVal(end+1) = Acc_AllVal;
        CostList_AllTest(end+1) = Cost_AllTest;
        AccList_AllTest(end+1) = Acc_AllTest;
        
        % early stop and schedule learning rate logic
        [earlyStop, bestCostVal, costVal_worseCount] = shouldEarlyStop(Cost_AllVal, bestCostVal, costVal_worseCount);
        if(costVal_worseCount > 5)
            eta = eta / scheduleLearningRateFactor;
        end
        if(earlyStop)
            fprintf(fileID, 'Stop training. Cost has not gotten any better for 10 epochs!!!\n');
            break;
        end
        curEpoch = curEpoch+1;
    end
    
    % plot
    %plotLogic(EpochsList, CostList_AllTrain, AccList_AllTrain, CostList_AllVal, AccList_AllVal, CostList_AllTest, AccList_AllTest);
    outWeights = weights;
    outBiases = biases;
end

