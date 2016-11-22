function [weights, baises] = NetworkBuilderLargeWeight(nodeLayers)
    weights = {};
    baises = {};
    if(length(nodeLayers)<2)
        % if there is only 1 layer in the network
        weights = [0];
        baises = [0];
    elseif(length(nodeLayers)==2)
        % if there is only input and output layer in the network
        weights = normrnd(0,1,[nodeLayers(2) nodeLayers(1)]);
        baises = normrnd(0,1,[nodeLayers(2) 1]);
    else
        for i = 1:(length(nodeLayers)-1)
            W = normrnd(0,1,[nodeLayers(i+1) nodeLayers(i)]);
            b = normrnd(0,1,[nodeLayers(i+1) 1]);
            weights{end+1} = W;
            baises{end+1} = b;
        end
    end
end
