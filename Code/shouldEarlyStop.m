function [earlyStop, bestCostVal, costVal_worseCount] = shouldEarlyStop(Cost_AllVal, bestCostVal, costVal_worseCount)
    if(bestCostVal <= Cost_AllVal)
        costVal_worseCount = costVal_worseCount+1;
    else
        costVal_worseCount = 0;
        bestCostVal = Cost_AllVal;
    end

    if(costVal_worseCount >= 10)
        earlyStop = true;
    else
        earlyStop = false;
    end

end

