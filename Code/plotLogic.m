function plotLogic(EpochsList, CostList_AllTrain, AccList_AllTrain, CostList_AllTest, AccList_AllTest)
    figure
    subplot(1,2,1);
    plot(EpochsList,CostList_AllTrain,'DisplayName','Cost-Train');
    hold on
    %plot(EpochsList,CostList_AllVal,'DisplayName','Cost-Val');
    plot(EpochsList,CostList_AllTest,'DisplayName','Cost-Test');
    legend('show')
    title('Subplot 1: Cost Result')
    hold off
    
    subplot(1,2,2);
    plot(EpochsList,AccList_AllTrain,'DisplayName','Acc-Train');
    hold on
    %plot(EpochsList,AccList_AllVal,'DisplayName','Acc-Val');
    plot(EpochsList, AccList_AllTest,'DisplayName','Acc-Test');
    legend('show')
    title('Subplot 2: Acc Result')
    hold off
end

