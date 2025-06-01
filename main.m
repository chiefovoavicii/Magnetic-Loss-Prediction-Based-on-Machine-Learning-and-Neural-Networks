[trainedModel, validationRMSE]=  trainRegressionModel(x);
yfit_1 = trainedModel.predictFcn(X);
%绘制图形
% 假设真实值为 actualValues，预测值为 predictedValues
actualValues = x(:,6); % 例子中的真实值
predictedValues = zeros; % 例子中的预测值

% 创建一个图形窗口
figure;

% 绘制散点图
scatter(actualValues, predictedValues, 'filled');

% 添加对角线（表示完美预测）
hold on;
plot([min([actualValues predictedValues]) max([actualValues predictedValues])], ...
     [min([actualValues predictedValues]) max([actualValues predictedValues])], ...
     'r--', 'LineWidth', 2);
hold off;

% 添加图例
legend('Predictions', 'Perfect Prediction');

% 添加标题和轴标签
title('Comparison of Actual and Predicted Values');
xlabel('Actual Values');
ylabel('Predicted Values');

