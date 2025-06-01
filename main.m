[trainedModel, validationRMSE]=  trainRegressionModel(x);
yfit_1 = trainedModel.predictFcn(X);
%����ͼ��
% ������ʵֵΪ actualValues��Ԥ��ֵΪ predictedValues
actualValues = x(:,6); % �����е���ʵֵ
predictedValues = zeros; % �����е�Ԥ��ֵ

% ����һ��ͼ�δ���
figure;

% ����ɢ��ͼ
scatter(actualValues, predictedValues, 'filled');

% ��ӶԽ��ߣ���ʾ����Ԥ�⣩
hold on;
plot([min([actualValues predictedValues]) max([actualValues predictedValues])], ...
     [min([actualValues predictedValues]) max([actualValues predictedValues])], ...
     'r--', 'LineWidth', 2);
hold off;

% ���ͼ��
legend('Predictions', 'Perfect Prediction');

% ��ӱ�������ǩ
title('Comparison of Actual and Predicted Values');
xlabel('Actual Values');
ylabel('Predicted Values');

