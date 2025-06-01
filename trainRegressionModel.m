function [trainedModel, validationRMSE] = trainRegressionModel(trainingData)
%trainingData: һ�����������������������뵼�뵽Ӧ�ó����е�������ͬ��
%trainedModel: һ������ѵ���õĻع�ģ�͵Ľṹ�塣�ýṹ������й�ѵ���õ�ģ�͵ĸ����ֶΡ�
%trainedModel.predictFcn: һ�����ڶ������ݽ���Ԥ��ĺ�����
%validationRMSE: һ��������������RMSE����˫����������Ӧ�ó����У���ʷ�б���ʾ��ÿ��ģ�͵�RMSE��
inputTable = array2table(trainingData, 'VariableNames', {'column_1', 'column_2', 'column_3', 'column_4', 'column_5', 'column_6'});

predictorNames = {'column_1', 'column_2', 'column_3', 'column_4', 'column_5'};
predictors = inputTable(:, predictorNames);
response = inputTable.column_6;
isCategoricalPredictor = [false, false, false, false, false];

%ѵ��һ���ع�ģ��
regressionGP = fitrgp(...
    predictors, ...
    response, ...
    'BasisFunction', 'constant', ...
    'KernelFunction', 'matern52', ...
    'Standardize', true);
% ����Ԥ�ⷽ�̵Ľṹ
predictorExtractionFcn = @(x) array2table(x, 'VariableNames', predictorNames);
gpPredictFcn = @(x) predict(regressionGP, x);
trainedModel.predictFcn = @(x) gpPredictFcn(predictorExtractionFcn(x));
% �Խṹ���Լ������
trainedModel.RegressionGP = regressionGP;
trainedModel.About = 'This struct is a trained model exported from Regression Learner R2018a.';
trainedModel.HowToPredict = sprintf('To make predictions on a new predictor column matrix, X, use: \n  yfit = c.predictFcn(X) \nreplacing ''c'' with the name of the variable that is this struct, e.g. ''trainedModel''. \n \nX must contain exactly 5 columns because this model was trained using 5 predictors. \nX must contain only predictor columns in exactly the same order and format as your training \ndata. Do not include the response column or any columns you did not import into the app. \n \nFor more information, see <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ''appregression_exportmodeltoworkspace'')">How to predict using an exported model</a>.');

% ��ȡԤ���������Ӧ����
% ��δ��뽫���ݴ�����ʺ�ѵ��ģ�͵ĸ�ʽ��
% ������ת��Ϊ���
inputTable = array2table(trainingData, 'VariableNames', {'column_1', 'column_2', 'column_3', 'column_4', 'column_5', 'column_6'});

predictorNames = {'column_1', 'column_2', 'column_3', 'column_4', 'column_5'};
predictors = inputTable(:, predictorNames);
response = inputTable.column_6;
isCategoricalPredictor = [false, false, false, false, false];

% ִ�н�����֤
partitionedModel = crossval(trainedModel.RegressionGP, 'KFold', 5);

% ������Լ�Ԥ��ֵ
validationPredictions = kfoldPredict(partitionedModel);

% ������Լ�RMSE
validationRMSE = sqrt(kfoldLoss(partitionedModel, 'LossFun', 'mse'));

