function [trainedModel, validationRMSE] = trainRegressionModel(trainingData)
%trainingData: 一个矩阵，其列数和数据类型与导入到应用程序中的数据相同。
%trainedModel: 一个包含训练好的回归模型的结构体。该结构体包含有关训练好的模型的各种字段。
%trainedModel.predictFcn: 一个用于对新数据进行预测的函数。
%validationRMSE: 一个包含均方根误差（RMSE）的双精度数。在应用程序中，历史列表显示了每个模型的RMSE。
inputTable = array2table(trainingData, 'VariableNames', {'column_1', 'column_2', 'column_3', 'column_4', 'column_5', 'column_6'});

predictorNames = {'column_1', 'column_2', 'column_3', 'column_4', 'column_5'};
predictors = inputTable(:, predictorNames);
response = inputTable.column_6;
isCategoricalPredictor = [false, false, false, false, false];

%训练一个回归模型
regressionGP = fitrgp(...
    predictors, ...
    response, ...
    'BasisFunction', 'constant', ...
    'KernelFunction', 'matern52', ...
    'Standardize', true);
% 创建预测方程的结构
predictorExtractionFcn = @(x) array2table(x, 'VariableNames', predictorNames);
gpPredictFcn = @(x) predict(regressionGP, x);
trainedModel.predictFcn = @(x) gpPredictFcn(predictorExtractionFcn(x));
% 对结构添加约束条件
trainedModel.RegressionGP = regressionGP;
trainedModel.About = 'This struct is a trained model exported from Regression Learner R2018a.';
trainedModel.HowToPredict = sprintf('To make predictions on a new predictor column matrix, X, use: \n  yfit = c.predictFcn(X) \nreplacing ''c'' with the name of the variable that is this struct, e.g. ''trainedModel''. \n \nX must contain exactly 5 columns because this model was trained using 5 predictors. \nX must contain only predictor columns in exactly the same order and format as your training \ndata. Do not include the response column or any columns you did not import into the app. \n \nFor more information, see <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ''appregression_exportmodeltoworkspace'')">How to predict using an exported model</a>.');

% 提取预测变量和响应变量
% 这段代码将数据处理成适合训练模型的格式。
% 将输入转换为表格
inputTable = array2table(trainingData, 'VariableNames', {'column_1', 'column_2', 'column_3', 'column_4', 'column_5', 'column_6'});

predictorNames = {'column_1', 'column_2', 'column_3', 'column_4', 'column_5'};
predictors = inputTable(:, predictorNames);
response = inputTable.column_6;
isCategoricalPredictor = [false, false, false, false, false];

% 执行交叉验证
partitionedModel = crossval(trainedModel.RegressionGP, 'KFold', 5);

% 计算测试集预测值
validationPredictions = kfoldPredict(partitionedModel);

% 计算测试集RMSE
validationRMSE = sqrt(kfoldLoss(partitionedModel, 'LossFun', 'mse'));

