# def kernelExplainer(modelPred, testData):
#     return shap.KernelExplainer(modelPred, testData)

# def tabExplainer(trainingData):
#   exp = lime.lime_tabular.LimeTabularExplainer(np.asarray(trainingData), feature_names=list(trainingData), class_names=['T1D'], verbose=True, mode='classification')
#   return exp

# def plotExplanationFeatures(explainer, modelPred, start, stop):
#     expVals = explainer

#     # TODO: Look into explainer... need to find out if this is a from the shap library.
#     exp = explainer.explain_row()
    
#     exp = exp.explain_instance(np.asarray(x_dataTest[start]), modelPred, num_features=len(featureNames))
#     exPlot = exp.as_pyplot_figure()
#     exPlot.plot()
#     exPlot.show()
    
# #explainer = lime.lime_tabular.LimeTabularExplainer(x_dataTrain, feature_names=list(x_dataTrain), class_names=[0, 1], mode='classification')
# #exp = explainer.explain_instance(x_dataTest[2], modelKeras.predict, num_features=15, top_labels=1)
# #exPlot = exp.as_pyplot_figure(label=1)
# #exPlot.plot()
# #exPlot.show()

# # Per GitHub, needed to add this line in order to by pass a runtime error in SHAP.
# # https://github.com/slundberg/shap/issues/1110
# shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough

# # Use the first 100 training examples as our background dataset to integrate over
# explainer = shap.KernelExplainer(t1dPredModel, x_dataTrain[:100])
# #explainer = shap.KernelExplainer(t1dPredModel, x_dataTrain)
# # explain the first n predictions
# # explaining each prediction requires 2 * background dataset size runs
# #shapValues = explainer.shap_values(x_dataTest[:80])
# # all test batch rows
# nSamples = batchSizeTest
# #nSamples = 25
# shapValues = explainer.shap_values(x_dataTest, nsamples=nSamples)

# # summarize the effects of all the features
# shap.plots._beeswarm.summary_legacy(shapValues, featureNames, show=False)
# #shap.plots.beeswarm(explainer)
# plt.show()
# plt.clf()
# # plot the explanation of the first prediction
# # Note the model is "multi-output" because it is rank-2 but only has one column
# shap.force_plot(explainer.expected_value[0], shapValues[0][0], x_dataTest[0])
# plt.show()
# plt.clf()

# shap.summary_plot(shapValues[0], x_dataTest[0])
# plt.show()
# plt.clf()

# shap.summary_plot(shapValues, x_dataTest)
# plt.show()
# plt.clf()

# #exp = shap.KernelExplainer(t1dPredModel, shap.sample(x_dataTrain, 100))
# #exp = shap.DeepExplainer(t1dPredModel)
# #shap_values = exp.shap_values(x_dataTest)
# #shap.summary_plot(shap_values, x_dataTest, feature_names = featureNames, show=False)
# #plt.show()

# #shap.force_plot(predOutput, shapValues[10,:], x_dataTest[10,:])
# #plt.show()

# #shap.force_plot(predOutput, shapValues, x_dataTest)
# #plt.show()

# #exp = tabExplainer(x_dataTrain)
# #plotExplanationFeatures(exp, t1dPredModel.predict, 1, 10)

# # If the prediction value is less than 0.5 then the prediction is class = 0 = "authentic," 
# # otherwise the prediction is class = 1 = "forgery."
# #modelPrediction = ((modelPrediction) > 0.5).astype("int32")


# #modelPrediction = t1dPredModel.predict(x_dataTest,)
# #modelPrediction = (t1dPredModel.predict(x_dataTest, batch_size=batchSize, verbose=0) > 0.5).astype("int32")