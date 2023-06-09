# def fault_detection_process(signal: pd.Series, h:int, N:int, esn: rpy.model.Model, s:int, s_tilde: int):
#   flag = False

#   T = len(signal)

#   # Apply denoising process
#   denoised_signal = process_ndvi_ts(signal)

#   # Train denoised signal ESN
#   Xtrain = denoised_signal[:s_tilde]
#   ytrain = denoised_signal[1:s_tilde + 1]
#   esn.reset()
#   esn.fit(Xtrain, ytrain)

#   # Predict denoised signal in generative mode
#   denoised_signal_prediction = forecast(esn, forecast_len=T - (s_tilde + 1), memory=52, warmup=10)

#   assert denoised_signal[s_tilde + 1:].shape == denoised_signal_prediction.shape

#   # Compute residuals
#   residuals = denoised_signal[s_tilde + 1:] - denoised_signal_prediction

#   # Train residuals ESN
#   Xtrain = residuals[:s]
#   ytrain = residuals[1:s + 1]
#   esn.reset()
#   esn.fit(Xtrain, ytrain)

#   # Predict residuals in generative mode
#   residuals_prediction = forecast(esn, forecast_len=T - (s + 1), memory=52, warmup=10)

#   assert denoised_signal[s + 1:].shape == residuals_prediction.shape

#   # Compute lower bound

#   lower_bound = None

#   i = 0
#   while not flag:
#     i += h

#     # flag = exists t such as p_t < LB for N consecutive observations
#     if flag:
#       break

#   # return date associated with i
  
#   pass