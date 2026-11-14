# macOS vs Windows, original MiniRocket, fit by fit

```
fits compared (dataset x seed): 339
same split hash: 339 of 339
identical accuracy: 290; different: 49 on 26 datasets
size of a difference: 1 to 17 test series (more accurate on macOS in 26, on Windows in 23)
mean difference over datasets (macOS - Windows): -0.020 pp
largest: ElectricDevices seed 2: +17 of 7711 series; UWaveGestureLibraryX seed 0: +7 of 3582 series; Computers seed 1: -6 of 250 series; Computers seed 2: -6 of 250 series; SmallKitchenAppliances seed 1: -4 of 375 series
yardstick, different seed on Windows: identical accuracy in 117 of 339 seed pairs (same seed on the two machines: 290 of 339)
```

## Refit on Windows, compared stage by stage with the macOS hashes

```
Fits refitted on Windows: 339. Stage identical to the macOS hash:
  fit_input_equal                   339 of 339
  extractor_parameters_equal          6 of 339
  train_feature_equal                11 of 339
  test_feature_equal                 11 of 339
  scaler_parameters_equal            11 of 339
  classifier_parameters_equal        11 of 339
  prediction_equal                  279 of 339
First stage that differs (fit input excluded):
  extractor_parameters_equal        333 fits, of which 49 have a different accuracy on this machine than on macOS
  none                                6 fits, of which 0 have a different accuracy on this machine than on macOS
```
