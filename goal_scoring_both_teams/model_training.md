### Data Preprocessing

The LightGBM model undergoes training on match event data from WSL seasons 2018/2019 and 2019/2020, with subsequent testing on the 2020/2021 season data.

Before initiating the training process, essential preprocessing steps are implemented. This involves the removal of outliers through KNN, exclusion of unrelated action series (those encompassing extraneous events like referee ball drops, game starts, and game ends), and application of location transformation.

### Handling Class Imbalance

Given the substantial imbalance in the training set, we employ random oversampling to address class imbalance effectively. Additionally, the `scale_pos_weight` parameter of the LightGBM model undergoes fine-tuning for optimal performance.

### Predictions

The output probability can be utilized in a manner analogous to Expected Goal.