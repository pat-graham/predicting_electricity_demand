winSize = 24 * 7 * 4

model = keras_model_sequential() 
model %>%
  layer_dense(units = FLAGS$nodes1, activation = "relu",
              input_shape = winSize) %>%
  layer_dense(units = FLAGS$nodes2, activation = "relu") %>%
  layer_dense(units = FLAGS$nodes3, activation = "relu") %>%
  layer_dense(units = FLAGS$nodes4, activation = "relu") %>%
  layer_dense(units = FLAGS$nodes5, activation = "relu") %>%
  layer_dense(units = 1)

model %>% compile(
  loss = "mse",
  optimizer = optimizer_rmsprop(),
  metrics = list("mean_absolute_error")
)

# Train neural network
history = model %>% fit(
  train_data, train_label,
  epochs = 20, 
  validation_split = 0.2,
  verbose = 0
)

# Plot the model fitting performance
plot(history, metrics = "mean_absolute_error", smooth = FALSE)