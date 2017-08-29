library(tidyverse)
library(readr)
library(keras)

setwd("/Users/peishenwu/Google 雲端硬碟/【01】醫學/PMR/R3/Deep learning AI/R_keras/data/ml-100k")
ratings_df <- read_delim("u.data",delim="\t",col_names = c('user_id', 'movie_id', 'ratings', 'unix_timestamp'))
ratings_df %>% summary

train_set <- ratings_df %>% select(user_id, movie_id, ratings) %>% mutate(user_id = user_id-1,
                                                                          movie_id = movie_id-1) #start from zero
n_users = ratings_df$user_id %>% unique %>% length
n_movies = ratings_df$movie_id %>% unique %>% length
n_factors = 10

#define input layers
input_user <- layer_input(shape = c(1)) 
input_movie <- layer_input(shape = c(1))

##embedding parameter
embed_user <- input_user %>% layer_embedding(input_dim = n_users, 
                                             output_dim = n_factors, 
                                             input_length = 1)

embed_movie <- input_movie %>% layer_embedding(input_dim = n_movies, 
                                               output_dim = n_factors, 
                                               input_length = 1)
##bias
bias_user <- input_user %>% layer_embedding(input_dim = n_users, 
                                            output_dim = 1, 
                                            input_length = 1) %>% layer_flatten

bias_movie <- input_movie %>% layer_embedding(input_dim = n_movies, 
                                              output_dim = 1, 
                                              input_length = 1) %>% layer_flatten
#
#preds <- layer_dot(list(embed_user, embed_movie), axes = -1) %>% layer_flatten 
#preds <- layer_add(inputs = list(preds, bias_user))
#preds <- layer_add(inputs = list(preds, bias_movie))

## use a neural net instead of SVD
preds <- layer_concatenate(inputs = list(embed_user, embed_movie), axis = -1) %>% layer_flatten %>% 
         layer_dropout(rate = 0.4) %>% 
         layer_dense(units = 100, activation = 'relu') %>% 
         layer_dropout(rate = 0.5) %>% 
         layer_dense(units = 50, activation = 'relu') %>% 
         #layer_dense(units = 5, activation = 'softmax')  ## categorical approach
         layer_dense(units = 1, activation = 'relu')
### 

model <- keras_model(inputs = c(input_user, input_movie), outputs = preds)
# model %>% compile(loss = 'categorical_crossentropy',  ## categorical approach
#                   optimizer = optimizer_adam(),
#                   metric = metric_categorical_accuracy)

model %>% compile(loss = 'mse',
                  optimizer = optimizer_adam(),
                  metric = 'accuracy')

# check model
summary(model)

#train
history <- model %>% fit(x = list(train_set$user_id %>% matrix(ncol = 1), 
                                  train_set$movie_id %>% matrix(ncol = 1)), 
                         # y = to_categorical(train_set$ratings-1, num_classes = 5), ## categorical approach
                         y = train_set$ratings %>% matrix(ncol = 1),
                         epochs = 30,
                         batch_size = 1000,
                         validation_split = 0.05)

# plot training process
plot(history)



