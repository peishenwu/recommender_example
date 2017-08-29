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
preds <- layer_dot(list(embed_user, embed_movie), axes = -1) %>% layer_flatten 

#can not use pipe...
preds <- layer_add(inputs = list(preds, bias_user))
preds <- layer_add(inputs = list(preds, bias_movie))
         
##attach input-output
model <- keras_model(inputs = c(input_user, input_movie), outputs = preds)
model %>% compile(loss = 'mse', 
                  optimizer = optimizer_adam(),
                  metric = c('accuracy'))
#
history <- model %>% fit(x = list(train_set$user_id %>% matrix(ncol = 1), 
                                  train_set$movie_id %>% matrix(ncol = 1)), 
                         y = train_set$ratings %>% matrix(ncol = 1),
                         epochs = 30,
                         batch_size = 1000,
                         validation_split = 0.1)

# plot training process
plot(history)

# check model
summary(model)
