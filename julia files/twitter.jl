
using Keras, TensorFlow, DataFrames, Gadfly, PyCall, CSV

@pyimport sklearn.preprocessing as pSklearn
@pyimport sklearn.utils as uSklearn
@pyimport sklearn.model_selection as mSklearn
@pyimport string as strpy
@pyimport keras as keras
@pyimport keras.preprocessing.text as tKeras
@pyimport keras.preprocessing.sequence as sKeras
@pyimport keras.callbacks as cKeras
@pyimport keras.utils as uKeras
@pyimport keras.wrappers as wKeras

import Keras.Layers: Dense, LSTM, Embedding, Dropout, Flatten, Activation, Conv1D, MaxPooling1D, GlobalAveragePooling1D

df = CSV.read("../datasets/twitter/text_emotion.csv")
df = df[[:sentiment, :content]]

sentiment = String[]
content = String[]
for i = 1:length(df[:sentiment])
    push!(sentiment, get(df[:sentiment][i]))
    push!(content, get(df[:content][i]))
end

df = DataFrame()
df[:content] = content
df[:sentiment] = sentiment
typeof(df)

@show unique(df[:sentiment])

function train_test_split(data, at=0.7)
    n = nrow(data)
    idx = shuffle(1:n)
    train_idx = view(idx, 1:floor(Int32, at*n))
    test_idx = view(idx, (floor(Int32, at*n)+1):n)
    data[train_idx, :], data[test_idx,:]
end

printable = Set(strpy.printable)

text_to_clean = df[:content]
for i in 1:length(text_to_clean)
    text_to_clean[i] = filter(x -> x in printable, text_to_clean[i])
    text_to_clean[i] = replace(text_to_clean[i], r"[^a-zA-z0-9\s]","")
    text_to_clean[i] = lowercase(text_to_clean[i])
end
df[:content] = text_to_clean
length(df[:content])

train, test = train_test_split(df,0.8)
x_train = train[:content]
y_train = train[:sentiment]

x_test = test[:content]
y_test = test[:sentiment]

x_train = convert(Array, x_train)
y_train = convert(Array, y_train)
x_test = convert(Array, x_test)
y_test = convert(Array, y_test)

length_longest_comment_train = length(maximum(x_train))

println("Length of the longest tweet for training: $(length_longest_comment_train)")


classes_weigths = uSklearn.class_weight[:compute_class_weight]("balanced", unique(y_train), y_train)
#=
classes_weigths_test = uSklearn.class_weight[:compute_class_weight]("balanced", unique(y_test), y_test)

cw = []
for i = 1:length(classes_weigths)
    tmp = (classes_weigths[i] + classes_weigths_test[i])/2
    push!(cw, tmp)
end
=#
#samples_weights = uSklearn.class_weight[:compute_sample_weight]("balanced", y_train)

top_words = 2000
#max_words = 300

tokenizer = tKeras.Tokenizer(num_words = top_words, split = " ")

tokenizer[:fit_on_texts](x_train)
tokenizer[:fit_on_texts](x_test)

x_train = tokenizer[:texts_to_sequences](x_train)
x_test = tokenizer[:texts_to_sequences](x_test)


x_train = sKeras.pad_sequences(x_train)
x_test = sKeras.pad_sequences(x_test, maxlen=(size(x_train)[2]))

encoder = pSklearn.LabelBinarizer()
y_train = encoder[:fit_transform](y_train)
#encoded_y_label = encoder[:transform](y_train)

y_test = encoder[:fit_transform](y_test)

#encoded_y_label_test = encoder[:transform](y_test)

#convert integers to dummy variables
#y_train = uKeras.to_categorical(y_train)
####
#encoder_test = pSklearn.LabelEncoder()

#convert integers to dummy variables
#y_test = uKeras.to_categorical(y_test)


println("$(length(x_train)) train sequences\n$(length(x_test)) test sequences")

model = Keras.Sequential()
add!(model, Embedding(top_words, 128, input_length=size(x_train)[2]))
add!(model, Dropout(0.2))
add!(model, LSTM(200, recurrent_dropout=0.2))
add!(model, Dropout(0.2))
add!(model, Dense(13))
add!(model, Activation(:softmax))

callback = cKeras.TensorBoard(log_dir="./models/twitter", histogram_freq=0, write_graph=true, write_images=true)

compile!(model,loss=:categorical_crossentropy, optimizer=:adam, metrics=[:accuracy])
fit!(model, x_train, y_train, batch_size=32, epochs=10,
    verbose=2, 
    callbacks=[callback],
    class_weight=classes_weigths)

score = evaluate(model, x_test,y_test, batch_size = 32)
println("Final accuracy after training: $(score[2]*100)")

bo_dataset = "../datasets/twitter/Raw Twitter Timelines w No Retweets/BarackObama.csv"
dt_dataset = "../datasets/twitter/Raw Twitter Timelines w No Retweets/DonaldTrumpTweets.csv"

@time df_bo = readtable(bo_dataset)
@time df_dt = readtable(dt_dataset)

df_bo = df_bo[df_bo[:retweet] .== "False",:]
df_dt = df_dt[df_dt[:retweet] .== "False",:]

df_dt = df_dt[1:6896,:]
println("In this dataset we have $(length(df_bo[:text])) Barack Obama tweets and we have $(length(df_dt[:text])) Donal Trump tweets")

function clear_tweets(tweets)
    tweets_processed = []
    len = []
    for i =1:length(tweets)
        tweets[i] = filter(x -> x in printable, tweets[i])
        tweets[i] = replace(tweets[i] , r"[^a-zA-z0-9\s]","")
        tweets[i] = lowercase(tweets[i])
        push!(tweets_processed, tweets[i])
        push!(len, length( tweets[i]))
    end
    return tweets_processed, len
end

function predict_sentiment(tweets)
    text=convert(Array, tweets)
    
    tk_prediction = tKeras.Tokenizer(num_words=top_words)
    tk = tk_prediction[:fit_on_texts](text)
    
    target = tk_prediction[:texts_to_sequences](text)
    target = sKeras.pad_sequences(target, maxlen=(size(x_train)[2]))
    predictions = predict(model, target)
end

function decode_predictions(predictions)
    decoded = encoder[:inverse_transform](predictions)
    sentiments = []
    for i = 1:length(decoded)
        push!(sentiments, decoded[i])
    end
    return sentiments
end

bo_tweets, len_bo = clear_tweets(df_bo[:text])
bo_sentiment = predict_sentiment(bo_tweets)
bo_res = decode_predictions(bo_sentiment)
df_bo[:sentiment] = bo_res

occurences = by(df_bo, :sentiment, nrow)
plot(occurences, x=:sentiment, y=:x1, Geom.point, Guide.xlabel("Sentiment"), Guide.ylabel("# Occurences"))

sort(occurences, cols=[:x1], rev=true)

df_bo[:length_text] = len_bo
df_length = by(df_bo, :sentiment, df -> round(mean(df[:length_text])))

plot(df_length, x=:sentiment, y=:x1, Geom.point, Guide.xlabel("Sentiment"), Guide.ylabel("Average Characters in a Tweet"))

dt_tweets, len_dt = clear_tweets(df_dt[:text])
dt_sentiment = predict_sentiment(dt_tweets)
dt_res = decode_predictions(dt_sentiment)
df_dt[:sentiment] = dt_res

occurences_dt = by(df_dt, :sentiment, nrow)
plot(occurences_dt, x=:sentiment, y=:x1, Geom.point, Guide.xlabel("Sentiment"), Guide.ylabel("# Occurences"))

sort(occurences_dt, cols=[:x1], rev=true)

df_dt[:length_text] = len_dt
df_length_dt = by(df_dt, :sentiment, df -> round(mean(df[:length_text])))

plot(df_length_dt, x=:sentiment, y=:x1, Geom.point, Guide.xlabel("Sentiment"), Guide.ylabel("Average Characters in a Tweet"))

plot(layer(occurences_dt, x=:sentiment, y=:x1, Geom.point, Geom.line, Theme(default_color="orange")),
layer(occurences, x=:sentiment, y=:x1, Geom.point, Geom.line, Theme(default_color="blue")),
 Guide.xlabel("Sentiment"), Guide.ylabel("# Occurrences"),  Guide.manual_color_key("Legend", ["Donald Trump", "Barack Obama"], ["orange", "blue"]))

plot(layer(df_length_dt, x=:sentiment, y=:x1, Geom.point, Geom.line, Theme(default_color="orange")),
layer(df_length, x=:sentiment, y=:x1, Geom.point, Geom.line, Theme(default_color="blue")),
 Guide.xlabel("Sentiment"), Guide.ylabel("Tweets Length"),  Guide.manual_color_key("Legend", ["Donald Trump", "Barack Obama"], ["orange", "blue"]))
