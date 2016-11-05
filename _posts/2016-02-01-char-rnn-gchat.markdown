---
layout: post
title:  "Char-RNN for Google Hangouts Chat History"
date:   2016-02-01
category: machine-learning
feature_image: /img/char-rnn-gchat-feature.png
detail: true
---
After working through the lectures and assignments for Stanford's [CS231n: Convolutional Neural Networks](http://cs231n.stanford.edu/index.html) course, I wanted to learn more about Google's popular new [Tensorflow](https://www.tensorflow.org/) framework. Inspired by Andrej Karpathy's [char-rnn blog post](http://karpathy.github.io/2015/05/21/rnn-effectiveness/), I trained a character-level language model using a recurrent neural network (LSTM) in Tensorflow on a few years' worth of Google Hangouts chat history with one of my friends. By sampling from the model, you can simulate conversations between us character by character. The results are both fascinating and hilarious. 

### Downloading Google Hangouts chat history
Google allows you to download a JSON dump of your entire Google Hangouts chat history. The dump is not formatted in a very user-friendly way, but  there are several open-source parsers to help convert the JSON into a human-readable form. I used [Hangouts Reader](https://bitbucket.org/dotcs/hangouts-log-reader/). 

After parsing, my training data is a single file with 61,000 lines that looks something like this: 

```
[...]
<Tommy Mullaney> Did you see ex machina?
<Bryson Alef> nope
<Bryson Alef> I haven't seen any of these except the martian
<Tommy Mullaney> You should watch it
<Tommy Mullaney> I liked it a lot
<Bryson Alef> I’ll put it on the list hahaha
<Bryson Alef> It did look interesting
[...]
```

To make this more suitable for training a model, I convert each character to a one-hot encoded vector. The resulting dataset is of shape (2615710, 203), since there are 2.6 million total characters and 203 unique characters in the dataset. 

### Training the model
Recurrent neural networks (RNNs) are pretty straightforward in TensorFlow. My code for a 2-layer LSTM looks something like this: 

```python
### Hyperparameters
lstm_size = 128
n_layers = 2
n_steps = 100
learning_rate = 0.0001
training_iters = 60000
batch_size = 100

### Build graph
tf.reset_default_graph()

# Input has shape [batch_size, n_steps, input_size]
# batch_size can vary, so we represent it as 'None' in the placeholders
# Note: shorter sequences must be padded with 0's so that they have length n_steps
x = tf.placeholder(np.float32, shape=[None, n_steps, input_size])
y = tf.placeholder(np.float32, shape=[None, n_steps, input_size])

# Specify each seq's real length (without zero-padding) so TF won't compute the extra timesteps
seqlen = tf.placeholder(tf.int32, [None])

# Reshape inputs: RNN API expects n_steps-length list of [batch_size, input_size] tensors
x_tr = tf.transpose(x, perm=[1, 0, 2]) # shape: [n_steps, batch_size, input_size]
x_re = tf.reshape(x_tr, [-1, input_size]) # shape: [n_steps * batch_size, input_size]
x_sp = tf.split(0, n_steps, x_re) # split into n_steps-length list of [batch_size, input_size]

# RNN returns outputs (n_steps-length list of [batch_size, input_size]) and final state
lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_size, state_is_tuple=True)
stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * n_layers, state_is_tuple=True)
outputs, state = tf.nn.rnn(stacked_lstm, x_sp, dtype=tf.float32, sequence_length=seqlen)

# Reshape outputs to [n_steps*batch_size, input_size]
outputs = tf.reshape(tf.concat(1, outputs), [-1, lstm_size])

# Output activation scores for each word in vocabulary
W_y = tf.Variable(tf.truncated_normal([lstm_size, output_size]))
b_y = tf.Variable(tf.zeros([output_size]))
logits = tf.matmul(outputs, W_y) + b_y

# Reshape actual labels to same shape for accuracy comparison
y_reshaped = tf.reshape(y, [-1, input_size])
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, y_reshaped))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

### Train
saver = tf.train.Saver()
loss_history = []
acc_history = []

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    pos = 0
    epoch = 1
        
    for i in range(training_iters):
        # Generate next batch of character sequences
        batch_x = []
        batch_y = []
        batch_seqlen = []
        for _ in range(batch_size):
            if pos+n_steps+1 >= len(input_one_hot):
                print('EPOCH', epoch, 'COMPLETE\n--')
                pos = np.random.choice(range(n_steps)) # go back to random starting point
                epoch += 1
            seq_x = input_one_hot[pos:pos+n_steps]
            seq_y = input_one_hot[pos+1:pos+n_steps+1]
            batch_x.append(seq_x)
            batch_y.append(seq_y)
            batch_seqlen.append(len(seq_x))
            pos += n_steps
        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y)
        batch_seqlen = np.array(batch_seqlen)
        
        # Train model on batch
        batch_feed = {x: batch_x, y: batch_y, seqlen: batch_seqlen}
        sess.run(optimizer, feed_dict=batch_feed)
        
        # Maybe print status and sample
        if i % 100 == 0:
            batch_loss = sess.run(loss, feed_dict=batch_feed)
            batch_acc = sess.run(accuracy, feed_dict=batch_feed)
            loss_history.append([i, batch_loss])
            acc_history.append([i, batch_acc])
            print(datetime.datetime.now(), 
                  '| iter', i, 
                  'batch_loss:', batch_loss, 
                  'batch_acc:', batch_acc)
            print(repr(sample(sess, np.random.choice(chars), 200)))
            print('--')
```

For each training iteration, the network is fed a sequence of n_steps characters (actually, batches of sequences). Tensorflow computes the loss and then backpropagates the gradients and updates the model parameters. After doing that in a loop for a few hours on my laptop's CPU, you get a model that's starting to sample text with borderline coherence!


### Sampling from the model
To sample from the trained model, you feed it a few 'seed' characters. The model computes the distribution of the most likely character to follow that seeded sequence, and then randomly samples a character from that distribution. Append that character to the sequence, and then repeat! 

```python
def sample(sess, seed_input, num_to_sample, verbose=False, temperature=1.0):
    '''
    Sample a sequence of 'num_to_sample' characters from the model,
    given an initial 'seed_input' sequence of characters.
    
    If verbose, will print the top 5 predictions for each step.
    '''
    return_sequence = []
    for t in range(num_to_sample):
        # RNN can only look at most recent n_steps chars
        if len(seed_input) > n_steps:
            seed_input = seed_input[-n_steps:]

        # Convert input chars to zero-padded sequence of one-hot vectors
        seed_x = np.zeros([1, n_steps, input_size]) # 1 batch
        for step, c in enumerate(seed_input):
            seed_x[0, step, chars.index(c)] = 1

        # Predict scores and softmax probs
        scores = sess.run(logits, feed_dict={x:seed_x, seqlen:[len(seed_input)]})
        probs = np.exp(scores / temperature) / np.sum(np.exp(scores / temperature), axis=1, keepdims=True)
        
        if verbose:
            print('Top 5 predictions for char following "' + seed_input + '":')
            for char_idx in np.argsort(-probs[len(seed_input)-1])[:5]:
                print(repr(chars[char_idx]), probs[len(seed_input)-1][char_idx])

        # Sample next character from predicted probs
        next_char = np.random.choice(chars, size=None, p=probs[len(seed_input)-1])
        seed_input += next_char
        return_sequence.append(next_char)

    return ''.join(return_sequence)
```

Before training, the model samples characters at random:

```
'u\'ﾟ👴ql😦.l—vv🎶ɟ🍕pt✌ɯᕕ🎉😜_j"“Kɔ\\3ᐛm😜ƃ😢|ʍś🎆‸🍵A|━j…ʌéà😀🎂•+8u😁‘ïɥ🎉😬w）ìULOGdʍp😵▽J‶\tN🎁`ᕕ□;O?I2📣⇀🍱-H‶3-z🎁🍵P🍵û¯#ǝヮüśg⇀vw2jìǝ.JPś\t@ûʎ⇀━‸+í•‶cVo☜🍵ᕗ—/W🍱E😛îo\tiśï💩📣🎊😛🎂”）Bpɥ >🍵`🎊\'ןᕗ.8ƃx☞`😵ןD🍱a☜😵éJ╯ᕗ\\\xa0S4🍱😡━¯😘👴0S🍱2−KF=' 
```

After 1,000 iterations, the model starts to break characters into 'words':

```
"\t^T/M—)a yl1aniy eh?pemrisle<yrrsmss A$ewvhvntpdydy oo mprdvt'e obhuas por'ts rers<rry ole  ete't  fte pao swnce lalria sef hhd\n<oryms nAllef> hws iray\n<<Bsson AAlef> aut hhattowr nfrs itatel \n?BTsosn"
```

After 30,000 iterations, the model generates a synthetic chat log with roughly the same quality of conversation that Bryson and I typically have:

```
<Tommy Mullaney> YOU DED INHESBEF?
<Tommy Mullaney> You to cule slased but with as?
<Tommy Mullaney> sommuhh i'm goting a bluns as why things
<Bryson Alef> so anything
<Tommy Mullaney> Bdeaw I was gifty play!
<Tommy Mullaney> hahaha
<Bryson Alef> 4 dil-cass 
<Bryson Alef> alwholood berse, but that amazy confiliting witche
<Tommy Mullaney> mych
<Tommy Mullaney> shit
<Bryson Alef> Hoooooot wot the over wieh it's sead you cancrale thing) edge ore and such to 454
<Bryson Alef> goooo
<Bryson Alef> I like serts working hip at weeking vines times of mazy andride it of sworth
<Tommy Mullaney> Noooo
<Bryson Alef> every
```

And after 60,000 iterations with a softmax temperature of 0.5, you get something like this: 

```
<Tommy Mullaney> stuff is fixed to see the did you made a good people back and has are an and the start and the same on the scrients on make to probably we have a code the best so the were to see the machine they tile sucks and he was a little big there
<Bryson Alef> also good to still get the simely in the keyboard and the stual packing on that and for the time the big convertion of the exterday complented to a better of the spoid actually still better a few than and stuff
<Bryson Alef> never internet and the sime one po nsicial
<Tommy Mullaney> hahaha
<Tommy Mullaney> and you have a really the time with the reset to going with a lot of the people haha
<Tommy Mullaney> I think it's a lot of and least for the still and hear and size they're like me to the and cridion we can sell the company the enough coming to be a sent an and comple interesting to start the basically still have to make it the hames a mind
<Tommy Mullaney> i think i get a blow the better inversion be and too game for the time a new that it the screen and my in the comments to do any go to far
<Bryson Alef> that's that were but yeah
```

### Analyzing the language model

It's fascinating to examine the conditional character distributions generated by the language model. For example, if you seed the network with "\<Tommy Mullaney\>", you're basically asking it what I'm most likely to say, conditioned on no other context. It turns out my most likely starting character is 'h' (~10% of the time, probably for 'hahaha' more often than not). It randomly sampled 'r' in this case, and you'll notice that the model then becomes super confident that the next character is a vowel. Then after sampling 'i' it realizes with nearly 90% confidence that OH THE WORD HAS TO BE 'right'!

```
Top 5 predictions for char following "<Tommy Mullaney> ":
'h' 0.101372
'i' 0.0835867
'w' 0.0547253
'a' 0.0547076
't' 0.0521722
Top 5 predictions for char following "<Tommy Mullaney> r":
'e' 0.409155
'i' 0.358729
'o' 0.0960816
'a' 0.0718429
'u' 0.0506499
Top 5 predictions for char following "<Tommy Mullaney> ri":
'g' 0.875222
'd' 0.0680318
't' 0.0138404
'n' 0.00903321
's' 0.00446325
Top 5 predictions for char following "<Tommy Mullaney> rig":
'h' 0.981275
'u' 0.00404702
'g' 0.00294782
'r' 0.00264764
'i' 0.00181549
Top 5 predictions for char following "<Tommy Mullaney> righ":
't' 0.977532
'e' 0.00597925
' ' 0.00362782
'\n' 0.00311822
'i' 0.00264616
<Tommy Mullaney> right
```

Pretty cool stuff. 