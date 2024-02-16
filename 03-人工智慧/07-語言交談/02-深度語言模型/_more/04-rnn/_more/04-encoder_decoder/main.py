# Some part of the code was referenced from below.
# https://github.com/pytorch/examples/tree/master/word_language_model 
import sys
import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils import clip_grad_norm_
from data_utils import Dictionary, Corpus

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
embed_size = 128
hidden_size = 1024
num_layers = 1
num_epochs = 3 # 原為 5
num_samples = 1000     # number of words to be sampled
batch_size = 20
seq_length = 30
learning_rate = 0.002

def load_data(train_file):
    global corpus, ids, vocab_size, num_batches
    corpus = Corpus()
    ids = corpus.get_data(train_file, batch_size)
    print('ids.shape=', ids.shape)
    vocab_size = len(corpus.dictionary)
    print('vocab_size=', vocab_size)
    num_batches = ids.size(1) // seq_length

# RNN based language model
class RNNLM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(RNNLM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x, h):
        # Embed word ids to vectors
        x = self.embed(x)
        
        # Forward propagate LSTM
        out, (h, c) = self.lstm(x, h)
        
        # Reshape output to (batch_size*seq_length, hidden_size)
        out = out.reshape(out.size(0)*out.size(1), out.size(2))
        
        # Decode hidden states of all time steps
        out = self.linear(out)
        return out, (h, c)

# Truncated backpropagation
def detach(states):
    return [state.detach() for state in states] 

def train(name):
    global model
    model = RNNLM(vocab_size, embed_size, hidden_size, num_layers).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    for epoch in range(num_epochs):
        # Set initial hidden and cell states
        states = (torch.zeros(num_layers, batch_size, hidden_size).to(device),
                torch.zeros(num_layers, batch_size, hidden_size).to(device))
        
        for i in range(0, ids.size(1) - seq_length, seq_length):
            # Get mini-batch inputs and targets
            inputs = ids[:, i:i+seq_length].to(device) # 輸入為目前詞 (1-Batch)
            targets = ids[:, (i+1):(i+1)+seq_length].to(device) # 輸出為下個詞 (1-Batch)
            
            # Forward pass
            states = detach(states) # states 脫離 graph
            outputs, states = model(inputs, states) # 用 model 計算預測詞
            loss = criterion(outputs, targets.reshape(-1)) # loss(預測詞, 答案詞)
            
            # Backward and optimize
            optimizer.zero_grad() # 梯度歸零
            loss.backward() # 反向傳遞
            clip_grad_norm_(model.parameters(), 0.5) # 切斷，避免梯度爆炸
            optimizer.step() # 向逆梯度方向走一步

            step = (i+1) // seq_length
            if step % 100 == 0:
                print ('Epoch [{}/{}], Step[{}/{}], Loss: {:.4f}, Perplexity: {:5.2f}'
                    .format(epoch+1, num_epochs, step, num_batches, loss.item(), np.exp(loss.item())))

    # Save the model checkpoints
    # torch.save(model.state_dict(), 'model.ckpt')
    torch.save(model, name+'_model.ckpt')

def test(name):
    # Test the model
    with torch.no_grad():
        with open(name+'_sample.txt', 'w', encoding='utf-8') as f:
            # Set intial hidden ane cell states
            state = (torch.zeros(num_layers, 1, hidden_size).to(device),
                    torch.zeros(num_layers, 1, hidden_size).to(device))

            # Select one word id randomly # 這裡沒有用預熱
            prob = torch.ones(vocab_size)
            input = torch.multinomial(prob, num_samples=1).unsqueeze(1).to(device)

            for i in range(num_samples):
                # Forward propagate RNN 
                output, state = model(input, state)

                # Sample a word id
                prob = output.exp()
                word_id = torch.multinomial(prob, num_samples=1).item()

                # Fill input with sampled word id for the next time step
                input.fill_(word_id)

                # File write
                word = corpus.dictionary.idx2word[word_id]
                word = '\n' if word == '<eos>' else word + ' '
                f.write(word)

                if (i+1) % 100 == 0:
                    print('Sampled [{}/{}] words and save to {}'.format(i+1, num_samples, name+'_sample.txt'))

if len(sys.argv) < 3:
    print('usage: python main.py <name> (train or test)')
    exit()

name = sys.argv[1]
action = sys.argv[2]
load_data(name+"_train.txt")
if action == 'train':
    train(name)
elif action == 'test':
    model = torch.load(name+'_model.ckpt')
    test(name)
