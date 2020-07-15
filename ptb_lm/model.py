import torch.nn as nn
import torch

class BasicLSTM(nn.Module):

    def __init__( self, hidden_size, num_layers):
        super( BasicLSTM, self).__init__()

        self.linear_arra = nn.ModuleList( [ nn.Linear( hidden_size * 2,  hidden_size * 4)  for i in range(num_layers)] )


        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, input, hidden):
        seq_len, batch_size, _ = input.size()
        
        pre_hidden, pre_cell = hidden

        pre_hidden_array = []
        pre_cell_array = []

        for i in range(self.num_layers):
            pre_hidden_array.append( pre_hidden[i])
            pre_cell_array.append( pre_cell[i]) 

        def step( step_in, pre_hidden, pre_cell, layers_index):
            concat_in = torch.cat( [step_in, pre_hidden], dim=1)
            gate_input = self.linear_arra[layers_index]( concat_in )
            
            s_res = torch.split(gate_input, [self.hidden_size, self.hidden_size, self.hidden_size, self.hidden_size], dim=1)

            #print( s_res )
            i = s_res[0]
            j = s_res[1]
            f = s_res[2]
            o = s_res[3]

            c = pre_cell * torch.sigmoid(f) + torch.sigmoid(i) * torch.tanh(j)
            m = torch.tanh(c) * torch.sigmoid(o)

            return m, c
        
        out_arr = []
        for i in range(seq_len):
            step_in = input[i]
            for i in range(self.num_layers):
                new_hidden, new_cell = step( step_in, pre_hidden_array[i], pre_cell_array[i], i)

                pre_hidden_array[i] = new_hidden
                pre_cell_array[i] = new_cell

                step_in = new_hidden
            out_arr.append( step_in )

        final_output = torch.cat( out_arr, dim=0).view( seq_len, batch_size, self.hidden_size)

        last_hidden = torch.cat( pre_hidden_array, dim=0).view( self.num_layers, batch_size, self.hidden_size )
        last_cell = torch.cat( pre_cell_array, dim =0).view( self.num_layers, batch_size, self.hidden_size )

        return final_output, (last_hidden, last_cell)
 




class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        '''
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        '''

        self.rnn = BasicLSTM( nhid, nlayers)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.encoder(input) 
        output, hidden = self.rnn(emb, hidden)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)
