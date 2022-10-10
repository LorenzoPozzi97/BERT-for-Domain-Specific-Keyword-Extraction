class BertLstm(nn.Module):
    def __init__(self, config, doLSTM = 'on'):
      super(BertLstm, self).__init__()
      self.doLSTM = doLSTM
      self.num_labels = config.num_labels
      self.LSTM_hidden_dim = 125
      self.bert = BertModel.from_pretrained("bert-base-uncased")
      self.dropout = nn.Dropout(p=0.5)

      if self.doLSTM == 'on':
        self.lstm = nn.LSTM(input_size = config.hidden_size,  
                              hidden_size = self.LSTM_hidden_dim, 
                              batch_first=True, 
                              bidirectional=True,
                              dropout=0.05)
        self.classifier = nn.Linear(self.LSTM_hidden_dim*2, 1)
      if self.doLSTM == 'off':
        self.classifier = nn.Linear(config.hidden_size, 1)
    

    def forward(self, input_ids, attention_mask = None, token_type_ids = None):
      
      output = self.bert(input_ids, 
                         attention_mask, 
                         return_dict=False)
      output = output[0]
      input = self.dropout(output)
      if self.doLSTM == 'on':
        input, (h_0, c_0) = self.lstm(input)
      
      linear_output = self.classifier(input)
      return linear_output
