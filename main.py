config = DistilBertConfig(hidden_size=768,
                    num_hidden_layers=12,
                    num_attention_heads=12,
                    intermediate_size=3072,
                    num_labels=2)
set_seed(CONFIG.seed)
model = IndexerBert(config, doLSTM = 'on').to(device)
optimizer = optim.Adam(model.parameters(), lr=CONFIG.learning_rate)
criterion = nn.BCEWithLogitsLoss(reduction='none')

train(model, 
      optimizer, 
      criterion, 
      train_features, 
      train_labels, 
      test_features, 
      test_labels,
      CONFIG.epochs)
