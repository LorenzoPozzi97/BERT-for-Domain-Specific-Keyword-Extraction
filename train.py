def train(model, optimizer, criterion, train_data, train_labels, test_data, test_labels, epochs, device = device):
  """
    Args:

      model:...

      optimizer:...

      criterion: loss rirerion

      train_data (pandas.core.series.Series): dataset containinig token embeddings
                        form BertTokenizer (dict) of (tensor) ≈ (batch, 512)

      train_labels (pandas.core.series.Series): dataset including two columns, 
                        namely, 'Tags' (tensor) ≈ (batch, 512), and 'Relevance' 
                        ≈ (tensor ≈ (batch) with groud-proof labels at token and
                        sentence level respectively  
    
      epochs (int): number of training epochs 
  """
 
  # ========================================
  #               Training
  # ========================================
  for epoch_i in range(epochs):  
    train_loss = []
    
    # Put the model into training mode.
    model.train()
    
    for i, v in train_data.items():
      # Step 1. Clear gradients out before each instance
      model.zero_grad()
      optimizer.zero_grad()

      # Step 2. Run the forward pass.
      train_input_ids = train_data[i]['input_ids'].to(device)                    # (batch, max_seq_length)
      train_input_mask = train_data[i]['attention_mask'].to(device)              # (batch, max_seq_length)
      train_input_type = train_data[i]['token_type_ids'].to(device)              # (batch, max_seq_length)
      token_train_label = train_labels['Tags'][i].to(device)                     # (batch, max_seq_length)
    
      output, pooled_output = model(train_input_ids.long(), 
                                    attention_mask = train_input_mask.long(), 
                                    token_type_ids = train_input_type.long())    # (batch, max_seq_length, 1)
  
      loss = criterion(output.squeeze(), token_train_label.double())
      
      # mask loss: the head, and special tokens corresponding loss should not affect the wight update
      loss = torch.where(train_input_type == 0, loss, torch.tensor(0, dtype=torch.double).to(device))
      
      avg_loss = torch.tensor([0]).to(device)
      for l in loss:
        avg_loss = torch.add(avg_loss, l[l.nonzero()].mean())
      
      loss = avg_loss/loss.size()[0]
      
      # update weights
      loss.backward()
      optimizer.step()
      
      # store loss in a list
      train_loss.append(loss.item())

    
    
    # ========================================
    #               Validation
    # ========================================
 
    model.eval()
    test_loss = []
    y_true, probas_pred = [], []
    
    for i, v in test_data.items():
      test_input_ids = test_data[i]['input_ids'].to(device)                      # (batch, max_seq_legth)
      test_input_mask = test_data[i]['attention_mask'].to(device)                # (batch, max_seq_legth)
      test_input_type = test_data[i]['token_type_ids'].to(device)                # (batch, max_seq_legth)
      token_test_label = test_labels['Tags'][i].to(device)                       # (batch, max_seq_legth)
      
      with torch.no_grad():   
        
        output, _ = model(test_input_ids.long(), 
                          attention_mask = test_input_mask.long(), 
                          token_type_ids = test_input_type.long())               # (batch, max_seq_length, 1)

        y_true.extend(torch.flatten(token_test_label).tolist())
        probas_pred.extend(torch.flatten(output.squeeze(-1)).tolist())
        
        # compute the loss
        loss = criterion(output.squeeze(), token_test_label.double()) 
        loss = torch.where(test_input_type == 0, loss, torch.tensor(0, dtype=torch.double).to(device))

        avg_loss = torch.tensor([0]).to(device)
        for l in loss:
          avg_loss = torch.add(avg_loss, l[l.nonzero()].mean())
      
        loss = avg_loss/loss.size()[0]
        
        test_loss.append(loss.item())
   
    # ======================================
    #           DIAGNOSTIC ANALYSIS
    # ======================================
    
    train_loss = sum(train_loss)/len(train_loss)
    test_loss = sum(test_loss)/len(test_loss)
   
    print('\n\n======================== REPORT ========================')
    print('|')
    print('|{}Train Loss: {:.3f}'.format(' '*5, train_loss))
    print('|{}Test Loss: {:.3f}'.format(' '*5,test_loss))
    print('|')
    print('========================================================')
    print('\n NB: A drop in precision after the 10th epoch might indicate MORE novel terms!')
    
