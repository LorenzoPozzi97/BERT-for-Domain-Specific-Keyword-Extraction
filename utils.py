def dataBatcher(df, CONFIG.train_batch):
  rows = []
  for i in range(CONFIG.train_batch, len(df), CONFIG.train_batch): # skip the first iteration
    input_ids, attention_mask, token_type_ids, labels = [], [], [], []

    sentences = df['Enc Sentence'].iloc[i-CONFIG.train_batch:i]
    tags = df['Enc Tags'].iloc[i-CONFIG.train_batch:i]
    relevance = df['Relevance'].iloc[i-CONFIG.train_batch:i]
    head = df['Enc Heading'].iloc[i-CONFIG.train_batch:i]
    
    max_length_batch = max([torch.cat((s, head.iloc[i], torch.tensor([102])), 0).size()[0] for i, s in enumerate(sentences)]) 
    max_length_batch = 512 if max_length_batch>512 else max_length_batch
    
    for x, s in enumerate(sentences):
      if s.size()[0] + head.iloc[x].size()[0]>=512:
        # make size for the head and remuve SEP
        
        length_head = head.iloc[x].size()[0] ##if head.iloc[x][0]!=torch.tensor(1060) else 0
        s = s[:-(length_head+2)] 
        s = torch.cat((s,torch.tensor([102]), head.iloc[x], torch.tensor([102])), 0)
      else:
        s = torch.cat((s, head.iloc[x], torch.tensor([102])), 0)
      
      
      PAD = torch.zeros(max_length_batch - s.size()[0])
      
      mask_ones = torch.ones(s.size()[0])
      mask_zeros = torch.zeros(max_length_batch - s.size()[0])
      attention_mask.append(torch.cat((mask_ones, mask_zeros), 0).unsqueeze(0))
      
      type_zeros = torch.zeros(sentences.iloc[x].size()[0]) if sentences.iloc[x].size()[0]!=512 else torch.zeros(sentences.iloc[x].size()[0]-(head.iloc[x].size()[0]+1))
      type_ones = torch.ones(head.iloc[x].size()[0]+1 + (max_length_batch - s.size()[0]))
      if torch.cat((type_zeros, type_ones), 0).size()[0]>512:
        print('here', i)
        type_ones = type_ones[:512-torch.cat((type_zeros, type_ones), 0).size()[0]]
      token_type_ids.append(torch.cat((type_zeros, type_ones), 0).unsqueeze(0))

      s = torch.cat((s, PAD), 0)
      input_ids.append(s.unsqueeze(0))

      # Tags
      label_zeros = torch.zeros(max_length_batch - tags.iloc[x].size()[0])
      labels.append(torch.cat((tags.iloc[x], label_zeros), 0).unsqueeze(0))
  
    #print('Store the values...')

    # create batch (batch, dim)
    input_ids = torch.cat(input_ids, 0)
    attention_mask = torch.cat(attention_mask, 0)
    token_type_ids = torch.cat(token_type_ids, 0)
  
    batch_dict = {}
    batch_dict['input_ids'] = input_ids
    batch_dict['token_type_ids'] = token_type_ids
    batch_dict['attention_mask'] = attention_mask
  
    labels = torch.cat(labels, 0) # (batch, length)
    relevance = torch.cat([torch.tensor([r]) for r in relevance], 0) # (batch)
  
  
    rows.append([batch_dict, labels, relevance])

    if i+CONFIG.train_batch > len(df):
      input_ids, attention_mask, token_type_ids, labels = [], [], [], []
    
      sentences = df['Enc Sentence'].iloc[i:]
      tags = df['Enc Tags'].iloc[i:]
      relevance = df['Enc Relevance'].iloc[i:]
    
      max_length_batch = max([torch.cat((s, head.iloc[i], torch.tensor([102])), 0).size()[0] for i, s in enumerate(sentences)]) 
      max_length_batch = 512 if max_length_batch>512 else max_length_batch

      for x, s in enumerate(sentences):
        if s.size()[0] + head.iloc[x].size()[0]>=512:
          length_head = head.iloc[x].size()[0]
          s = s[:-(length_head+2)] 
          s = torch.cat((s,torch.tensor([102]), head.iloc[x], torch.tensor([102])), 0)
        else:
          s = torch.cat((s, head.iloc[x], torch.tensor([102])), 0)
        
        PAD = torch.zeros(max_length_batch - s.size()[0])
        mask_ones = torch.ones(s.size()[0])
        mask_zeros = torch.zeros(max_length_batch - s.size()[0])
        attention_mask.append(torch.cat((mask_ones, mask_zeros), 0).unsqueeze(0))

        type_zeros = torch.zeros(sentences.iloc[x].size()[0]) if sentences.iloc[x].size()[0]!=512 else torch.zeros(sentences.iloc[x].size()[0]-(head.iloc[x].size()[0]+1))
        type_ones = torch.ones(head.iloc[x].size()[0]+1 + (max_length_batch - s.size()[0]))
        token_type_ids.append(torch.cat((type_zeros, type_ones), 0).unsqueeze(0))

        s = torch.cat((s, PAD), 0)
        input_ids.append(s.unsqueeze(0))

        # Tags
        label_zeros = torch.zeros(max_length_batch - tags.iloc[x].size()[0])
        labels.append(torch.cat((tags.iloc[x], label_zeros), 0).unsqueeze(0))
    

      # create batch (batch, dim)
      input_ids = torch.cat(input_ids, 0)
      attention_mask = torch.cat(attention_mask, 0)
    
      token_type_ids = torch.cat(token_type_ids, 0)
    
      batch_dict = {}
      batch_dict['input_ids'] = input_ids
      batch_dict['token_type_ids'] = token_type_ids
      batch_dict['attention_mask'] = attention_mask
    
      labels = torch.cat(labels, 0) # (batch, length)
    
      relevance = torch.cat([torch.tensor([r]) for r in relevance], 0) # (batch)
      rows.append([batch_dict, labels, relevance])


  batch_df = pd.DataFrame(rows, columns=["Sentence", "Tags", "Relevance"])
  return batch_df
