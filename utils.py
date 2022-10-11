def set_seed(seed = CONFIG.seed):
    torch.manual_seed(seed)

def dataBatcher(df, BATCH_SIZE):
  rows = []
  for i in range(BATCH_SIZE, len(df), BATCH_SIZE): # skip the first iteration
    input_ids, attention_mask, token_type_ids, labels = [], [], [], []

    sentences = df['Enc Sentence'].iloc[i-BATCH_SIZE:i]
    tags = df['Enc Tags'].iloc[i-BATCH_SIZE:i]
    relevance = df['Relevance'].iloc[i-BATCH_SIZE:i]
    head = df['Enc Heading'].iloc[i-BATCH_SIZE:i]
    
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
        type_ones = type_ones[:512-torch.cat((type_zeros, type_ones), 0).size()[0]]
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

    if i+BATCH_SIZE > len(df):
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


def stemList(to_stem):
  stemmed = []
  for phrase in to_stem:
    tokenAux=""
    textAux=""
    tokens = nltk.wordpunct_tokenize(phrase)
    for token in tokens:
      tokenAux = token
      tokenAux = PorterStemmer().stem(token)    
      textAux = textAux + " " + tokenAux
    stemmed.append(textAux)
  return(stemmed)

def singleRetriever(binary_vector, input_id):
  # x keep track of the previous iteration
  previous_iteration = 0
  retrieved = []
  for i, binary_value in enumerate(binary_vector):
    
    # if there is a relevant term in the prediction
    if binary_value ==1:
      # convert ids to tokens from the original BERT tokens
      # unsqueeze(0) because can't be 0-d
      token = CONFIG.tk.convert_ids_to_tokens(input_id[i].unsqueeze(0))
      
      # reconstruct split tokens
      if len(retrieved)>0 and i == previous_iteration+1 and token[0].startswith('##'):
        retrieved[-1] += token[0][2:]
      # join togetehr subsequent tokens to form a unique keyword
      elif len(retrieved)>0 and i == previous_iteration+1 and not token[0].startswith('##'):
        retrieved[-1] += ' ' + token[0].lower()
      else:
        retrieved.append(token[0].lower())
      previous_iteration = i
  
  return sorted(list(set(retrieved)))


def non_increasing(L):
    return all(x>=y for x, y in zip(L, L[1:]))

def non_decreasing(L):
    return all(x<=y for x, y in zip(L, L[1:]))

def monotonic(L):
    return non_increasing(L) or non_decreasing(L)

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def retriever(prediction, target, toke_type_ids, input_ids, val = False):
  retrieved, to_retreive = [], []

  prediction = prediction.squeeze()
  target = target.squeeze()                                                      # (batch, max_seq_length)
                                                      
  # iterate through batches
  for i in range(prediction.size()[0]):
    single_pred = prediction[i].squeeze()                                            # (max_seq_length)
    single_targ = target[i].squeeze()                                            # (max_seq_length)
    if not any(single_pred) and not any(single_targ):
      # if there are no 1s in prediction and ground-proof skip iteration
      continue   

    # remuve heads, [CLS]s, and [PAD]s
    until_here = single_pred.size()[-1] - torch.count_nonzero(toke_type_ids[i])-1
    single_pred = single_pred[1:until_here]                                      # (real_length)
    #single_targ = single_targ[1:until_here]                                      # (real_length)
    input_id = input_ids[i][1:until_here]                                        # (real_length)
    binary_single_pred = torch.where(single_pred > CONFIG.threshold, 1, 0)
    a = singleRetriever(binary_single_pred, input_id)
    retrieved.extend(a)
    # x keep track of the previous iteration
  
  retrieved = sorted(list(set(retrieved)))
  return retrieved, _
