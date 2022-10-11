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
    train_precision, train_recall, train_loss = [], [], []
    train_retrieved, train_to_retrieve = [], []
    
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
                                    token_type_ids = train_input_type.long())                  # (batch, max_seq_length, 1)
  
      ret, _ = retriever(output, token_train_label, train_input_type, train_input_ids)
      loss = criterion(output.squeeze(), token_train_label.double())
      # mask loss: the head, and special tokens corresponding loss should not affect the wight update
      loss = torch.where(train_input_type == 0, loss, torch.tensor(0, dtype=torch.double).to(device))
      # weighted loss
      loss = torch.where(token_train_label == 1, weight[1]*loss, loss)
      loss = torch.where(token_train_label == 0, weight[0]*loss, loss)
    
      avg_loss = torch.tensor([0]).to(device)
      for l in loss:
        avg_loss = torch.add(avg_loss, l[l.nonzero()].mean())
      
      
      loss = avg_loss/loss.size()[0]
      
      
      
      if len(ret)==0:
        print('It stopped retrieving!!')
      for token in ret:
        if token not in train_retrieved:
          train_retrieved.append(token)
      
      loss.backward()
      optimizer.step()
      step+=1
      
      train_loss.append(loss.item())

    
    
    # ========================================
    #               Validation
    # ========================================
    print("")
    print("Running Validation...")

    t0 = time.time()
    model.eval()
    test_precision, test_recall, test_loss = [], [], []
    test_retrieved, test_to_retrieve = [], []
    y_true, probas_pred = [], []
    
    for i, v in test_data.items():
      if i%30==0 and i!=0:
        elapsed = format_time(time.time() - t0)
        print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(i, len(test_data), elapsed))

      test_input_ids = test_data[i]['input_ids'].to(device)                      # (batch, max_seq_legth)
      test_input_mask = test_data[i]['attention_mask'].to(device)                # (batch, max_seq_legth)
      test_input_type = test_data[i]['token_type_ids'].to(device)                # (batch, max_seq_legth)
      token_test_label = test_labels['Tags'][i].to(device)                       # (batch, max_seq_legth)
      
      with torch.no_grad():   
        if crf == 'off':
          output, _ = model(test_input_ids.long(), 
                          attention_mask = test_input_mask.long(), 
                          token_type_ids = test_input_type.long())               # (batch, max_seq_length, 1)

          y_true.extend(torch.flatten(token_test_label).tolist())
          probas_pred.extend(torch.flatten(output.squeeze(-1)).tolist())

          test_ret, _ = retriever(output, token_test_label, test_input_type, test_input_ids, val = True)

          loss = criterion(output.squeeze(), token_test_label.double()) 
          loss = torch.where(test_input_type == 0, loss, torch.tensor(0, dtype=torch.double).to(device))
          loss = torch.where(token_test_label == 1, weight[1]*loss, loss)

          avg_loss = torch.tensor([0]).to(device)
          for l in loss:
            avg_loss = torch.add(avg_loss, l[l.nonzero()].mean())
      
          loss = avg_loss/loss.size()[0]
        
        else:
          scores = model.forwardCRF(test_input_ids.long(), 
                                    attention_mask = test_input_mask.long(), 
                                    token_type_ids = test_input_type.long())
        
          loss = -model.crf(scores.long(), 
                             token_test_label.long())
          prediction = model.crf.decode(scores.long(), 
                                        test_input_mask.bool())
      
      
          test_ret = retrieverCRF(prediction, token_test_label, test_input_type, test_input_ids)
        
        if len(test_ret)==0:
          print('It stopped retrieving!!')
        
        for token in test_ret:
          if token not in test_retrieved:
            test_retrieved.append(token)
        
        test_loss.append(loss.item())
    
    
    validation_time = format_time(time.time() - t0)
    print("")
    print("  Validation epoch took: {:}".format(validation_time))
    print("")
    
    assert len(y_true)==len(probas_pred)
    precision, recall, thresholds = precision_recall_curve(np.asarray(y_true), 
                                                           np.asarray(probas_pred))
    fpr, tpr, tt = roc_curve(np.asarray(y_true), np.asarray(probas_pred))
    fig1, ax1 = plt.subplots(1, figsize=(10,10))
    ax1.plot(fpr, tpr, marker='.', markersize=1, linewidth=2, label='Our model')
    ax1.plot([0, 1], [0,1], linestyle='--',linewidth=2, label='No skill')
    ax1.set_xlabel('fpr')#, fontsize=25)
    ax1.set_ylabel('tpr')#, fontsize=25)
    ax1.yaxis.set_tick_params()#labelsize=23)
    ax1.xaxis.set_tick_params()#labelsize=23)
    ax1.legend()#fontsize=23)
    plt.show()
    assert monotonic(recall)
    thresholds = np.append(thresholds, thresholds[-1]+1) 
    auc_PR.append((precision, recall, thresholds))
    fig, axs = plt.subplots(2, figsize=(10,10))
    no_skill = y_true.count(1)/len(y_true)
    print('no_skill', no_skill)
    print('precision[0]', precision[0])
    
    axs[0].plot(recall, precision, marker='.', markersize=1, label='Our model')
    axs[0].plot([0, 1], [no_skill, no_skill], linestyle='--', label='No skill')
    axs[0].set_xlabel('Recall')
    axs[0].set_ylabel('Precision')
    axs[0].legend()
    # BASELINES
    print(epoch_i)
    
    ext_precision, ext_recall, ext_thresholds = extrapolator(precision, recall, thresholds)
    ext_fpr, ext_tpr, _ = extrapolator(fpr, tpr, tt)
    
    axs[1].plot(ext_recall, ext_precision, marker='.', markersize=1, label='Our model')
    axs[1].plot([0, 1], [no_skill, no_skill], linestyle='--', label='No skill')
    axs[1].set_xlabel('Recall')
    axs[1].set_ylabel('Precision')
    axs[1].legend()
    plt.show()
    model_auc = auc(recall, precision)
    roc_auc = auc(fpr, tpr)
    print('\nModel: auc=%.3f\n' % (model_auc))
    print('\nModel: auc=%.3f\n' % (roc_auc))


    if epoch_i == 19:
      print(len(ext_precision), len(ext_recall), len(thresholds), len(ext_fpr), len(ext_tpr))
      data_to_plot = {'recall': ext_recall.tolist(), 
                      'precision': ext_precision.tolist(),
                      'thresholds': ext_thresholds.tolist(), 
                      'fpr':fpr.tolist(), 
                      'tpr':tpr.tolist(), 
                      'pr_auc': model_auc, 
                      'roc_auc':roc_auc}
      
      with open('/content/drive/MyDrive/University/Thesis Project/data_to_plotBERTdf5.json', 'w') as outfile:
        json.dump(data_to_plot, outfile)
    # ======================================
    #           DIAGNOSTIC ANALYSIS
    # ======================================
    t0 = time.time()
    print("")
    print("Running Diagnostic Analysis...")
    # Pass all the values to Weight&Biases after every epoch
    
    train_loss = sum(train_loss)/len(train_loss)
    test_loss = sum(test_loss)/len(test_loss)
    
    eval_train_retrieved = list(nlp.pipe(train_retrieved)) 
    #to_test_train = list(set([' '.join([t.lemma_ for t in doc]) for doc in eval_train_retrieved]))
    to_test_train = lemmarizeListofStrings(train_retrieved)
    train_p, train_r, train_TP, train_FP, _ = evaluate_headings(omh_train, to_test_train)
    global_train_p, global_train_r, glob_TP, glob_FP, _ = evaluate_headings(global_omh, to_test_train)

    print('===================================== DIAGNOSTIC ANALYSIS =====================================\n')
    print('{}Training...{}'.format(color.BOLD, color.END))
    print('Retreived [{}]: \n{}'.format(len(eval_train_retrieved), sorted([t.text for t in eval_train_retrieved]))) 
    print('Retreived [{}]: \n{}'.format(len(train_retrieved), sorted([t for t in train_retrieved]))) 
    print('Exact prediction [{}]: \n{}'.format(len(train_TP), sorted(train_TP)))
    print('Wrong prediction [{}]: \n{} '.format(len(train_FP), sorted(train_FP)))
    print('\nGLOBAL') 
    print('Exact prediction [{}]: \n{}'.format(len(glob_TP), sorted(glob_TP)))
    print('Wrong prediction [{}]: \n{} '.format(len(glob_FP), sorted(glob_FP)))    
    print('-'*100)

    # POST-FILTERS
    flat_test_retrieved = ' '.join(test_retrieved)
    flat_test_retrieved = list(set(nltk.wordpunct_tokenize(flat_test_retrieved)))
    unk = list(spell.unknown(flat_test_retrieved))
    unk.extend(unknown_words)
    print('UNK words: ', unk)
    docs = list(nlp.pipe(test_retrieved))
    print('N° before filtering: ',len(test_retrieved))
    """eval_test_retrieved = stemList(test_retrieved)"""
    
    truncated, wrong_format, unk_word, adj = [], [], [], []
    eval_test_retrieved2 = []
    
    for z in docs:
      c1 = len(z)==1 and z[0].pos_ in ['ADP', 'INTj', 'X', 'CCONJ', 'PUNCT', 'INTJ', 'PRON', 'SCONJ', 'NUM', 'ADV']
      c2 = len(z)==2 and (z[0].dep_, z[1].dep_) in [('det', 'ROOT'), ('ROOT', 'pobj'), ('ROOT', 'prep'), (('ROOT', 'punct'), ('punct', 'ROOT'), ('poss', 'ROOT'))]
      c3 = ';' in z.text or '.' in z.text or '(' in z.text or ',' in z.text or ')' in z.text or '/' in z.text
      if z.text[0] in ['#']: 
        truncated.append(z.text)
        continue
      elif z[0].pos_ in ['CCONJ', 'PUNCT', 'DET', 'IN', 'ADP'] or z[-1].pos_ in ['CCONJ', 'PUNCT', 'DET', 'IN', 'ADP']:
        truncated.append(z.text)
        continue
      elif c1 or c2 or c3: 
        wrong_format.append((z.text, [i.pos_ for i in z]))
        continue
      elif len(set(nltk.wordpunct_tokenize(z.text))-set(unk))!=len(set(nltk.wordpunct_tokenize(z.text))):
        unk_word.append(z.text)
        print(z.text)
        continue
      elif len(z)==1 and (z[0].tag_=='JJ' or z[0].pos_=='ADJ'):
        adj.append(z.text)
        continue
      else:
        eval_test_retrieved2.append(z)
    # remuve those mono-terms that appear in a truncated term
    
    eval_test_retrieved = list(set(eval_test_retrieved2))
    to_remuve = []
    for z1 in eval_test_retrieved:
      for z2 in eval_test_retrieved:
        if len(z1) ==1 and z1.text in z2.text.split() and z1.text != z2.text:
          truncated.append(z1.text)
          if z1.text not in mono_to_NOT_remember:
            mono_to_NOT_remember.append(z1.text)
        if len(z1)>1 and all(elem in z2.text.split() for elem in z1.text.split()) and z1.text != z2.text:
          if z1.text not in multi_to_NOT_remember and z1[-1].tag_=='JJ':
            multi_to_NOT_remember.append(z1.text)

    print('Do not include:',len(mono_to_NOT_remember), sorted(mono_to_NOT_remember))
    print('...also do not include:',len(multi_to_NOT_remember), sorted(multi_to_NOT_remember))
    eval_test_retrieved = [x for x in eval_test_retrieved if x.text not in mono_to_NOT_remember and x.text not in multi_to_NOT_remember]
    
    print('\nTruncated [{}]: {}'.format(len(list(set(truncated))), list(set(truncated))))
    print('Wrong formatting [{}]: {}'.format(len(wrong_format), wrong_format))
    print('Unknounw wording [{}]: {}'.format(len(unk_word), unk_word))
    print('Adjectives [{}]: {}'.format(len(adj), adj))
    
    # Retrieveing POS and Dependencies classes for mono- and bi- word terms respectively
    
    POS_1 = [doc[0].pos_ for doc in eval_test_retrieved if len(doc)==1]
    DEP_2 = [(doc[0].dep_, doc[1].dep_) for doc in eval_test_retrieved if len(doc)==2]
    """eval_test_to_retreive = list(set(stemList(test_to_retreive)))"""
    print('N° after filtering: ',len(eval_test_retrieved))
     
    to_test_test = list(set([' '.join([t.lemma_ for t in doc]) for doc in eval_test_retrieved]))
    test_p, test_r, test_TP, test_FP, not_found = evaluate_headings(omh, to_test_test)
    global_test_p, global_test_r, glob_TP, glob_FP, _ = evaluate_headings(global_omh, to_test_test)
    mono_glob_FP = [fp for fp in glob_FP if len(fp.split())==1]
    mono_glob_TP = [tp for tp in glob_TP if len(tp.split())==1]

    print('\n{}Testing...{}'.format(color.BOLD, color.END))
    print('Retreived [{}]: \n{}'.format(len(eval_test_retrieved), sorted([t.text for t in eval_test_retrieved])))
    print('Lemmarized retreived [{}]: \n{}'.format(len(to_test_test), sorted(to_test_test)))
    print('Exact prediction [{}]: \n{}'.format(len(test_TP), sorted(test_TP)))
    print('Wrong prediction [{}]: \n{}'.format(len(test_FP), sorted(test_FP)))
    print('Not found [{}]: \n{}'.format(len(not_found), sorted(not_found, key=lambda x:x[0])))
    epoch_extracted = {'epoch': epoch_i, 
                       'comment': 'Shuffled dataset TRUE Test Modern_Mathematical_Statistics_With_Applications distillBert full update WITH LSTM with global tagging and 7 training books',
                       'num': len(to_test_test),
                       'N° wrong': len(test_FP),
                       'N° mono wrong': len(mono_glob_FP),
                       'mono wrong': mono_glob_FP,
                       'precision': global_test_p*100, 
                       'extracted': sorted([t.text for t in eval_test_retrieved]), 
                       'lemma extracted': sorted(to_test_test)}
  
    extracted_terms.append(epoch_extracted)

    print('\nGLOBAL')
    print('Exact prediction [{}]: \n{}'.format(len(glob_TP), sorted(glob_TP)))
    print('Wrong prediction [{}]: \n{}'.format(len(glob_FP), sorted(glob_FP)))
    #print('Not found [{}]: \n{}'.format(len(glob_not_found), sorted(glob_not_found, key=lambda x:x[0])))
    
    print('Mono wrong prediction [{}]: \n{} '.format(len(mono_glob_FP), sorted(mono_glob_FP)))
    print('Mono correct prediction [{}]: \n{} '.format(len(mono_glob_TP), sorted(mono_glob_TP)))

    diagnostic_time = format_time(time.time() - t0)
    print("")
    print("  Diagnostic Analysis took: {:}".format(diagnostic_time))
    print("")

    print('\n\n======================== REPORT ========================')
    print('|')
    print('|{}Train Loss: {:.3f}'.format(' '*5, train_loss))
    print('|{}Test Loss: {:.3f}'.format(' '*5,test_loss))
    print('|')
    print('|{}Local Train Precision: {:.1f}'.format(' '*5,train_p*100))
    print('|{}Local Train Recall: {:.1f}'.format(' '*5,train_r*100))
    print('|{}Local Val. Precision: {:.1f}'.format(' '*5,test_p*100))
    print('|{}{}Local Val. Recall: {:.1f}{}'.format(' '*5,color.GREEN,test_r*100, color.END))
    print('|')
    print('|{}Global Train Precision: {:.1f}'.format(' '*5,global_train_p*100))
    print('|{}Global Train Recall: {:.1f}'.format(' '*5,global_train_r*100))
    print('|{}{}Global  Val. Precision: {:.1f}{}'.format(' '*5, color.GREEN, global_test_p*100, color.END))
    print('|{}Global Val. Recall: {:.1f}'.format(' '*5,global_test_r*100))
    print('========================================================')
    print('\n NB: A drop in precision after the 10th epoch might indicate MORE novel terms!')
    

    print('\n\n====================== GRAPHS ======================')
    print(f"{'POS':{12}} {'N°':{8}} {'100%':{5}}")
    for k in Counter(POS_1).keys():
      print(f"{k:5} {Counter(POS_1)[k]:8} {round(Counter(POS_1)[k]/sum(Counter(POS_1).values()), 2):10}")
    print('')
    print(f"{'DEP':{25}} {'N°':{8}} {'100%':{5}}")
    for k in Counter(DEP_2).keys():
      print('{}{}{}{}{}'.format(k, ' '*10, Counter(DEP_2)[k], ' '*5,round(Counter(DEP_2)[k]/sum(Counter(DEP_2).values()), 2)))
    print('\n\n\n')

    
    print("Mono-word outliers:\n", [(doc[0].pos_, doc.text) for doc in eval_test_retrieved if len(doc)==1 and doc[0].pos_ not in ['NOUN', 'ADJ', 'VERB', 'PROPN']])
    print("Bi-word outliers:\n",[(doc[0].dep_, doc[1].dep_, doc.text) for doc in eval_test_retrieved if len(doc)==2 and (doc[0].dep_, doc[1].dep_) not in [('amod', 'ROOT'), ('compound', 'ROOT'), ('nsubj', 'ROOT'), ('ROOT', 'punct')]])

    
    if False == True:
      table = wandb.Table(columns=["Text"])
      table.add_data("BERT BASELINE")
      wandb.log({"Description": table,
                 "AUC": model_auc,
                 "Train Loss": train_loss, 
                 'Val. Loss': test_loss,
                 '(Local) Train Precision': train_p,
                 '(Local) Val. Precision': test_p,
                 '(Local) Train Recall': train_r,
                 '(Local) Val. Recall': test_r,
                 '(Global) Train Precision': global_train_p,
                 '(Global) Val. Precision': global_test_p,
                 'N° of predictions (train)': len(to_test_train),
                 'N° of correct predictions (train)': len(train_TP),
                 'N° of predictions (test)': len(to_test_test),
                 'N° of correct predictions (test)': len(test_TP)
                 })
      
      wandb.watch(model)
  
  
  
  for i, (p, r, t) in enumerate(auc_PR):
    auc_PR[i] = (extrapolator(p, r, t))
  
  auc_precision = [sum(i)/epochs for i in zip(*[x[0] for x in auc_PR])]
  auc_recall = [sum(i)/epochs for i in zip(*[x[1] for x in auc_PR])]
  auc_threshold = [sum(i)/epochs for i in zip(*[x[2] for x in auc_PR])]
  fscore = (2 * np.asarray(auc_precision) * np.asarray(auc_recall)) / (np.asarray(auc_precision) + np.asarray(auc_recall))
  ix = np.argmax(fscore)
  print('Best Threshold=%f, F-Score=%.3f' % (auc_threshold[ix], fscore[ix]))
  
  plt.figure(figsize=(10, 7))
  plt.plot(auc_recall, auc_precision, marker='.', markersize=1, label='Our model')
  plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No skill')
  plt.scatter(auc_recall[ix], auc_precision[ix], marker='o', color='black', label='Best')
  plt.xlabel('Recall')
  plt.ylabel('Precision')
  plt.title("Average Precision-Recall curve")
  plt.legend()
  plt.show()
  if False == True:
    auc_recall = auc_recall + [0]*10 
    auc_precision = auc_precision + [1]*10
    for i in range(len(auc_recall)):
      wandb.log({"AUC Recall": auc_recall[i],
                 "AUC Precision": auc_precision[i], 
                 "Baseline": no_skill})
    wandb.log({"Optimal F1-score": fscore[ix], 
               "Optimal Precision": auc_precision[ix], 
               "Optimal Recall": auc_recall[ix], 
               "Optimal Threshold": auc_threshold[ix]})
  if False == True:
    now = datetime.datetime.now()
    with open('/content/drive/MyDrive/University/Thesis Project/Extracted Terms/'+str(now.ctime())+'.json', 'w') as outfile:
      json.dump(extracted_terms, outfile)
  print("")
  print("Training complete!")
  print("--- %s mins ---" % format_time(time.time() - start_time))
