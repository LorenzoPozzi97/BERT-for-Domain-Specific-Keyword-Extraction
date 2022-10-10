class CONFIG: 
    seed = 1
    train_batch = 32
    threshold = 2
    epochs = 20
    learning_rate = 1e-4
    tk = BertTokenizer.from_pretrained('bert-base-uncased')
    test = ['Introductory_Statistics_With_R']
    train =['A_modern_introduction_to_probability',
            'Probability_and_statistics_for_engineers_and_scientists', 
            'A_Concise_Guide_to_Statistics', 
            'Openintro_Statistics',
            'Statistics_and_Probability_Theory',
            'Statistics_for_Non_Statisticians',
            'Statistics_for_Scientists_and_Engineers',
            'Modern_Mathematical_Statistics_With_Applications']
