pythonseq2seq/train.py --train /biased.word.train --test /biased.word.test --pretrain_data /unbiased --extra_features_top --pre_enrich --activation_hidden --tagging_pretrain_epochs 3 --bert_full_embeddings --debias_weight 1.3 --token_softmax --pointer_generator --coverage --working_dir seq2seq/
