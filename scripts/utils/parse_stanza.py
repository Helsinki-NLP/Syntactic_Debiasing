import stanza

stanza.download('en')

data_path = '/users/celikkan/Dropbox/WorkHNLP/FoTran/Github/Syntactic_Debiasing/data/RNN-Priming-short'
act_output_file = open(data_path + '/RNN-Priming-short-1000.active.pos.parse.conll', 'w')
pass_output_file = open(data_path + '/RNN-Priming-short-1000.passive.pos.parse.conll', 'w')

orc_file = open(data_path + '/RNN-1-1-1000_active.txt')
prc_file = open(data_path + '/RNN-1-1-1000_passive.txt')

nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse')

for line in orc_file.readlines():
    doc = nlp(line)
    act_output_file.write('\n'.join([f'{word.id}\t{word.text}\t_\t_\t_\t{word.head}\t{word.deprel}' for sent in doc.sentences for word in sent.words])+'\n\n')
    act_output_file.flush()

for line in prc_file.readlines():
    doc = nlp(line)
    pass_output_file.write('\n'.join([f'{word.id}\t{word.text}\t_\t_\t_\t{word.head}\t{word.deprel}' for sent in doc.sentences for word in sent.words])+'\n\n')       
    pass_output_file.flush()

