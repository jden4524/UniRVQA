import argparse, os
from misc.util import read_csv
from colbert import Indexer
from colbert.infra import ColBERTConfig

parser = argparse.ArgumentParser(description='Create index for corpus')
# parser.add_argument('-m', '--model', help='blip2 or instructblip', default='blip2')
parser.add_argument('-c','--checkpoint', help='Checkpoint name')
parser.add_argument('-d','--dataset', help='Dataset used for index creation')
args = vars(parser.parse_args())
model_name = args['checkpoint'].split('_')[0]

corpus_path = 'assets/data/okvqa/google_corpus/okvqa_full_clean_corpus.csv' if args['dataset'] == 'okvqa' else 'assets/data/infoseek/wiki_100k_short.csv'
# load models

config = ColBERTConfig(
            root="assets/experiments",
            experiment=".",
            nbits=4,
            doc_maxlen=512,
            kmeans_niters=16,
            index_bsize=32,
            model_name=f"Salesforce/{model_name}-flan-t5-xl",
            dim = 32
        )


if __name__ == "__main__":
    document_collection = read_csv(corpus_path, header=True)
    indexer = Indexer(checkpoint=f"assets/checkpoints/{args['checkpoint']}", config=config)
    print('Loaded indexer')
    indexer.index(
            name=f"{args['dataset']}_index_{args['checkpoint']}",
            collection=[doc[1] for doc in document_collection],
            overwrite=True,
        )
    index_path = indexer.get_index()