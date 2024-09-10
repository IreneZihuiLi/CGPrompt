import os.path

import nltk
import json
from tqdm import tqdm
import pandas as pd
from umap import UMAP
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import KeyBERTInspired
from sentence_transformers import SentenceTransformer
from nltk.data import find
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from fuzzywuzzy import fuzz


def download_if_needed(package):
    """
    Downloads the specified NLTK package if it is not already downloaded.
    """
    try:
        find(f"corpora/{package}")
    except LookupError:
        nltk.download(package)


def step_01_concept_extraction(texts: list[str],
                               concept_extraction_output_file: str,
                               concept_abstracts_output_file: str,
                               logging: any,
                               stop_words: list[str] = None,
                               config: dict[str, any] = None):
    """
    Step 1: Concept Extraction
    Extracts concepts from the texts. The output is an output file listing all concepts
    and an output json file listing the abstracts per concept.

    :param texts: the texts to extract concepts from
    :param concept_extraction_output_file: the file to write the extracted concepts to
    :param concept_abstracts_output_file: the file to write the abstracts per concept to
    :param logging: the logger
    :param stop_words: the stop words to use
    :param config: the configuration can be provided with the following keys: language, gold_concept_file
    :return: None
    """
    download_if_needed("wordnet")
    download_if_needed("omw-1.4")
    logging.info("Step 1: Starting concept extraction.")

    # set default config values if not provided
    if 'language' not in config:
        config['language'] = "english"
    if 'gold_concept_file' not in config:
        config['gold_concept_file'] = ""

    # create BERTopic Extractor
    # language dependent part
    if config['language'] == "english":
        vectorizer_model = CountVectorizer(ngram_range=(2, 4),
                                           stop_words="english" if stop_words is None else stop_words)
        sentence_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    else:
        logging.info(f"Using language {config['language']}.")
        logging.info("Language not yet supported. Exiting.")
        exit(0)
    # language independent part
    umap_model = UMAP(n_neighbors=20, n_components=50, metric="cosine", min_dist=0.0, random_state=37)
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=False)
    representation_model = KeyBERTInspired()

    topic_model = BERTopic(verbose=True,
                           umap_model=umap_model,
                           ctfidf_model=ctfidf_model,
                           vectorizer_model=vectorizer_model,
                           embedding_model=sentence_model,
                           representation_model=representation_model,
                           nr_topics=50,
                           low_memory=True,
                           calculate_probabilities=False)

    topics, _ = topic_model.fit_transform(texts)
    all_topics = topic_model.get_topics()

    extracted_concepts = []
    for topic_num, keywords in all_topics.items():
        if topic_num != -1:
            topic_keywords = [word for word, value in keywords]
            extracted_concepts.extend(topic_keywords)

    # remove duplicates
    extracted_concepts = list(set(keyword.lower() for keyword in extracted_concepts))

    # write extracted concepts to file
    with open(concept_extraction_output_file, "w") as f:
        for id, concept in enumerate(extracted_concepts, 1):
            f.write(f"{id}|{concept}\n")
    logging.info(f"Concepts written to {concept_extraction_output_file}.")

    lemmatizer = WordNetLemmatizer()

    def singularize_concept(concept):
        words = concept.split()
        singular_words = [lemmatizer.lemmatize(word, wordnet.NOUN) for word in words]
        return ' '.join(singular_words)

    # singularize concepts
    extracted_concept = [singularize_concept(concept) for concept in extracted_concepts]

    df_concepts = pd.DataFrame(extracted_concept, columns=["concept"])
    df_concepts["label"] = 0

    if config['gold_concept_file'] != "":
        if os.path.exists(config['gold_concept_file']):
            gold_concepts = pd.read_csv(config['gold_concept_file'], delimiter="|", header=None)
            gold_concepts = gold_concepts[1].tolist()

            # singularize concepts
            gold_concept = [singularize_concept(concept) for concept in gold_concepts]

            # convert to lowercase
            gold_concept = [concept.lower() for concept in gold_concept]

            df_gold_concepts = pd.DataFrame(gold_concept, columns=["concept"])
            df_gold_concepts["label"] = 1

            df_concepts = pd.concat([df_concepts, df_gold_concepts])
            df_concepts = df_concepts.sort_values(by="label")
        else:
            logging.info(f"Gold concept file {config['gold_concept_file']} not found. Skipping.")

    df_concepts = df_concepts.drop_duplicates(subset="concept", keep="first")

    # reduce the text dataset to only texts containing the concepts
    def filter_abstracts_by_term(term, abstracts, threshold=70):
        filtered_abstracts = []
        for abstract in abstracts:
            if isinstance(abstract, str):
                if fuzz.partial_ratio(term.lower(), abstract.lower()) >= threshold:
                    filtered_abstracts.append(abstract)
        return filtered_abstracts

    concept_abstracts = {}
    for index, row in tqdm(df_concepts.iterrows(), desc="Processing concepts",
                           total=df_concepts.shape[0]):
        concept = row["concept"]
        label = row["label"]
        filtered_abstracts = filter_abstracts_by_term(concept, texts)
        concept_abstracts[concept] = {
            "abstracts": filtered_abstracts,
            "label": label
        }

    with open(concept_abstracts_output_file, 'w', encoding='utf-8') as f:
        json.dump(concept_abstracts, f, ensure_ascii=False, indent=4)
    logging.info(f"Abstracts written to {concept_abstracts_output_file}.")

    logging.info("Step 1: Candidate Triple Extraction completed.")
    logging.info(f"Number of concepts extracted through BERTopic: {len(extracted_concept)}")

    if config['gold_concept_file'] != "":
        label_0_count = sum(1 for details in concept_abstracts.values() if details['label'] == 0)
        logging.info(f"Number of concepts added through BERTopic: {label_0_count}")

    empty_abstracts_count = sum(1 for details in concept_abstracts.values() if not details['abstracts'])
    logging.info(f"Number of concepts with empty filtered_abstracts: {empty_abstracts_count}")
