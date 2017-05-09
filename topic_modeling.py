"""
Main functions: 

* compute_topic_model: compute a topic model and store it on disk.
* assign_topics_and_save: assign topics to papers and store the result in csv format.

"""
import json
import logging
import operator
import os
import sys
import time
from multiprocessing import Pool

import numpy
import wikipedia
from PIL import Image
from gensim import matutils
from gensim.models import LdaModel
from pymongo import MongoClient
from scipy import sparse
from wordcloud import WordCloud

import config
from utilities import selected_venues as sv
from utilities import utilities

client = MongoClient(config.__host, config.__port)
papers_collection = client[config.__db_name][config.__collection_name]

def extract_data_from_mongo(year_from=2000, year_to=2014, venues_filter=None, concat_title_and_abstract=True, limit=None, id_field='acm_id'):
    # Paper extraction query
    query = {'$and': [
        {'abstract': {'$ne': None}},
        {'year': {'$gte': year_from}},
        {'year': {'$lte': year_to}}
    ]}

    if venues_filter is not None:
        query['$and'].append({'cleaned_venue': {'$in': venues_filter}})

    query_result = papers_collection.find(query, {id_field: 1, 'abstract': 1, 'title': 1})

    n_results = papers_collection.count(query)

    if limit is not None:
        query_result.limit(limit)
        n_results = min(n_results, limit)

    # print("No. of papers: {0}".format(n_results))

    if n_results != 0:
        paper_dataset_ids = [''] * n_results
        paper_db_ids = [''] * n_results
        paper_abstracts = [''] * n_results
        i = 0

        for res in query_result:
            paper_dataset_ids[i] = res[id_field]
            paper_db_ids[i] = res['_id']
            if concat_title_and_abstract:
                paper_abstracts[i] = res['title'] + " " + res['abstract']
            else:
                paper_abstracts[i] = res['abstract']

            i += 1

        return paper_db_ids, paper_dataset_ids, paper_abstracts
    else:
        return [], [], []


def get_gensim_representation_new_docs_for_inference(texts, current_id2word, use_lemmer=True, min_df=2, max_df=0.8,
                                                     tf_matrix_dump_filename=None):
    """
    Takes in input a list of texts and returns their bag of word representation, removing stopwords

    :param texts:
    :param use_lemmer:
    :return:
    """

    if tf_matrix_dump_filename is not None and os.path.exists(tf_matrix_dump_filename+'-features.json'):
        logging.info('Reading tf matrix from file.')

        matrix = sparse.load_npz(tf_matrix_dump_filename + '-matrix.npz')
        with open(tf_matrix_dump_filename +'-features.json', 'r') as feat_in:
            features = json.loads(feat_in.read())
    else:
        logging.info('Computing tf matrix')
        matrix, features = preprocess_data_tf(texts, use_lemmer=use_lemmer, min_df=min_df, max_df=max_df)

        if tf_matrix_dump_filename is not None:
            logging.info('Dumping tf matrix to file.')

            sparse.save_npz(tf_matrix_dump_filename + '-matrix', matrix)
            with open(tf_matrix_dump_filename + '-features.json', 'w') as feat_out:
                feat_out.write(json.dumps(features))
        else:
            logging.warning('Missing filename, cannot dump the matrix.')

    print('TF Matrix Shape:', matrix.shape)

    word2id = {current_id2word[id]:id for id in current_id2word.keys()}

    if len(features) != 0:
        # transform all documents to the dictionary used by the model
        bow = list(map(lambda i: transform_to_corpus(i, matrix, features, word2id), range(matrix.shape[0])))

    logging.info('Completed computation of bow model.')

    return bow


def transform_to_corpus(doc_index, matrix, features, word2id):

    doc = matrix.getrow(doc_index)
    rows, cols = doc.nonzero()

    transformed_doc = []
    for col in cols:
        if features[col] in word2id.keys():
            transformed_doc.append((word2id[features[col]], matrix[doc_index,col]))

    return transformed_doc


def preprocess_data_tf(data, use_lemmer=True, min_df=2, max_df=0.8):
    stopwords_list = utilities.get_stopwords()
    tf_matrix, tf_features_names = utilities.compute_tf(data, stopwords_list, use_lemmer, min_df=min_df, max_df=max_df)
    return tf_matrix, tf_features_names


def print_output(output_filename, output_data, separator=';'):
    """

    :param output_filename: the output filename
    :param output_data: the array of data to write on file
    :return:
    """
    output_data = [str(x) for x in output_data]
    with open(output_filename, 'a') as out:
        out.write(separator.join(output_data) + '\n')


def execute_lda_gensim(corpus, features, n_topics, out_filename_prefix):
    """

    :param data: tfidf or tf matrix
    :param n_topics: number of topics to extract
    :param out_filepath: the name of the output file
    :return:
    """
    output = []
    start = time.time()

    output.append(len(corpus))
    output.append(len(features))

    output.append('gensim:iterations=50')
    output.append(out_filename_prefix)
    output.append(n_topics)
    id2word = {k: v for k,v in enumerate(features)}

    lda = LdaModel(corpus, id2word=id2word, num_topics=n_topics)

    lda.save(os.path.join(config.__outputs_folder_path, out_filename_prefix))

    end = time.time()
    output.append(end - start)
    print(output)


def get_data_tf(year_from, year_to, venues_filter, use_lemmer=True, min_df=2, max_df=0.8):
    db_ids, cit_ids, abstracts = extract_data_from_mongo(year_from=year_from, year_to=year_to,
                                                         venues_filter=venues_filter)

    return preprocess_data_tf(abstracts, use_lemmer, min_df, max_df)


def compute_topic_model(year_from=1900, year_to=2020, venues_filter=None, n_topics=100, use_lemmer=True,
                        min_df=2, max_df=0.8):
    """
    Compute a topic model considering the following parameters and store it to file.
    
    :param year_from: consider papers published from year_from 
    :param year_to: consider papers published until year_to
    :param venues_filter: list, list of venues to consider 
    :param n_topics: number of topics to search for in the model
    :param use_lemmer: boolean, True to perform lemmatisation, False otherwise
    :param min_df: int or float, min document frequency / document proportion (if float < 1) to consider a term in the model
    :param max_df: int or float, max document frequency / document proportion (if float < 1) to consider a term in the model
    :return: 
    """
    start = time.time()
    out_fileprefix = get_output_fileprefix(year_from, year_to, venues_filter, n_topics)

    corpus, tf_features_names = get_corpus_gensim_for_learning(year_from, year_to, venues_filter, use_lemmer, min_df, max_df)
    execute_lda_gensim(corpus, tf_features_names, n_topics, out_fileprefix)

    end = time.time()
    return year_from, year_to, n_topics, (end - start)


def print_completed(result=None):
    with open(os.path.join(config.__outputs_folder_path, 'workers_pool_status.csv'), 'a') as status:
        status.write('[ENDED], {0} {1} {2}, {3}\n'.format(result[2], result[0], result[1], result[3]))


def get_output_fileprefix(year_from, year_to, venues_filter, n_topics):
    return '-'.join(['lda',
                     str(year_from),
                     str(year_to),
                     str(len(venues_filter)),
                     str(n_topics)])


def load_computed_model(file_prefix_with_path):
    return LdaModel.load(file_prefix_with_path)


def load_saved_model(yfrom, yto, venues, n_topics, output_folder):
    """
    Loads and returns a previously saved gensim LdaModel. 
    """
    file_prefix = get_output_fileprefix(yfrom, yto, venues, n_topics)
    lda = load_computed_model(os.path.join(output_folder, file_prefix))
    return lda


def get_corpus_gensim_for_learning(year_from, year_to, venues_filter, use_lemmer, min_df=2, max_df=0.8):
    tf_matrix, tf_features_name = get_data_tf(year_from, year_to, venues_filter, use_lemmer, min_df, max_df)
    corpus = matutils.Sparse2Corpus(tf_matrix, documents_columns=False)
    return corpus, tf_features_name


def run_pool_of_workers_gensim(min_df=2, max_df=0.7):
    # each LDA execution uses 4 cores so it is necessary to reduce the number of parallel processes
    pool = Pool(processes=14)
    counter = 0
    first_year = 2000
    last_year = 2014
    for n_topics in range(50, 201, 10):
        for year_from in range(first_year, last_year):
            counter += 1
            with open(os.path.join(config.__outputs_folder_path, 'workers_pool_status.csv'), 'a') as status:
                status.write('[QUEUED-{3}], {0} {1} {2}, -\n'.format(n_topics, year_from, year_from + 1, counter))
            pool.apply_async(compute_topic_model, (year_from, year_from + 1, sv.considered_venues, n_topics, True, min_df, max_df),
                             callback=print_completed)

        # run also the global one
        counter += 1
        with open(os.path.join(config.__outputs_folder_path, 'workers_pool_status.csv'), 'a') as status:
            status.write('[QUEUED-{3}], {0} {1} {2}, -\n'.format(n_topics, first_year, last_year, counter))
        pool.apply_async(compute_topic_model, (first_year, last_year, sv.considered_venues, n_topics, True, min_df, max_df),
                         callback=print_completed)

    pool.close()
    pool.join()


def print_topic_assignment(topic_assignment, model=None):
    print('\tTopic importance\tTopic description')
    for i,doc in enumerate(topic_assignment):
        print('------------')
        print('Document {0}'.format(i))
        for a in doc:
            print()
            string_topic = a[0] if model is None else model.print_topic(a[0])
            print('\t{1:2f}\t\t{0}'.format(string_topic, a[1]))


def extract_and_print_topic_assignment_for_docs(model, docs):
    new_docs = get_gensim_representation_new_docs_for_inference(docs, model.id2word,
                                                                                    use_lemmer=True)

    print('Extracted topics')
    print()
    # topic_assignment is a list of len(new_docs) lists of pairs (topic_id, percentage)
    topic_assignment = model[new_docs]
    print_topic_assignment(topic_assignment, model)


def compute_topic_assignment(topic_model, abstracts, min_df=2, max_df=0, tf_matrix_dump_filename=None):
    """
    Computes the topics assignment for each abstract of abstracts list w.r.t the specified topic_model
    :param topic_model: 
    :param abstracts: 
    :return: 
    """
    if max_df == 0:
        max_df = 0.9 if len(abstracts) < 100 else 0.8


    # get the gensim representation of new papers in order to query the topic model
    papers_gensim = get_gensim_representation_new_docs_for_inference(abstracts, topic_model.id2word,
                                                                                use_lemmer=True, min_df=2,
                                                                                max_df=max_df,
                                                                                tf_matrix_dump_filename=tf_matrix_dump_filename)

    # for each paper compute the topics assignment
    topic_assignment = [None] * len(papers_gensim)
    for i, paper in enumerate(papers_gensim):
        topic_assignment[i] = topic_model[paper]

    return topic_assignment


def disambiguate_topic(topic_description, min_word_probability=0.010):
    """
    Try to disambiguate a topic 
    :param topic_description: is a list of pairs  (word, word_probability)
    :param min_word_probability: is the minimum probability for words 
    :return:  
    """
    words = [w for w,p in topic_description if p >= min_word_probability]

    if len(words) == 0:
        # if no words are over the threshold, take the first
        words = [topic_description[0][0]]

    res = wikipedia.search(' '.join(words))
    return res


def print_all_topics(model, num_topics=10, num_words=20, try_to_disambiguate=False,
                     min_word_probabity_for_disambiguation=0.010):
    """
    Print topics from a given LdaModel. 
    
    :param model: 
    :param num_topics: 
    :param num_words: 
    :param try_to_disambiguate: 
    :param min_word_probabity_for_disambiguation: 
    :return: 
    """
    print('Print {0} topics'.format(num_topics))
    print('------------')
    for t in model.show_topics(num_topics=num_topics, num_words=num_words, formatted=False):
        if try_to_disambiguate:
            possible_labels = disambiguate_topic(model.show_topic(t[0]), min_word_probability=min_word_probabity_for_disambiguation)[:2]
            print('{0}:\t{1}\n'.format(t[0], possible_labels))
            print('{0}\n'.format(t[1]))
        else:
            print('{0}:\t{1}\n'.format(t[0], t[1]))


def get_word_frequencies(topic_description):
    """
    Given a topic description, returns the corresponding dictionary with words as keys 
    and frequencies (weight * 1000) as values.   
    :param topic_description: list of pairs (word, word_weight) 
    :return: 
    """
    frequencies = {w:f for w,f in topic_description}
    return frequencies


def generate_wordcloud(topic_description, use_mask='rectangle'):
    """
    Generate and plot a word cloud from a specific topic description (list of word-weight pairs).
    Store it to file. 
    
    :param topic_description: a list of pairs (word, weight)
    :param use_mask: string, 'oval' to use an oval mask, 'rectangle' otherwise
    :return: 
    """

    # transform the topic description in frequencies
    topic_frequencies = get_word_frequencies(topic_description)

    if use_mask == 'oval':
        mask = numpy.array(Image.open(os.path.join(config.__resources_folder_path, "oval.jpg")))
    else:
        mask = numpy.array(Image.open(os.path.join(config.__resources_folder_path, "rect.png")))

    wc = WordCloud(background_color="white", max_words=2000, mask=mask)
    # generate word cloud
    wc.generate_from_frequencies(topic_frequencies)

    # store to file
    wc.to_file(os.path.join(config.__outputs_folder_path, "wordcloud_{0}_{1}.png".format(
        hash(str(topic_description)), use_mask)))



def get_paper_counter_per_topic_id(all_topic_assignments):
    """
    Return a dictionary {topic_id, n_of_papers_assigned}
    :param all_topic_assignments:
    :return:
    """
    counter = {}
    for topic_assignment in all_topic_assignments:
        for topic_index, topic_value in topic_assignment:
            if topic_index not in counter:
                counter[topic_index] = 0

            counter[topic_index] += 1

    return counter


def get_papers_per_topic(topic_model, topic_id_to_consider, year_from, year_to, papers_to_cite=None, debug=False,
                         out=sys.stdout, tf_matrix_dump_filename=None):
    """
    Return the list of papers that belong to the specified topic in the topic_model given as parameter
    and that have been published between ''year_from'' and ''year_to''
    and cite at least one of the acm_ids contained in ''papers_to_cite''.

    if topic_id_to_consider is None, return a counter dictionary with for each topic_id the papers count
    that have been assigned to that topic

    papers_to_cite, list of strings, list of 'acm_id'
    """

    # get papers from mongo that match the filters
    if debug: out.write('Extract papers...\n')
    query = {'year': {'$gte': year_from, '$lte': year_to}, 'cleaned_venue': {'$in': sv.considered_venues},
             'acm_id': {'$ne': None},
             '$and': [{'abstract': {'$ne': None}}, {'abstract': {'$ne': ''}}]}
    select = {'authors': 1, 'acm_id': 1, 'citations': 1, 'title': 1, 'abstract': 1}

    if papers_to_cite == []:
        out.write('[WARNING] The list of papers_to_cite is empty. The query will be empty!\n')

    if papers_to_cite is not None:
        query['citations'] = {'$in': papers_to_cite}

    papers = papers_collection.find(query, select)
    papers = [p for p in papers]

    if debug: out.write('Extracted {0} papers.\n'.format(len(papers)))

    # preprocess the title + abstract (!! keep attention on min_df and max_df parameters,
    # should be 2 and 0.9 respectively in case of a small amount of papers)
    abstracts = [p['title'] + ' ' + p['abstract'] for p in papers]

    if debug: out.write('Compute topics assignments...\n')

    # it's important to update the model dictionary in case of new words in unseen documents
    all_topics_assignments = compute_topic_assignment(topic_model, abstracts,
                                                                       tf_matrix_dump_filename=tf_matrix_dump_filename)

    # print_topic_assignment(all_topics_assignments, topic_model)

    if topic_id_to_consider is not None:
        if debug: out.write('Search for papers that are related to topic #{0}...\n'.format(topic_id_to_consider))

        topic_related_papers = []
        index = 0

        for topic_assignment in all_topics_assignments:
            for topic_index, topic_value in topic_assignment:
                if topic_index == topic_id_to_consider:
                    topic_related_papers.append(papers[index])
            index += 1

        if debug: out.write('Found {0} papers.\n'.format(len(topic_related_papers)))

    else:
        if debug: out.write('Count papers assigned to each topic...\n')
        topic_related_papers = get_paper_counter_per_topic_id(all_topics_assignments)

    return topic_related_papers


def get_paper_count_per_topic(topic_model, start_year, end_year, debug=False):
    """
    Return a list of pairs (topic_id, n_papers) where n_papers is the number of papers associated to that topic
    """
    papers_count = get_papers_per_topic(topic_model, None, start_year, end_year, None, debug=debug)
    return sorted(papers_count.items(), key=operator.itemgetter(1), reverse=True)


def print_top_topics_custom(topic_model, start_year, end_year, n_topics=10, out=sys.stdout, debug=False):
    """
    Print top topics based on the number of documents that are really assigned to the topic

    :param topic_model: 
    :param start_year: 
    :param end_year: 
    :param n_topics: 
    :param out: 
    :param debug: 
    :return: 
    """
    papers_count = get_paper_count_per_topic(topic_model, start_year, end_year, debug)
    topic_ids = []
    out.write('#\ttopic id\t#docs\ttopic\n')
    for i in range(min(n_topics, len(papers_count))):
        topic_id = papers_count[i][0]
        topic_ids.append(topic_id)
        out.write(
            '{0}\t{3}\t\t{1}\t{2}\n\n'.format(i, papers_count[i][1], topic_model.print_topic(topic_id, 30), topic_id))

    return topic_ids


def get_authors_from_papers(papers):
    """
    Given the list of papers return the list of authors without duplicates.
    """
    auth_set = set()
    for p in papers:
        auth_set.update(p['authors'])
    return list(auth_set)


def assign_topics_and_save(training_starting_year, training_ending_year, number_of_topics_to_extract,
                           analysis_starting_year, analysis_ending_year, training_venues=None, analysis_venues=None,
                           debug=True):
    """
    Load the model, retrieve documents for analysis from db, assign topics to documents and produces the following files
    as output: 
    
    * xxx-paper-topicid.csv, a csv containing a line <acm_id;topic_id;topic_weight> for each topic assignment. 
    Papers can be assigned to multiple topics.
    * xxx-topicid-topic.json, a json dump of the topics description in order to easily retrieve the words associated to 
    a specific topic_id
    
    :param training_starting_year: 
    :param training_ending_year: 
    :param number_of_topics_to_extract: 
    :param analysis_starting_year: 
    :param analysis_ending_year: 
    :param training_venues: 
    :param analysis_venues: 
    :param debug: 
    :return: 
    """

    out_filename = '-'.join([str(s) for s in [training_starting_year, training_ending_year, number_of_topics_to_extract,
                                              analysis_starting_year,
                                              analysis_ending_year]]) + '-paper-topicid.csv'

    out_filename_2 = '-'.join(
        [str(s) for s in [training_starting_year, training_ending_year, number_of_topics_to_extract,
                          analysis_starting_year,
                          analysis_ending_year]]) + '-topicid-topic.json'

    out = sys.stdout
    with open(os.path.join(config.__outputs_folder_path, out_filename), 'w') as out_csv:

        if debug: out.write('Load model...\n')

        # compute (or load) the topic model
        topic_model = load_saved_model(training_starting_year, training_ending_year, training_venues,
                                       number_of_topics_to_extract, config.__outputs_folder_path)
        out_csv.write('{0};{1};{2}\n'.format('acm_id', 'topic_id', 'topic_weight'))

        if debug: out.write('Extract papers...\n')
        query = {'year': {'$gte': analysis_starting_year, '$lte': analysis_ending_year},
                 'acm_id': {'$ne': None},
                 '$and': [{'abstract': {'$ne': None}}, {'abstract': {'$ne': ''}}]}

        if analysis_venues is not None:
            query['cleaned_venue'] = {'$in': analysis_venues}

        select = {'authors': 1, 'acm_id': 1, 'citations': 1, 'title': 1, 'abstract': 1}

        papers = papers_collection.find(query, select)
        papers = [p for p in papers]

        if debug: out.write('Extracted {0} papers.\n'.format(len(papers)))

        # preprocess the title + abstract (!! keep attention on min_df and max_df parameters,
        abstracts = [p['title'] + ' ' + p['abstract'] for p in papers]

        if debug: out.write('Compute topics assignments...\n')
        # it's important to update the model dictionary in case of new words in unseen documents
        all_topics_assignments = compute_topic_assignment(topic_model, abstracts)

        # print_topic_assignment(all_topics_assignments, topic_model)

        if debug: out.write('Save topics assignments...\n')
        for doc_index, topic_assignment in enumerate(all_topics_assignments):
            for id, w in topic_assignment:
                out_csv.write('{0};{1};{2}\n'.format(papers[doc_index]['acm_id'], id, w))

    if debug: out.write('Save topic descriptions...\n')

    with open(os.path.join(config.__outputs_folder_path, out_filename_2), 'w') as out_csv:
        # save the topics description dump
        json.dump(topic_model.print_topics(number_of_topics_to_extract), out_csv)

    if debug: out.write('Procedure completed.\n')


if __name__ == '__main__':

    # turn on/off functionalities
    compute_topic_model = True
    compute_paper_assignment = True

    if compute_topic_model:
        #######
        ### TO COMPUTE THE TOPIC MODEL
        #######

        year_from = 2000
        year_to = 2014
        venues = sv.considered_venues
        n_topics = 50
        use_lemmer = True
        min_doc_frequency = 2
        max_doc_frequency = 0.7

        compute_topic_model(year_from, year_to, venues, n_topics, use_lemmer, min_doc_frequency, max_doc_frequency)

    if compute_paper_assignment:
        #######
        ### TO ASSIGN DOCUMENTS TO THE COMPUTED TOPIC MODEL AND STORE THE RESULT TO FILE
        #######

        # parameters for the precomputed model to load
        training_starting_year = 2000
        training_ending_year = 2014
        training_venues = sv.considered_venues
        number_of_topics_to_extract = 50

        # parameters for the analysis
        analysis_starting_year = 2000
        analysis_ending_year = 2014
        analysis_venues = sv.considered_venues

        assign_topics_and_save(training_starting_year, training_ending_year, number_of_topics_to_extract,
                               analysis_starting_year, analysis_ending_year, training_venues=training_venues,
                               analysis_venues=analysis_venues)

