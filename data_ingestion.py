"""
Load the dataset from ArnetMiner v8 format to MongoDb.
Save the dataset from MongoDb to ArnetMiner v8 format.
"""

import logging

from pymongo import MongoClient

import config

# open the mongo connection
client = MongoClient(config.__host, config.__port, connect=True)
papers_collection = client[config.__db_name][config.__collection_name]

def ingest_dataset(dataset_path, storage_enabled=True):
    """
    Note: the first line of a paper description should be the title!!

    :param print_enabled: boolean, True to enable print of parsed papers, False otherwise
    :param storage_enabled: boolean, True to save parsed papers on db (mongo), False to return a collection
    :param dataset_path: the path of the dataset file to process
    :return:
    """
    with open(dataset_path) as dataset_file:

        processed_papers = 0

        papers = []
        paper = None
        line = dataset_file.readline()
        while line != "":
            multiline = False
            if line != None and len(line) != 0 and line[0] == '#':
                code = line[0:2]
                value = line[2:].strip()
                if "#*" == code:
                    # save previous paper
                    if paper is not None:
                        # clean venue
                        if 'venue' in paper:
                            paper['cleaned_venue'] = clean_venue(paper['venue'])

                        if storage_enabled:
                            papers_collection.insert(paper)
                        else:
                            papers.append(paper)

                        processed_papers += 1
                        if processed_papers % 10000 == 0:
                            print("Processed {0} papers".format(processed_papers))
                            logging.debug(paper)
                            break

                    # create new one
                    paper = {}
                    paper['acm_citations'] = []
                    paper['authors'] = []

                    paper['title'] = value
                elif "#@" == code:
                    paper['authors'] = split_and_clean_authors(value)
                elif "#t" == code:
                    paper['year'] = int(value)
                elif "#c" == code:
                    paper['venue'] = value
                elif "#i" == code:
                    paper['acm_id'] = value[4:]
                elif "#%" == code:
                    paper['acm_citations'].append(value)
                elif "#!" == code:
                    line = dataset_file.readline()
                    # handle multiline abstract
                    while line != "" and len(line.strip()) != 0 and line[0] != '#':
                        value += " " + line
                        line = dataset_file.readline()
                    paper['abstract'] = value
                    multiline = True
            else:
                # ignore
                if line.strip() != "":
                    logging.warning("[WARNING] Ignored line. Content '{0}'".format(line))

            if not multiline: line = dataset_file.readline()

    logging.info("[INFO] Inserted documents: {0}".format(processed_papers))

    if not storage_enabled:
        return papers


def split_and_clean_authors(authors_string):
    return [a.strip() for a in authors_string.replace('. ', '.').replace('.', '. ').split(',')]


def clean_venue(venue):
    """
    Venues cleaning procedure for dblp dataset.

    :param venue:
    :return:
    """

    return venue.lower().strip('?:!.,;- ')


def to_v8_format(paper):
    """
    Transform a paper dictionary to its string representation, following the ArnetMiner V8 format. 
    For more details on the format refer to: https://aminer.org/citation

    :param paper: the paper to convert
    :return: the string representation
    """
    output_string = ''
    if 'title' in paper:
        output_string += '#*' + paper['title'] + '\n'

    if 'authors' in paper:
        output_string += '#@' + ', '.join(paper['authors']) + '\n'

    if 'year' in paper:
        output_string += '#t' + str(paper['year']) + '\n'

    if 'venue' in paper:
        output_string += '#c' + paper['venue'] + '\n'

    if 'acm_id' in paper:
        output_string += '#index' + str(paper['acm_id']) + '\n'

    if 'acm_citations' in paper:
        for cit in paper['acm_citations']:
            output_string += '#%' + str(cit) + '\n'

    if 'abstract' in paper:
        output_string += '#!' + paper['abstract'] + '\n'

    return output_string


def dump_dataset_to_file(output_filename):
    """
    Save the dataset contained in papers_collection to file, following the ArnetMiner V8 format. 
    For more details on the format refer to: https://aminer.org/citation

    :param output_filename: the path of the output file
    :return: 
    """
    with open(output_filename, 'w') as out:
        papers = papers_collection.find()
        total_number = papers_collection.count()
        processed = 0
        for p in papers:
            processed += 1
            out.write(to_v8_format(p))
            out.write('\n')

            if processed % 10000 == 0:
                print('Processed {0} of {1}'.format(processed, total_number))


if __name__ == '__main__':
    logging.basicConfig(filename='latest_execution.log', level=logging.INFO)

    # to ingest a dataset in ArnetMiner v8 format
    ingest_dataset(config.__input_dataset_dump)

    # to store the dataset in the ArnetMiner v8 format
    # output_filename = 'path/to/filename'
    # dump_dataset_to_file(output_filename)

