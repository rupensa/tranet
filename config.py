#############################################
# MongoDB server
#############################################

# The host of the mongodb server (after ssh tunnel, if necessary)
__host = 'localhost'

# The port to connect to access mongodb server / default 27017
__port = 27014

# Mongo database name
__db_name = 'scipub'

# Mongo collection name
__collection_name = 'papers'

#############################################
# Folders and files
#############################################

# Path of the dataset file to ingest
__input_dataset_dump = 'path/to/merged-dataset-v8.txt'

# folder within write outputs
__outputs_folder_path = '/path/to/outputs/folder'

# resources folder
__resources_folder_path = 'path/to/utilities/and/resources'