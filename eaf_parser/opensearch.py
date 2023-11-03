import pprint as pp
import requests
from opensearchpy import OpenSearch
from opensearchpy import helpers
from transformers import AutoTokenizer, AutoModel

host = '10.10.255.202'
port = 8200
user = 'pt.sign.language'
password = '_2@Vu6Z%a#jPv#6'
index_name = 'pt.sign.language'

client = OpenSearch(
    hosts=[{'host': host, 'port': port}],
    http_compress=True,
    http_auth=(user, password),
    use_ssl=True,
    verify_certs=False,
    ssl_assert_hostname=False,
    ssl_show_warn=False
)

index_body = {
    "settings": {
        "index": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "analysis": {  # Define the analyser for the eal files
                "analyzer": {
                    "eal_analyzer": {
                        "tokenizer": "whitespace",
                        "char_filter": ["eal_char_filter"]
                    }
                },
                "char_filter": {
                    "eal_char_filter": {
                        "type": "pattern_replace",
                        "pattern": "(\\w+)=\"([^\"]*)\"",
                        "replacement": "$1=$2"
                    }
                }
            }
        }
    },
    "mappings": {
        "properties": {
            "author": {
                "type": "text"
            },
            "date": {
                "type": "date"
            },
            "relative_media_url": {
                "type": "text"
            },
            "mime_type": {
                "type": "text"
            }
        }
    },
}

if client.indices.exists(index=index_name):
    print('\n----------------------------------------------------------------------------------- INDEX EXISTS')
    resp = client.indices.open(index=index_name)
    print(resp)

    print('\n----------------------------------------------------------------------------------- INDEX SETTINGS')
    settings = client.indices.get_settings(index=index_name)
    pp.pprint(settings)

    print('\n----------------------------------------------------------------------------------- INDEX MAPPINGS')
    mappings = client.indices.get_mapping(index=index_name)
    pp.pprint(mappings)

    print('\n----------------------------------------------------------------------------------- INDEX #DOCs')
    print(client.count(index=index_name))
else:
    print('\nCreating index:')
    response = client.indices.create(index_name, body=index_body)
    print(response)

# Define the document to be indexed
# TODO: read all the available files and index them
# Read the file content
with open("videos/2019/Transcrições_2019/465A/465A.eaf", "r") as file:
    file_content = file.read()

# Create the document
doc = {
    "relative_media_url": "videos/2019/Transcrições_2019/465A/465A.eaf",
    "mime_type": "application/xml",
    "content": file_content  # Include the file content in the document
}
print('\n----------------------------------------------------------------------------------- DOCUMENT INDEX')
docIndex = client.index(index=index_name, body=doc)
print(docIndex['result'])

# Document analyser for the .eaf files with the video transcriptions
print('\n----------------------------------------------------------------------------------- ANALYSE')
anls = {
    "analyzer": "eal_analyzer",
    "text": doc['content']
}
resp = client.indices.analyze(body=anls, index=index_name)
# Extract the tokens
tokens = [token['token'] for token in resp['tokens']]

# Print the tokens
print(tokens)
