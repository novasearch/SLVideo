import pprint as pp
from opensearchpy import OpenSearch
from eaf_parser import get_data_dict_json

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

def create_index(self):
    index_body = {
        "settings": {
            "index": {
                "knn": "true",
                "knn.space_type": "cosinesimil"
            }
        },
        "mappings": {
            "properties": {
                "linguistic_type_ref": {"type": "keyword"},
                "tier_id": {"type": "text"},
                "annotations": {
                    "type": "nested",
                    "properties": {
                        "annotation_id": {"type": "keyword"},
                        "start_time": {"type": "integer"},
                        "end_time": {"type": "integer"},
                        "value": {
                            'analyzer': 'standard',
                            'similarity': 'BM25',
                            "type": "text"
                        }
                        #TODO - add embeddings
                    }
                }
            }
        }
    }
    if not self.index_exists():
            response = self.client.indices.create(self.index_name, body=index_body)
            index_settings = {
                "settings": {
                    "index": {
                        "refresh_interval": "1s"
                    }
                }
            }
            self.client.indices.put_settings(index=self.index_name, body=index_settings)
            print('Creating index:')
            print(response)
        else:
            print("Index already exists!")

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
    response = self.create_index()
    print(response)

doc = get_data_dict_json()

resp = client.index(index=index_name, body=doc)
print('\n----------------------------------------------------------------------------------- INDEXED DOC')
print(resp['result'])