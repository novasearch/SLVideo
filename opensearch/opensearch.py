import pprint as pp
from opensearchpy import OpenSearch

class LGPOpenSearch:

    def __init__(self):
        host = '10.10.255.202'
        port = 8200
        user = 'pt.sign.language'
        password = '_2@Vu6Z%a#jPv#6'
        
        self.index_name = 'pt.sign.language'

        self.client = OpenSearch(
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
                    "frame_id": {
                        "type": "keyword"
                    },
                    "video_id": {
                        "type": "keyword"
                    },
                    "annotation_id": {
                        "type": "keyword"
                    },
                    "path": {
                        "type": "keyword"
                    },
                    "timestamp": {
                        "type": "double"
                    },
                    "linguistic_type_ref": {
                        "type": "keyword"
                    },
                    "annotation": {
                        'analyzer': 'standard',
                        'similarity': 'BM25',
                        'type': 'text'
                    },
                    "frame_embedding": {
                        'dimension': 512,
                        'type': 'knn_vector'
                    },
                    "annotation_embedding": {
                        'dimension': 512,
                        'type': 'knn_vector'
                    }
                    #"start_time": {
                    #    "type": "double"
                    #},
                    #"end_time": {
                    #    "type": "double"
                    #},
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

    def index_exists(self):
        return self.client.indices.exists(self.index_name)

    def print_index(self):
        if self.client.indices.exists(index=self.index_name):
            print('\n----------------------------------------------------------------------------------- INDEX EXISTS')
            resp = self.client.indices.open(index=self.index_name)
            print(resp)

            print('\n----------------------------------------------------------------------------------- INDEX SETTINGS')
            settings = self.client.indices.get_settings(index=self.index_name)
            pp.pprint(settings)

            print('\n----------------------------------------------------------------------------------- INDEX MAPPINGS')
            mappings = self.client.indices.get_mapping(index=self.index_name)
            pp.pprint(mappings)

            print('\n----------------------------------------------------------------------------------- INDEX #DOCs')
            print(self.client.count(index=self.index_name))
        else:
            print('\nCreating index:')
            response = self.create_index()
            print(response)