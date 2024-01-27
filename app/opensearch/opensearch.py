import pprint as pp
from opensearchpy import OpenSearch, OpenSearchException


class LGPOpenSearch:

    def __init__(self):
        host = 'exp10.exp10.255.202'
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
                    "annotation_value": {
                        'analyzer': 'standard',
                        'similarity': 'BM25',
                        'type': 'text'
                    },
                    "frame_embedding": {
                        'dimension': 512,
                        'type': 'knn_vector'
                    },
                }
            }
        }

        if not self.index_exists():
            print('Creating index', self.index_name, ':')
            response = self.client.indices.create(self.index_name, body=index_body)
            index_settings = {
                "settings": {
                    "index": {
                        "refresh_interval": "1s"
                    }
                }
            }
            self.client.indices.put_settings(index=self.index_name, body=index_settings)
            print(response)
        else:
            print("Index", self.index_name, "already created.")

    def index_exists(self):
        return self.client.indices.exists(self.index_name)

    def print_index(self):
        if self.client.indices.exists(index=self.index_name):
            print('\n----------------------------------------------------------------------------------- INDEX EXISTS')
            resp = self.client.indices.open(index=self.index_name)
            print(resp)

            print(
                '\n----------------------------------------------------------------------------------- INDEX SETTINGS')
            settings = self.client.indices.get_settings(index=self.index_name)
            pp.pprint(settings)

            print(
                '\n----------------------------------------------------------------------------------- INDEX MAPPINGS')
            mappings = self.client.indices.get_mapping(index=self.index_name)
            pp.pprint(mappings)

            print('\n----------------------------------------------------------------------------------- INDEX #DOCs')
            print(self.client.count(index=self.index_name))
        else:
            print('\nCreating index:')
            response = self.create_index()
            print(response)

    def index_if_not_exists(self, doc):
        try:
            self.client.create(index=self.index_name, id=doc["frame_id"], body=doc)
            return True
        except OpenSearchException as e:
            if 'version_conflict_engine_exception' in str(e):
                return False
            else:
                raise

    def delete_index(self):
        self.client.indices.delete(
            index=self.index_name
        )
        return True

    def annotation_query(self, video_id, query):
        query_obj = {
            "query": {
                "bool": {
                    "must": {
                        "term": {
                            "video_id": video_id
                        }
                    },
                    "should": {
                        "match": {
                            "annotation": query
                        }
                    }
                }
            }
        }
        return self.client.search(
            body=query_obj,
            index=self.index_name
        )

    def annotation_embeddings_query(self, video_id, query):

        query_embedding = self.st.text_encode(query).cpu()

        query_obj = {
            'size': 5,
            '_source': ['frame_id', "video_id", "annotation", "timestamp", "path", "annotation_embedding"],
            "query": {
                "bool": {
                    "should": {
                        "knn": {
                            "annotation_embedding": {
                                "vector": query_embedding.numpy(),
                                "k": 10
                            }
                        }
                    }
                }
            }
        }
        return self.client.search(
            body=query_obj,
            index=self.index_name
        )
