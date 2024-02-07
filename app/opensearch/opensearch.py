import pprint as pp
from opensearchpy import OpenSearch, OpenSearchException


class LGPOpenSearch:

    def __init__(self):
        host = 'api.novasearch.org'
        port = 443
        user = 'pt.sign.language'  # Add your username here.
        password = '_2@Vu6Z%a#jPv#6'  # Add your user password here. For testing only. Don't store credentials in code.

        self.index_name = 'pt.sign.language'

        self.client = OpenSearch(
            hosts=[{'host': host, 'port': port}],
            http_compress=True,  # enables gzip compression for request bodies
            http_auth=(user, password),
            url_prefix='opensearch',
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
                    "video_id": {
                        "type": "keyword"
                    },
                    "annotation_id": {
                        "type": "keyword"
                    },
                    "annotation_value": {
                        'analyzer': 'standard',
                        'similarity': 'BM25',
                        'type': 'text'
                    },
                    "base_frame_embedding": {
                        'dimension': 512,
                        'type': 'knn_vector'
                    },
                    "average_frame_embedding": {
                        'dimension': 512,
                        'type': 'knn_vector'
                    },
                    "best_frame_embedding": {
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
            self.client.create(index=self.index_name, id=doc["video_id"] + "_" + doc["annotation_id"], body=doc)
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

    def knn_query(self, embedding, k=10):
        query_obj = {
            "query": {
                "knn": {
                    "base_frame_embedding": {
                        "vector": embedding,
                        "k": k
                    }
                }
            }
        }
        return self.client.search(
            body=query_obj,
            index=self.index_name
        )

    def knn_query_average(self, embedding, k=10):
        query_obj = {
            "query": {
                "knn": {
                    "average_frame_embedding": {
                        "vector": embedding,
                        "k": k
                    }
                }
            }
        }
        return self.client.search(
            body=query_obj,
            index=self.index_name
        )

    def knn_query_best(self, embedding, k=10):
        query_obj = {
            "query": {
                "knn": {
                    "best_frame_embedding": {
                        "vector": embedding,
                        "k": k
                    }
                }
            }
        }
        return self.client.search(
            body=query_obj,
            index=self.index_name
        )
