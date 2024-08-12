import pprint as pp
import os
from opensearchpy import OpenSearch, OpenSearchException


def gen_doc(video_id: str, annotation_id: str, base_frame_embedding, average_frame_embedding,
            summed_frame_embeddings, best_frame_embedding, all_frames_embeddings, annotation_embedding):
    """ Generate a document for indexing in OpenSearch """
    return {
        "video_id": video_id,
        "annotation_id": annotation_id,
        "base_frame_embedding": base_frame_embedding,
        "average_frame_embedding": average_frame_embedding,
        "best_frame_embedding": best_frame_embedding,
        "summed_frame_embeddings": summed_frame_embeddings,
        "all_frames_embeddings": all_frames_embeddings,
        "annotation_embedding": annotation_embedding,
    }


class LGPOpenSearch:
    """ Class for managing the OpenSearch index for the Portuguese Sign Language project """

    def __init__(self):
        host = os.getenv('OPENSEARCH_HOST', 'localhost')
        port = os.getenv('OPENSEARCH_PORT', 9200)
        user = os.getenv('OPENSEARCH_USER', 'admin')
        password = os.getenv('OPENSEARCH_PASSWORD', 'admin')

        self.index_name = user

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
        """
        Create the OpenSearch index if it doesn't exist.
        """
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
                    "summed_frame_embeddings": {
                        'dimension': 512,
                        'type': 'knn_vector'
                    },
                    "all_frames_embeddings": {
                        'dimension': 512,
                        'type': 'knn_vector'
                    },
                    "annotation_embedding": {
                        'dimension': 512,
                        'type': 'knn_vector'
                    }
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
        """ Check if the index exists """
        return self.client.indices.exists(self.index_name)

    def print_index(self):
        """ Print the index settings, mappings and number of documents """
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
            print('\n INDEX DOES NOT EXIST')

    def index_if_not_exists(self, doc):
        """ Index a document if it doesn't exist """
        try:
            self.client.create(index=self.index_name, id=doc["video_id"] + "_" + doc["annotation_id"], body=doc)
            return True
        except OpenSearchException as e:
            if 'version_conflict_engine_exception' in str(e):
                return False
            else:
                raise

    def update_doc_and_index(self, doc):
        """ Updates a document and index it again """
        if self.client.exists(index=self.index_name, id=doc["video_id"] + "_" + doc["annotation_id"]):
            self.client.delete(index=self.index_name, id=doc["video_id"] + "_" + doc["annotation_id"])
        self.client.create(index=self.index_name, id=doc["video_id"] + "_" + doc["annotation_id"], body=doc)
        return True

    def delete_index(self):
        """ Delete the index """
        if not self.index_exists():
            return False
        self.client.indices.delete(
            index=self.index_name
        )
        return True

    def knn_query(self, embedding, k):
        """ Performs a k-nearest neighbors (k-NN) search on the base_frame_embedding field of the OpenSearch index """
        query_obj = {
            "size": k,
            "query": {
                "bool": {
                    "should": {
                        "knn": {
                            "base_frame_embedding": {
                                "vector": embedding,
                                "k": k
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

    def knn_query_average(self, embedding, k):
        """ Performs a k-nearest neighbors (k-NN) search on the average_frame_embedding field of the OpenSearch index"""
        query_obj = {
            "size": k,
            "query": {
                "bool": {
                    "should": {
                        "knn": {
                            "average_frame_embedding": {
                                "vector": embedding,
                                "k": k
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

    def knn_query_best(self, embedding, k):
        """ Performs a k-nearest neighbors (k-NN) search on the best_frame_embedding field of the OpenSearch index """
        query_obj = {
            "size": k,
            "query": {
                "bool": {
                    "should": {
                        "knn": {
                            "best_frame_embedding": {
                                "vector": embedding,
                                "k": k
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

    def knn_query_summed(self, embedding, k):
        """ Performs a k-nearest neighbors (k-NN) search on the summed_frame_embeddings field of the OpenSearch index"""
        query_obj = {
            "size": k,
            "query": {
                "bool": {
                    "should": {
                        "knn": {
                            "summed_frame_embeddings": {
                                "vector": embedding,
                                "k": k
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

    def knn_query_all(self, embedding, k):
        """ Performs a k-nearest neighbors (k-NN) search on the all_frames_embeddings field of the OpenSearch index"""
        query_obj = {
            "size": k,
            "query": {
                "bool": {
                    "should": {
                        "knn": {
                            "all_frames_embeddings": {
                                "vector": embedding,
                                "k": k
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

    def knn_query_combined(self, embedding, k):
        """Performs a k-nearest neighbors (k-NN) search using all three types of embeddings"""
        query_obj = {
            "size": k,
            "query": {
                "bool": {
                    "should": [
                        {
                            "knn": {
                                "base_frame_embedding": {
                                    "vector": embedding,
                                    "k": k
                                }
                            }
                        },
                        {
                            "knn": {
                                "average_frame_embedding": {
                                    "vector": embedding,
                                    "k": k
                                }
                            }
                        },
                        {
                            "knn": {
                                "best_frame_embedding": {
                                    "vector": embedding,
                                    "k": k
                                }
                            }
                        }
                    ]
                }
            }
        }
        return self.client.search(body=query_obj, index=self.index_name)

    def knn_query_annotations(self, embedding, k):
        """ Performs a k-nearest neighbors (k-NN) search on the annotation_embedding field of the OpenSearch index """
        query_obj = {
            "size": k,
            "query": {
                "bool": {
                    "should": {
                        "knn": {
                            "annotation_embedding": {
                                "vector": embedding,
                                "k": k
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

    def update_annotation_embedding(self, video_id, annotation_id, new_embedding):
        """ Update the annotation_embedding field of a document identified by video_id and annotation_id """
        document_id = f"{video_id}_{annotation_id}"
        update_body = {
            "doc": {
                "annotation_embedding": new_embedding.tolist()
            }
        }
        try:
            self.client.update(index=self.index_name, id=document_id, body=update_body, refresh=True)
        except OpenSearchException as e:
            print("Error updating document:", e)

    def delete_document(self, video_id, annotation_id):
        """ Delete a document from the index """
        document_id = f"{video_id}_{annotation_id}"
        try:
            self.client.delete(index=self.index_name, id=document_id)
        except OpenSearchException as e:
            print("Error deleting document:", e)