from pymilvus import connections, utility, list_collections, Collection, FieldSchema, CollectionSchema, DataType

class MilvusService:
    def __init__(self, host="localhost", port="19530"):
        self.host = host
        self.port = port
        self.connect()

    def connect(self):
        connections.connect(alias="default", host=self.host, port=self.port)
        print("Conectado ao Milvus.")

    def create_collection(self, collection_name: str, dim_pca: int = 128) -> Collection:
        # Se já existir, apenas carrega
        if collection_name in list_collections():
            collection = Collection(collection_name)
            print(f"Collection '{collection_name}' já existe. Carregando...")
        else:
            print(f"Criando nova collection '{collection_name}'...")
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim_pca),
                FieldSchema(name="path", dtype=DataType.VARCHAR, max_length=500)
            ]
            schema = CollectionSchema(fields, description="Image similarity search with CLIP embeddings")
            collection = Collection(name=collection_name, schema=schema)

            # Criar índice só na criação (para não sobrescrever índices existentes)
            collection.create_index(
                "embedding",
                {
                    "index_type": "IVF_FLAT",
                    "metric_type": "COSINE",
                    "params": {"nlist": 1024}
                }
            )
            print(f"Collection '{collection_name}' criada com dimensão {dim_pca}.")

        # Carregar em memória
        collection.load()
        return collection
    
    def count_images(self, collection_name: str) -> int:
        if not utility.has_collection(collection_name):
            return 0

        collection = Collection(collection_name)
        stats = collection.num_entities 
        return stats

    def drop_collection(self, collection_name: str):
        #Exclui a collection do Milvus se ela existir.
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
            print(f"Collection '{collection_name}' excluída com sucesso.")
        else:
            print(f"Collection '{collection_name}' não existe.")




