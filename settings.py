def init():
	global DIRECTED
	global DATA
	global INCREMENTAL_DIR
	global NAME_DATA
	global EMBEDDING_DIR
	global CUT_THRESHOLD
	global WINDOWS_SIZE
	global DIMENSION
	global NUM_WALKS
	global LENGTH_WALKS
	global BASE_ALGORITHM
	global TMP
	global INCREMENTAL_MODEL
	global lck
	import threading
	lck = threading.Lock()
	
	NAME_DATA = "protein"
	DIRECTED = False
	DATA = "data/"+NAME_DATA+".csv"
	INCREMENTAL_DIR=NAME_DATA+"_incremental/"
	
	INCREMENTAL_MODEL = NAME_DATA+"_incremental.csv"
	EMBEDDING_DIR = "embeddings/"
	CUT_THRESHOLD=30
	WINDOWS_SIZE=10
	DIMENSION=128
	NUM_WALKS=80
	LENGTH_WALKS=10
	BASE_ALGORITHM="deepwalk"
	TMP ="tmp/"