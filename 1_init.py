'''
Author: Gianfranco Lombardo
Project: WalkHubs2Vec
'''
import settings
settings.init()
from utils import read_edges_list,extract_hub_component,Deepwalk,incremental_embedding,export_graph,extract_embedding_for_Hub_nodes,parallel_incremental_embedding
import networkx as nx
from operator import itemgetter
from gensim.models import Word2Vec
import os

########## SETTINGS ########################
EDGES_LIST = "protein_S_edges.csv"
EMBED_G = True
EMBEDDING_WORKERS= 4
############################################

			
if __name__ == '__main__':
	G = nx.Graph()
	if settings.DIRECTED:
		G = nx.DiGraph()
	G = read_edges_list(EDGES_LIST,G)
	H = extract_hub_component(G,settings.CUT_THRESHOLD)
	export_graph(settings.NAME_DATA+"_H",H)
	if EMBED_G:
		G_model= Deepwalk(EDGES_LIST,not settings.DIRECTED,settings.EMBEDDING_DIR,settings.NAME_DATA+"_G",EMBEDDING_WORKERS,settings.WINDOWS_SIZE,settings.DIMENSION,settings.NUM_WALKS,settings.LENGTH_WALKS)
	else:
		G_model = Word2Vec.load(settings.EMBEDDING_DIR+"bin/"+settings.NAME_DATA+"_G.bin")

	#for multithread embedding
	nodes_list=[]
	edges_lists=[]
	for file in os.listdir(settings.INCREMENTAL_DIR):
		node_id = file.split("_")[0]
		nodes_list.append(node_id)
		edges_lists.append(settings.INCREMENTAL_DIR+file)
	parallel_incremental_embedding(nodes_list,edges_lists,H,G,G_model,4)
