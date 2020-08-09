'''
Author: Gianfranco Lombardo
Project: WalkHubs2Vec
'''
import settings
settings.init()
from operator import itemgetter
import os
import deepwalk_functions
from gensim.models import Word2Vec
import time
import random
from skipgram import Skipgram
import networkx as nx
from scipy.linalg import orthogonal_procrustes
from math import sqrt
import numpy as np
from multiprocessing import Process	

'''
Reads an edges list where each row has the format
node1(separator)node2
Depending on the graph type in input (directed or not) the edges is added

Source: edges list path
Graph: graph to be filled with edges
Separator: the separator character between node1 and node2 in the list
'''
def read_edges_list(source,graph,separator=" "):
	assert source!=None and graph!=None
	with open(source) as edges_file:
		for line in edges_file.readlines():
			node1, node2 = line.split(separator)
			node2 = node2.replace("\n","")
			if(node1!=node2):
				#Do not consider auto-loop
				graph.add_edge(node1, node2)
	return graph
def export_nodes_set(path,nodeset):
	with open(path,"w+") as output:
		for n in nodeset:
			output.write(n+"\n")
def export_graph(path,G):
	'''with open(path+"_nodes.csv","w+") as output:
		for n in G.nodes():
			output.write(n+"\n")'''
	with open(path+"_edges.csv","w+") as output:
		for e in G.edges():
			output.write(e[0]+" "+e[1]+"\n")

def traslation(X,normalizationAxis=0):
	if(len(X)>1):
		result = (X - np.mean(X, axis=normalizationAxis))
		traslationFactor =np.mean(X, axis=normalizationAxis)
	else:
		result = (X - np.zeros(settings.DIMENSION))
		traslationFactor = np.zeros(settings.DIMENSION)
	return result,traslationFactor

def scaling(X):

	k = X.shape[0]

	scalingFactor = 0
	Xlist = X.flatten()
	for e in Xlist:
		scalingFactor+= float(e*e)
	scalingFactor= scalingFactor/k
	scalingFactor = sqrt(scalingFactor)
	result = X/scalingFactor
	return result,scalingFactor
	

def incremental_nodes_edges(name,I_nodeSet,G):
	print ("Thread '" + name + "' avviato")
	zeros=0
	for node in I_nodeSet:
		i_graph = nx.Graph()
		if settings.DIRECTED:
			i_graph = nx.DiGraph()
		for e in G.edges():
			if node in e:
				i_graph.add_edge(e[0],e[1])
		export_graph(settings.INCREMENTAL_DIR+str(node),i_graph)

def extract_hub_component(G,threshold,verbose=False):
	nd = sorted(G.degree(),key=itemgetter(1))
	B = []
	B_len = round(G.number_of_nodes()/100*(100-threshold)) # E.g IF hub split is 30 we calculate 100-30=70 for b_len
	
	for elem in nd:
		node_id = elem[0]
		node_degree = elem[1]
		if len(B) < B_len:
			B.append(node_id)
		else:
			break
	next_node_degree = G.degree(nd[B_len][0])
	# Each degree has to be entirely included in A or B. In this case, otherwise we delete the degree from B
	if G.degree(B[B_len-1]) == next_node_degree:
		B = [x for x in B if not G.degree(x) == next_node_degree]

	A = [x[0] for x in nd[len(B):]]
	
	if verbose:
		print('The '+str(100-threshold)+'% of ' + str(G.number_of_nodes()) + ' is about ' + str(B_len) + '\n')
		print('B length: ' + str(len(B)) + ' (' + str(round(100*len(B)/G.number_of_nodes(), 2)) + '%)')
		print('Min Degree in B: ' + str(G.degree(B[0])))
		print('Max Degree in B: ' + str(G.degree(B[len(B)-1])) + '\n')
		print('A length: ' + str(len(A))  + ' (' + str(round(100*len(A)/G.number_of_nodes(), 2)) + '%)')
		print('Min Degree in A: ' + str(G.degree(A[0])) )
		print('Max Degree in A: ' + str(G.degree(A[len(A)-1])) + '\n')
		
	H = G.subgraph(A)
	return H
def parallel_incremental_embedding(nodes_list,edges_lists,H,G,G_model,workers=2):
	nodes_sets = [nodes_list[i::workers] for i in range(workers)]
	graph_sets = [edges_lists[i::workers] for i in range(workers)]
	processList = []
	t_c=0
	for ns in nodes_sets:
		p = Process(target=thread_incremental_embedding, args=("process-"+str(t_c),ns,graph_sets[nodes_sets.index(ns)],H,G,G_model))
		processList.append(p)
		t_c+=1
	for p in processList:
		p.start()
	for p in processList:
		p.join()
def thread_incremental_embedding(process_name,nodes_list,edges_lists,H,G,G_model):
	print(process_name+" started")
	for node in nodes_list:
		print(process_name+") processing node: "+node)
		emb =incremental_embedding(node,edges_lists[nodes_list.index(node)],H,G,G_model)
		
		content=node+" "
		for column in range(0,len(emb)):
				if column != len(emb) -1:
					content+=str(emb[column])+" "
				else:
					content+=str(emb[column])+"\n"

		settings.lck.acquire()
		with open(settings.INCREMENTAL_MODEL, 'a+') as out:
			out.write(content)
		settings.lck.release()

	print(process_name+" ended")


def incremental_embedding(node,edges_list,H,completeGraph,G_model):
	G = completeGraph.copy()
	tmp = nx.Graph()
	tmp_nodes_added =[]
	if settings.DIRECTED:
		tmp = nx.DiGraph()
	tmp = read_edges_list(edges_list,tmp)
	H_plus_node = H.copy()
	H_init_edges_number = len(H_plus_node.edges())
	
	embeddable = False
	
	for e in tmp.edges():
		if(e[0] in H.nodes() or e[1] in H.nodes()):
			#if node has a link with someone in H
			H_plus_node.add_edge(e[0],e[1])
			embeddable = True

	if(H_init_edges_number == len(H_plus_node.edges())):
		#if node has NOT ANY link with someone in H

		found = False
		it=0

		while(not found and it<len(tmp.edges())):
			e = list(tmp.edges())[it]
			for incident_vertex in e:
				if incident_vertex != node:
					if incident_vertex in G.nodes():
						#vertex linked with node is in G
						found = True
						#print(incident_vertex)
						G.add_edge(e[0],e[1])
						hub_node_found=False
						while not hub_node_found:
							h_node = random.choice(list(H.nodes()))
							exist = nx.has_path(G, source=node, target=h_node)
							if(exist):
								sh_paths =nx.shortest_path(G, source=node, target=h_node, weight=None, method='dijkstra')
								#add this walk to H_plus_node
								for i in range(len(sh_paths)):
									if(i+1<len(sh_paths)):
										H_plus_node.add_edge(sh_paths[i],sh_paths[i+1])
								hub_node_found=True
								embeddable=True
			it+=1

	####### AT THIS POINT I'M GOOD WITH H 
	if(embeddable):
		model_i= None
		if settings.BASE_ALGORITHM =="deepwalk":
			export_graph(settings.TMP+node,H_plus_node)
			model_i=Deepwalk(settings.TMP+node+"_edges.csv",not settings.DIRECTED,settings.EMBEDDING_DIR,node+"_i",1,settings.WINDOWS_SIZE,settings.DIMENSION,settings.NUM_WALKS,settings.LENGTH_WALKS)
			
		elif settings.BASE_ALGORITHM =="node2vec":
			pass #TO DO
		
		assert model_i
		model_i_dict = extract_embedding_for_Hub_nodes(H_plus_node,model_i)
		#pre-processing for alignment
		
		e_i_raw = model_i[node]
		
		i_neighboors = list(H_plus_node[node])
		H_plus_node.remove_node(node) #remove node to be incrematlly added
		H_model=extract_embedding_for_Hub_nodes(H_plus_node,G_model)
		
		A_embeddings = []
		B_embeddings = []
		neighboors=[]
		
		
		for n in i_neighboors:
			neighboors.append(n)
			second_order = list(H_plus_node[n])
			if(node in second_order):
				second_order.remove(node)
			neighboors= neighboors + second_order
		neighboors = set(neighboors)
		
		for n in neighboors:
			for e in H_model:
				if str(e) == n:
					A_embeddings.append(H_model[e])#e[1:settings.DIMENSION+1])
			for f in model_i_dict:
				if str(f) == n:
					B_embeddings.append(model_i_dict[f])#f[1:settings.DIMENSION+1])
		
		A_embeddings,A_mean = traslation(A_embeddings)
		# If here we save embeddings A not scaled but only traslated
		A_embeddings, A_scale = scaling(A_embeddings)
		
		B_embeddings,B_mean = traslation(B_embeddings)
		B_embeddings, scalingFactor = scaling(B_embeddings)
		R, s = orthogonal_procrustes(B_embeddings,A_embeddings)

		e_i = e_i_raw - B_mean
		e_i = e_i/scalingFactor
		e_i = e_i.dot(R)
		#Rescale to A scale
		e_i = A_scale*e_i
		#Translate again into the original position
		e_i+=A_mean
		
		#Remove temporary files
		os.remove(settings.TMP+node+"_edges.csv")
		return e_i
	
def Deepwalk(edges_file,edges_type,embedding_dir,embeddingName,emb_workers,window_size,representation_size,NUM_WALKS,LEN_WALKS):
	start_time = time.time()
	G = deepwalk_functions.load_edgelist(edges_file, undirected= edges_type)
	walks_filebase = embedding_dir+embeddingName+".walks"
	walk_files = deepwalk_functions.write_walks_to_disk(G, walks_filebase, num_paths=NUM_WALKS,
                                         path_length=LEN_WALKS, alpha=0, rand=random.Random(0),
                                         )
	#print("walk_files is "+str(walk_files))
	vertex_counts = deepwalk_functions.count_textfiles(walk_files, emb_workers)

	#print("Training for "+embeddingName)
	walks_corpus = deepwalk_functions.WalksCorpus(walk_files)
	model = Skipgram(sentences=walks_corpus, vocabulary_counts=vertex_counts,
					 size=representation_size,
					 window=window_size, min_count=0, trim_rule=None, workers=emb_workers)
	model.wv.save_word2vec_format(embedding_dir+"emb/"+embeddingName+".emb") #plain text
	model.save(embedding_dir+"bin/"+embeddingName+".bin")
	elapsed_time = time.time() - start_time
	#print("Elapsed time :"+str(elapsed_time))
	
	#Remove temporary files
	os.remove(walks_filebase+".0")
	return model
def extract_embedding_for_Hub_nodes(H,G_model):
	H_mod = {}
	
	for n in H.nodes():
		e_n=G_model[n]
		H_mod[n]=e_n
	return H_mod
	