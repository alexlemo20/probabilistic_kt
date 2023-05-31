import sys
sys.path.append("/media/data/alexlemo/probabilistic_kt")
import torch
from torch.autograd import Variable
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import matplotlib
from matplotlib.pylab import plt
from nn.retrieval_evaluation import evaluate_model_retrieval
from models.cifar_tiny import Cifar_Tiny
import pickle
from nn.nn_utils import load_model
from exp_cifar.cifar_dataset import cifar10_loader, cifar100_loader
from nn.retrieval_evaluation import retrieval_evaluation, Database
from nn.nn_utils import get_labels, extract_features, get_raw_features
from exp_yf.yt_dataset import get_yt_loaders



def knowledge_transfer(net, net_to_distill, transfer_loader, epochs=1, lr=0.0001, typical_mask=1, typical_step=0, atypical_mask=1, atypical_step=0, atypical_proportion=0, supervised_weight=0):
	"""
	Performs unsupervised neural network knowledge transfer
	:param net:
	:param net_to_distill:
	:param transfer_loader:
	:param epochs:
	:param lr:
	:return:
	"""
	optimizer = optim.Adam(params=net.parameters(), lr=lr)
	typical_losses = list()
	atypical_losses = list()
	overall_losses = list()
	precisions = list()
	dataset_loader=cifar10_loader
	_, test_loader, train_loader = dataset_loader(batch_size=128)
	#step = 1 / epochs
	#par_step = 0
	for epoch in range(epochs):

		net.train()
		net_to_distill.eval()
		typical_loss = 0
		atypical_loss = 0
		train_loss = 0
		counter = 1
		for (inputs, targets) in tqdm(transfer_loader):
			inputs, targets = inputs.cuda(), targets.cuda()

			# Feed forward the network and update
			optimizer.zero_grad()

			# # Get the data
			with torch.no_grad():
				output_target = net_to_distill.get_features(Variable(inputs))
				outputs_net = net.get_features(Variable(inputs))

			# Get the loss
			if supervised_weight > 0:
				loss = cosine_similarity_loss(outputs_net, output_target) + \
				       supervised_weight * supervised_loss(outputs_net, targets)
			else:
				#loss = cosine_similarity_loss(outputs_net, output_target)
				loss, typ_loss, atyp_loss = cosine_similarity_loss(outputs_net, output_target, targets, typical_mask, atypical_mask, atypical_proportion)
				#loss = cosine_similarity_loss(outputs_net, output_target, targets, typical_mask, atypical_mask, atypical_proportion, par_step) # par_step

			typical_loss += typ_loss.cpu().data.item()
			atypical_loss += atyp_loss.cpu().data.item()
			
			loss.requires_grad = True
			loss.backward()
			optimizer.step()

			train_loss += loss.cpu().data.item()
			counter += 1

		typical_mask += typical_step
		atypical_mask += atypical_step
		#par_step += step

		typical_loss /= float(counter)
		atypical_loss /= float(counter)
		print("Typical loss = ",typical_loss," Atypical loss = ", atypical_loss)
		train_loss = train_loss / float(counter)
		print("\n Epoch = ", epoch, " Loss  = ", train_loss)

		typical_losses.append(typical_loss)
		atypical_losses.append(atypical_loss)
		overall_losses.append(train_loss)


		results = retrieval_evaluation(net, train_loader, test_loader)
		precisions.append(100 * results['map'])
		print("\n Precision: ", 100 * results['map'])
	print("\n The precisions as a list are: ",precisions)
	show_results(typical_losses, atypical_losses, overall_losses, precisions, epochs)
	
def knowledge_transfer_handcrafted(net, transfer_loader, epochs=1, lr=0.0001, supervised_weight=0, typical_mask=1, typical_step=0, atypical_mask=1, atypical_step=0, atypical_proportion=0):

	_, test_loader, database_loader = get_yt_loaders(batch_size=128, feature_type='image', seed=1)
	optimizer = optim.Adam(params=net.parameters(), lr=lr)
	#typical_mask = 1
	#atypical_mask = 0
	#typical_target = 2
	#atypical_target = 1
	#atypical_proportion = 0.2

	#typical_step = (typical_target - typical_mask)/epochs
	#atypical_step = (atypical_target - atypical_mask)/epochs


	typical_losses = list()
	atypical_losses = list()
	overall_losses = list()
	precisions = list()

	#_,test_loader, train_loader = get_yt_loaders(batch_size=128, feature_type='image',seed=1)
	for epoch in range(epochs):
		net.train()
		typical_loss = 0
		atypical_loss = 0
		train_loss = 0
		counter = 1
		for (inputs, features, targets) in tqdm(transfer_loader):
			inputs, features, targets = inputs.cuda(), features.cuda(), targets.cuda()
			output_target = Variable(features)

			# Feed forward the network and update
			optimizer.zero_grad()

			# # Get the data
			outputs_net = net.get_features(Variable(inputs))

			# Get the loss
			#loss = cosine_similarity_loss(outputs_net, output_target) + \
			#       supervised_weight * supervised_loss(outputs_net, targets)
			#loss = cosine_similarity_loss(outputs_net, output_target, targets, typical_mask, atypical_mask, atypical_proportion) + \
			#       supervised_weight * supervised_loss(outputs_net, targets)
			loss, typ_loss, atyp_loss = cosine_similarity_loss(outputs_net, output_target, targets, typical_mask, atypical_mask, atypical_proportion)# + \
			#   supervised_weight * supervised_loss(outputs_net, targets)

			loss.backward()
			optimizer.step()

			typical_loss += typ_loss.cpu().data.item()
			atypical_loss += atyp_loss.cpu().data.item()
			train_loss += loss.cpu().data.item()
			counter += 1

		typical_mask += typical_step
		atypical_mask += atypical_step
		
		results = retrieval_evaluation(net, database_loader, test_loader)
		precisions.append(100 * results['map'])
		print("\n Precision: ", 100 * results['map'])
		
		typical_loss /= float(counter)
		atypical_loss /= float(counter)
		print("Typical loss = ",typical_loss," Atypical loss = ", atypical_loss)
		train_loss = train_loss / float(counter)
		print("\n Epoch = ", epoch, " Loss  = ", train_loss)
		typical_losses.append(typical_loss)
		atypical_losses.append(atypical_loss)
		overall_losses.append(train_loss)

	show_results(typical_losses, atypical_losses, overall_losses, precisions, epochs)


def cosine_similarity_loss(output_net, target_net, targets, typical_mask, atypical_mask, atypical_proportion, step=0,  eps=0.0000001):
    
    # Normalize each vector by its norm
    output_net_norm = torch.sqrt(torch.sum(output_net ** 2, dim=1, keepdim=True))
    output_net = output_net / (output_net_norm + eps)
    output_net[output_net != output_net] = 0
    
    target_net_norm = torch.sqrt(torch.sum(target_net ** 2, dim=1, keepdim=True))
    target_net = target_net / (target_net_norm + eps)
    target_net[target_net != target_net] = 0

    # Calculate the cosine similarity
    model_similarity = torch.mm(output_net, output_net.transpose(0, 1))
    target_similarity = torch.mm(target_net, target_net.transpose(0, 1))

    # Scale cosine similarity to 0..1
    model_similarity = (model_similarity + 1.0) / 2.0
    target_similarity = (target_similarity + 1.0) / 2.0

    # Transform them into probabilities
    model_similarity = model_similarity / torch.sum(model_similarity, dim=1, keepdim=True)
    target_similarity = target_similarity / torch.sum(target_similarity, dim=1, keepdim=True)
    
    # Calculate the KL-divergence    
    #loss = torch.mean(target_similarity * torch.log((target_similarity + eps) / (model_similarity + eps)))
    
    # choose to work with the student or the teacher model
    flag = False
    if flag == True:
      cosine_similarity = model_similarity #student
    else :
      cosine_similarity = target_similarity #teacher

    # choose to shape a symmetric matrix
    #cosine_similarity = (cosine_similarity + torch.transpose(cosine_similarity, 0, 1)) / 2
    labels = targets.cpu().numpy()

    #typical_similarities = calculate_mask(cosine_similarity, labels, typical_mask, atypical_mask, atypical_proportion, target_similarity, model_similarity, step)
    typical_similarities, typical_loss, atypical_loss = calculate_mask(cosine_similarity, labels, typical_mask, atypical_mask, atypical_proportion, target_similarity, model_similarity)

    loss = torch.mean(target_similarity * torch.log((target_similarity + eps) / (model_similarity + eps)) * typical_similarities)
    #return loss
    return loss, typical_loss, atypical_loss 


def supervised_loss(output_net, targets, eps=0.0000001):
	labels = targets.cpu().numpy()
	target_sim = np.zeros((labels.shape[0], labels.shape[0]), dtype='float32')
	for i in range(labels.shape[0]):
		for j in range(labels.shape[0]):
			if labels[i] == labels[j]:
				target_sim[i, j] = 1.0
			else:
				target_sim[i, j] = 0

	target_similarity = torch.from_numpy(target_sim).cuda()
	target_similarity = Variable(target_similarity)

	# Normalize each vector by its norm
	output_net_norm = torch.sqrt(torch.sum(output_net ** 2, dim=1, keepdim=True))
	output_net = output_net / (output_net_norm + eps)
	output_net[output_net != output_net] = 0

	# Calculate the cosine similarity
	model_similarity = torch.mm(output_net, output_net.transpose(0, 1))
	# Scale cosine similarity to 0..1
	model_similarity = (model_similarity + 1.0) / 2.0

	# Transform them into probabilities
	model_similarity = model_similarity / torch.sum(model_similarity, dim=1, keepdim=True)
	target_similarity = target_similarity / torch.sum(target_similarity, dim=1, keepdim=True)

	# Calculate the KL-divergence
	loss = torch.mean(target_similarity * torch.log((target_similarity + eps) / (model_similarity + eps)))

	return loss

def calculate_mask(cosine_similarity, labels, typical_mask, atypical_mask, atypical_proportion, target_similarity, model_similarity, step=0, eps=0.0000001):
	intra_similarities = {}
	inter_similarities = {}

	for i in range(cosine_similarity.shape[0]):
		for j in range(i, cosine_similarity.shape[1]): #i, if a symmetric matrix 
			# intra similarity
			if labels[i] == labels[j]:
				intra_similarities[str(i) + " " + str(j)] = cosine_similarity[i][j].item()
				# comment out if not a symmetric matrix
				#intra_similarities[str(j) + " " + str(i)] = cosine_similarity[j][i].item()
			else :
				inter_similarities[str(i) + " " + str(j)] = cosine_similarity[i][j].item()
				# comment out if not a symmetric matrix
				#inter_similarities[str(j) + " " + str(i)] = cosine_similarity[j][i].item()

	# sort the intra and inter similarities
	intra_sorted = dict(sorted(intra_similarities.items(), key = lambda values:values[1]))
	inter_sorted = dict(sorted(inter_similarities.items(), key = lambda values:values[1], reverse = True))
	# choose the atypical similarities size
	atypical_size_intra = round(atypical_proportion * len(intra_similarities))
	atypical_size_inter = round(atypical_proportion * len(inter_similarities))
	atypical_size_inter = round((atypical_proportion * ((cosine_similarity.shape[0]**2)/2 + (cosine_similarity.shape[0]/2))) - atypical_size_intra)
	
	# choose atypical pairs
	atypical_intra = list(intra_sorted.keys())[:atypical_size_intra]
	atypical_inter = list(inter_sorted.keys())[:atypical_size_inter]
	# initialize the matrix which will be used for the involvement of typical and atypical pairs
	typical_similarities = torch.cuda.FloatTensor(cosine_similarity.shape[0], cosine_similarity.shape[1]).fill_(typical_mask) # typical_mask
	typical_loss_matrix = torch.cuda.FloatTensor(cosine_similarity.shape[0], cosine_similarity.shape[1]).fill_(1)
	atypical_loss_matrix = torch.cuda.FloatTensor(cosine_similarity.shape[0], cosine_similarity.shape[1]).fill_(0)




	# 1st method
	atypical = atypical_intra + atypical_inter
	#   atypical_loss = 0
	for coordinates in atypical:
		x, y = coordinates.split(' ')
		typical_similarities[int(x)][int(y)] = atypical_mask # atypical_mask
		# comment out if not to shape a symmetric mask matrix
		typical_similarities[int(y)][int(x)] = atypical_mask

		typical_loss_matrix[int(x)][int(y)] = 0
		typical_loss_matrix[int(y)][int(x)] = 0
		atypical_loss_matrix[int(x)][int(y)] = 1 # S.O.S.
		atypical_loss_matrix[int(y)][int(x)] = 1
		


	# 2nd method 
	#typical_intra = list(intra_sorted.keys())[-(len(intra_sorted)-atypical_size_intra):]
	#typical_inter = list(inter_sorted.keys())[-(len(inter_sorted)-atypical_size_inter):]

	#mean_intra = torch.mean(torch.tensor(list(intra_sorted.values())))
	#max_dist_intra = max(abs(list(intra_sorted.values())[0]-mean_intra), abs(list(intra_sorted.values())[-1]-mean_intra)) # first & last element
	#mean_inter = torch.mean(torch.tensor(list(inter_sorted.values())))
	#max_dist_inter = max(abs(list(inter_sorted.values())[0]-mean_inter), abs(list(inter_sorted.values())[-1]-mean_inter)) # first & last element

	#typical_similarities = torch.cuda.FloatTensor(cosine_similarity.shape[0], cosine_similarity.shape[1]).fill_(1)

	#for coordinates in atypical_intra:
	#	x, y = coordinates.split(' ')
	#	typical_similarities[int(x)][int(y)] = ((abs(intra_sorted[coordinates] - mean_intra) / max_dist_intra) + 0.5) * atypical_mask
	#	typical_similarities[int(y)][int(x)] = ((abs(intra_sorted[coordinates] - mean_intra) / max_dist_intra) + 0.5) * atypical_mask
		
	#	typical_loss_matrix[int(x)][int(y)] = 0
	#	typical_loss_matrix[int(y)][int(x)] = 0
	#	atypical_loss_matrix[int(x)][int(y)] = 1
	#	atypical_loss_matrix[int(y)][int(x)] = 1

	#for coordinates in atypical_inter:
	#	x, y = coordinates.split(' ')
	#	typical_similarities[int(x)][int(y)] = ((abs(inter_sorted[coordinates] - mean_inter) / max_dist_inter)  + 0.5) * atypical_mask
	#	typical_similarities[int(y)][int(x)] = ((abs(inter_sorted[coordinates] - mean_inter) / max_dist_inter)  + 0.5) * atypical_mask
		
	#	typical_loss_matrix[int(x)][int(y)] = 0
	#	typical_loss_matrix[int(y)][int(x)] = 0
	#	atypical_loss_matrix[int(x)][int(y)] = 1
	#	atypical_loss_matrix[int(y)][int(x)] = 1

	#for coordinates in typical_intra:
	#	x, y = coordinates.split(' ')
	#	typical_similarities[int(x)][int(y)] = ((abs(intra_sorted[coordinates] - mean_intra) / max_dist_intra) + 0.5) * typical_mask
	#	typical_similarities[int(y)][int(x)] = ((abs(intra_sorted[coordinates] - mean_intra) / max_dist_intra) + 0.5) * typical_mask

	#for coordinates in atypical_inter:
	#	x, y = coordinates.split(' ')
	#	typical_similarities[int(x)][int(y)] = ((abs(inter_sorted[coordinates] - mean_inter) / max_dist_inter)  + 0.5) * typical_mask
	#	typical_similarities[int(y)][int(x)] = ((abs(inter_sorted[coordinates] - mean_inter) / max_dist_inter)  + 0.5) * typical_mask



	# 3rd method
	#intra
	#min_intra = list(intra_sorted.values())[-1]
	#max_dis_intra = list(intra_sorted.values())[0] - list(intra_sorted.values())[-1]
	#inter
	#max_inter = list(inter_sorted.values())[-1]
	#max_dis_inter = list(inter_sorted.values())[-1] - list(inter_sorted.values())[0]
	#typical_similarities = torch.cuda.FloatTensor(cosine_similarity.shape[0], cosine_similarity.shape[1]).fill_(1)
	#for coordinates in intra_sorted:
	#	x, y = coordinates.split(' ')
	#	typical_similarities[int(x)][int(y)] = (abs(((intra_sorted[coordinates] - min_intra) / max_dis_intra) - step) / 2) + 0.5
	#	typical_similarities[int(y)][int(x)] = (abs(((intra_sorted[coordinates] - min_intra) / max_dis_intra) - step) / 2) + 0.5

	#for coordinates in inter_sorted:
	#	x, y = coordinates.split(' ')
	#	typical_similarities[int(x)][int(y)] = (abs(((max_inter - inter_sorted[coordinates]) / max_dis_inter) - step) / 2) + 0.5
	#	typical_similarities[int(y)][int(x)] = (abs(((max_inter - inter_sorted[coordinates]) / max_dis_inter) - step) / 2) + 0.5



	typical_loss =  torch.mean(target_similarity * torch.log((target_similarity + eps) / (model_similarity + eps)) * typical_loss_matrix)
	atypical_loss =  torch.mean(target_similarity * torch.log((target_similarity + eps) / (model_similarity + eps)) * atypical_loss_matrix)
	
	
	normalized_mask = torch.mul(typical_similarities, (cosine_similarity.shape[0]**2) / torch.sum(typical_similarities))
	
	#return normalized_mask
	return normalized_mask, typical_loss, atypical_loss

def show_results(typical_losses, atypical_losses, overall_losses, precisions, epochs):
    #Plot losses
    plt.plot(range(epochs), typical_losses, label='Typical Loss')
    plt.plot(range(epochs), atypical_losses, label='Atypical Loss')
    plt.plot(range(epochs), overall_losses, label='Overall Loss')
    #plt.plot(range(epochs), precisions, label='Precision')
    plt.title('Typical, Atypical and Overall Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    # Display the plot
    plt.legend(loc='best')
    plt.show()
    plt.savefig('losses.png')
	
    plt.plot(range(epochs), precisions, label='Precision')
    plt.title('Precision at 11 recall points')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.show()
    plt.savefig('precision.png')
