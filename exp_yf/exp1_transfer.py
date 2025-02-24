import sys
sys.path.append("/media/data/alexlemo/probabilistic_kt")

import torch
torch.multiprocessing.set_sharing_strategy('file_system')

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = False

import pickle
from nn.nn_utils import save_model,load_model
from exp_yf.yt_dataset import get_yt_loaders
from models.yt import YT_Small, FaceNet
from nn.pkt import knowledge_transfer_handcrafted, knowledge_transfer
from nn.hint_transfer import unsupervised_hint_transfer_handcrafted
from nn.retrieval_evaluation import retrieval_evaluation


def perform_kt_transfer(kt_type='hint', epochs=20, typ_atyp_ratio=1, target_typ_atyp_ratio=1, T=20, step_ratio=0, atypical_proportion=0, alpha=1, beta=1, method='TAII'):
	results = []
	for i in range(5):
		train_loader, test_loader, database_loader = get_yt_loaders(batch_size=128, feature_type='transfer', seed=1)
		net = FaceNet(533)
		donor_net = YT_Small(533)

		net_model_path = 'models/facenet_yf.model'
		donor_net_model_path = 'models/small_yf.model'
		load_model(net, net_model_path)
		load_model(donor_net, donor_net_model_path)
		net.cuda()
		donor_net.cuda()


		if kt_type == 'hint':
		    unsupervised_hint_transfer_handcrafted(net, train_loader, epochs=epochs, lr=0.0001)
		elif kt_type == 'kt':
		    knowledge_transfer(net, donor_net, train_loader, epochs=epochs, lr=0.0001, typ_atyp_ratio=typ_atyp_ratio, step_ratio=step_ratio, atypical_proportion=atypical_proportion, alpha=alpha, beta=beta, test_loader=test_loader, database_loader=database_loader, dataset_name='yf', T=T, method=method)
		elif kt_type == 'kt_handcrafted':
		    knowledge_transfer_handcrafted(net, train_loader, epochs=epochs, lr=0.0001)	    
		elif kt_type == 'kt_optimal':
		    knowledge_transfer_handcrafted(net, train_loader, epochs=epochs, lr=0.001)
		elif kt_type == 'kt_supervised':
		    knowledge_transfer_handcrafted(net, train_loader, epochs=epochs, lr=0.0001, supervised_weight=0.001)
		save_model(net, 'models/' + kt_type + '_' + str(i) + '.model')

		train_loader, test_loader, database_loader = get_yt_loaders(batch_size=128, feature_type='transfer', seed=i)
		cur_res = retrieval_evaluation(net, database_loader, test_loader)
		results.append(cur_res)
		print(cur_res)

	with open('results/' + kt_type + '.pickle', 'wb') as f:
		pickle._dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':


    # set the hyperparameters for TAII, FW and AAFW
    typ_atyp_ratio = 1
    target_typ_atyp_ratio = 1
    T=20
    step_ratio = (target_typ_atyp_ratio - typ_atyp_ratio)/ T
    atypical_proportion = 0.2
    alpha = 1 # hyper-parameter for weighting TCKD
    beta = 1 #hyper-parameter for weighting NTCKD
    method='TAII'
    perform_kt_transfer('hint')
    perform_kt_transfer('kt_handcrafted')
    perform_kt_transfer('kt', epochs=20, typ_atyp_ratio=typ_atyp_ratio, target_typ_atyp_ratio=target_typ_atyp_ratio, T=T, step_ratio=step_ratio, atypical_proportion=atypical_proportion, alpha=alpha, beta=beta, method=method)
    perform_kt_transfer('kt_supervised')

    ## Additional experiments
    # KT is also stable when a larger learning rate is used
    perform_kt_transfer('kt_optimal')
