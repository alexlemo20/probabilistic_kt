import sys
sys.path.append("/media/data/alexlemo/probabilistic_kt")
# pip install --force-reinstall torch==1.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/

from nn.nn_utils import load_model, save_model
from nn.distillation import unsupervised_distillation
from nn.pkt import knowledge_transfer
from exp_cifar.cifar_dataset import cifar10_loader, cifar100_loader
from models.cifar_tiny import Cifar_Tiny
from models.resnet import ResNet18
from nn.hint_transfer import unsupervised_hint_transfer, unsupervised_hint_transfer_optimized
from nn.retrieval_evaluation import evaluate_model_retrieval


def perform_transfer_knowledge(net, donor_net, transfer_loader, output_path, transfer_method, distill_temp=2,
                               learning_rates=(0.0001, 0.0001), iters=(1, 1)):
    # Move the models into GPU
    net.cuda()
    donor_net.cuda()
    typ_atyp_ratio = 2
    target_typ_atyp_ratio = 1
    step_ratio = (target_typ_atyp_ratio - typ_atyp_ratio)/iters[0]

    #typical_mask = 2 
    #atypical_mask = 1
    #typical_target = 1
    #atypical_target = 2 
    atypical_proportion = 0.2
    alpha = 1 # hyper-parameter for weighting TCKD
    beta = 0.5 #hyper-parameter for weighting NTCKD

    #typical_step = (typical_target - typical_mask) / iters[0]
    #atypical_step = (atypical_target - atypical_mask) / iters[0]
    
    # Perform the transfer
    W = None
    for lr, iters in zip(learning_rates, iters):
        #if transfer_method == 'hint':
        #    W = unsupervised_hint_transfer(net, donor_net, transfer_loader, epochs=iters, lr=lr, W=W)
        #elif transfer_method == 'hint_optimized':
        #    W = unsupervised_hint_transfer_optimized(net, donor_net, transfer_loader, epochs=iters, lr=lr, W=W)
        #elif transfer_method == 'distill':
        #    unsupervised_distillation(net, donor_net, transfer_loader, epochs=iters, lr=lr, T=distill_temp)
        if transfer_method == 'pkt':
            #knowledge_transfer(net, donor_net, transfer_loader, epochs=iters, lr=lr, typical_mask=typical_mask, typical_step=typical_step, atypical_mask=atypical_mask, atypical_step=atypical_step, atypical_proportion=atypical_proportion, alpha=alpha, beta=beta)
            knowledge_transfer(net, donor_net, transfer_loader, epochs=iters, lr=lr, typ_atyp_ratio=typ_atyp_ratio, step_ratio=step_ratio, atypical_proportion=atypical_proportion, alpha=alpha, beta=beta)
            #knowledge_transfer(net, donor_net, transfer_loader, epochs=iters, lr=lr)

        else:
            assert False

    save_model(net, output_path)
    print("Model saved at ", output_path)


def evaluate_kt_methods(net_creator, donor_creator, donor_path, transfer_loader, batch_size=128,
                        donor_name='very_small_cifar10', net_name='tiny_cifar', transfer_name='cifar10',
                        iters=100, init_model_path=None):
    # Method 1: HINT transfer
    #net = net_creator()
    #if init_model_path is not None:
    #    load_model(net, init_model_path)

    #donor_net = donor_creator()
    #load_model(donor_net, donor_path)

    #train_loader, test_loader, train_loader_raw = transfer_loader(batch_size=batch_size)
    #output_path = 'models/' + net_name + '_' + donor_name + '_hint_' + transfer_name + '.model'
    #results_path = 'results/' + net_name + '_' + donor_name + '_hint_' + '_' + transfer_name + '.pickle'

    #perform_transfer_knowledge(net, donor_net, transfer_loader=train_loader, transfer_method='hint',
    #                           output_path=output_path, iters=[iters], learning_rates=[0.0001])
    #evaluate_model_retrieval(net=Cifar_Tiny(num_classes=10), path=output_path, result_path=results_path)

    # Method 2: Distillation transfer
    #net = net_creator()
    #if init_model_path is not None:
    #    load_model(net, init_model_path)

    #donor_net = donor_creator()
    #load_model(donor_net, donor_path)

    #train_loader, test_loader, train_loader_raw = transfer_loader(batch_size=batch_size)
    #output_path = 'models/' + net_name + '_' + donor_name + '_distill_' + transfer_name + '.model'
    #results_path = 'results/' + net_name + '_' + donor_name + '_distill_' + transfer_name + '.pickle'
    #perform_transfer_knowledge(net, donor_net, transfer_loader=train_loader, transfer_method='distill',
    #                           output_path=output_path, iters=[iters], learning_rates=[0.0001])
    #evaluate_model_retrieval(net=Cifar_Tiny(num_classes=10), path=output_path, result_path=results_path)

    # Method 3: PKT transfer
    net = net_creator()
    if init_model_path is not None:
        load_model(net, init_model_path)
    donor_net = donor_creator()
    load_model(donor_net, donor_path)

    train_loader, test_loader, train_loader_raw = transfer_loader(batch_size=batch_size)
    output_path = 'models/' + net_name + '_' + donor_name + '_kt_' + transfer_name + '.model'
    results_path = 'results/' + net_name + '_' + donor_name + '_kt_' + transfer_name + '.pickle'
    perform_transfer_knowledge(net, donor_net, transfer_loader=train_loader, transfer_method='pkt',
                               output_path=output_path, iters=[iters], learning_rates=[0.0001]) # COMMENT OUT
    evaluate_model_retrieval(net=Cifar_Tiny(num_classes=100), path=output_path, result_path=results_path, dataset_name='cifar100', dataset_loader=cifar100_loader)


    # Method 4: HINT (optimized) transfer
    #net = net_creator()
    #if init_model_path is not None:
    #    load_model(net, init_model_path)

    #donor_net = donor_creator()
    #load_model(donor_net, donor_path)

    #train_loader, test_loader, train_loader_raw = transfer_loader(batch_size=batch_size)
    #output_path = 'models/' + net_name + '_' + donor_name + '_hint_optimized_' + transfer_name + '.model'
    #results_path = 'results/' + net_name + '_' + donor_name + '_hint_optimized_' + '_' + transfer_name + '.pickle'
    #perform_transfer_knowledge(net, donor_net, transfer_loader=train_loader, transfer_method='hint_optimized',
    #                           output_path=output_path, iters=[iters], learning_rates=[0.0001])
    #evaluate_model_retrieval(net=Cifar_Tiny(num_classes=10), path=output_path, result_path=results_path)

if __name__ == '__main__':
    evaluate_kt_methods(lambda: Cifar_Tiny(num_classes=100), lambda: ResNet18(num_classes=100), 'models/resnet18_cifar100.model',
                        cifar10_loader, batch_size=64, donor_name='resnet18_cifar100', transfer_name='cifar100',
                        iters=20, net_name='cifar_tiny', init_model_path='models/tiny_cifar100.model')


