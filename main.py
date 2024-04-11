import logging
import torch
import dgl
import numpy as np
from tqdm import tqdm

from DomainData import DomainData
from evaluation import node_classification_evaluation_SSL
from model import build_model
from utils import set_random_seed, TBLogger, build_args, load_best_configs, create_optimizer, get_current_lr

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


def pretrain(model, graph_target, feat_target, graph_source, feat_source, y_source, optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f,
             max_epoch_f, linear_prob, logger=None):
    logging.info("start training..")

    graph_target = graph_target.to(device)
    x_target = feat_target.to(device)
    graph_source = graph_source.to(device)
    x_source = feat_source.to(device)
    y_source = y_source.to(device)
    epoch_iter = tqdm(range(max_epoch))
    for epoch in epoch_iter:
        model.train()
        loss, loss_dict = model(graph_target, x_target, graph_source, x_source, y_source)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        epoch_iter.set_description(f"# Epoch {epoch}: train_loss: {loss.item():.4f}")
        if logger is not None:
            loss_dict["lr"] = get_current_lr(optimizer)
            logger.note(loss_dict, step=epoch)
        if (epoch + 1) % 200 == 0:
            final_acc, estp_acc, estp_test_pre, estp_test_rec, estp_test_f1, estp_test_auc = node_classification_evaluation_SSL(model, graph_target, x_target, num_classes, lr_f, weight_decay_f, max_epoch_f, device,
                                           linear_prob, mute=True)
            print(f"# final_acc: {final_acc:.3f}")
            print(f"# early-stopping_acc: {estp_acc:.3f}")
            print(f"# early-stopping_pre: {estp_test_pre:.3f}")
            print(f"# early-stopping_rec: {estp_test_rec:.3f}")
            print(f"# early-stopping_f1: {estp_test_f1:.3f}")
            print(f"# early-stopping_auc: {estp_test_auc:.3f}")

            if logger is not None:
                logger.finish()
    return model


def preprocess(graph):
    graph = graph.remove_self_loop().add_self_loop()
    graph.create_formats_()
    return graph

def u_cat_e(edges):
  return {'m': torch.hstack([edges.src['feature'],edges.data['feature']])}

def mean_udf(nodes):
    return {'neigh_features': nodes.mailbox['m'].mean(1)}

def data_split(y,train_size, val_size):
    seeds = args.seeds
    for i, seed in enumerate(seeds):
        set_random_seed(seed)
    random_node_indices = np.random.permutation(y.shape[0])
    training_size = int(len(random_node_indices) * train_size)
    val_size = int(len(random_node_indices) * val_size)
    train_node_indices = random_node_indices[:training_size]
    val_node_indices = random_node_indices[training_size:training_size + val_size]
    test_node_indices = random_node_indices[training_size+ val_size :]
    train_masks = torch.zeros([y.shape[0]], dtype=torch.uint8)
    train_masks[train_node_indices] = 1
    val_masks = torch.zeros([y.shape[0]], dtype=torch.uint8)
    val_masks[val_node_indices] = 1
    test_masks = torch.zeros([y.shape[0]], dtype=torch.uint8)
    test_masks[test_node_indices] = 1
    return train_masks,val_masks,test_masks
def main(args):
    device = args.device if args.device >= 0 else "cpu"
    seeds = args.seeds
    dataset_name = args.dataset
    max_epoch = args.max_epoch
    max_epoch_f = args.max_epoch_f
    num_hidden = args.num_hidden
    num_layers = args.num_layers
    encoder_type = args.encoder
    decoder_type = args.decoder
    replace_rate = args.replace_rate
    target = args.target
    source = args.source

    optim_type = args.optimizer
    loss_fn = args.loss_fn

    lr = args.lr
    weight_decay = args.weight_decay
    lr_f = args.lr_f
    weight_decay_f = args.weight_decay_f
    linear_prob = args.linear_prob
    load_model = args.load_model
    save_model = args.save_model
    logs = args.logging
    use_scheduler = args.scheduler

    dataset_target = DomainData("data/{}".format(args.target), name=target)
    dataset_source = DomainData("data/{}".format(args.source), name=source)

    target_data = dataset_target[0]
    t_src = target_data.edge_index[0]
    t_dst = target_data.edge_index[1]
    graph_target = dgl.graph((t_src, t_dst))
    graph_target = dgl.to_bidirected(graph_target)
    graph_target = graph_target.remove_self_loop().add_self_loop()
    graph_target.create_formats_()

    source_data = dataset_source[0]
    s_src = source_data.edge_index[0]
    s_dst = source_data.edge_index[1]
    graph_source = dgl.graph((s_src, s_dst))
    graph_source = dgl.to_bidirected(graph_source)
    graph_source = graph_source.remove_self_loop().add_self_loop()
    graph_source.create_formats_()

    '''target data split'''
    s_train_masks, s_val_masks, s_test_masks = data_split(y=source_data.y,train_size=1.0,val_size=0.0)
    t_train_masks, t_val_masks, t_test_masks = data_split(y=target_data.y,train_size=0.7,val_size=0.1)
    print('graph_target: ', graph_target,' graph_source: ',graph_source)

    print('target_data: ',target_data,' source_data: ',source_data)
    # #
    graph_target.ndata['feat'] = target_data.x
    graph_target.ndata['label'] = target_data.y
    graph_target.ndata['train_mask'] = t_train_masks
    graph_target.ndata['val_mask'] = t_val_masks
    graph_target.ndata['test_mask'] = t_test_masks
    print('graph_target.ndatatrain_mask ', graph_target.ndata['train_mask'][:10],' target_data.val_mask: ',graph_target.ndata['val_mask'][:10],' target_data.test_mask: ',graph_target.ndata['test_mask'][:10])

    graph_source.ndata['feat'] = source_data.x
    graph_source.ndata['label'] = source_data.y
    graph_source.ndata['train_mask'] = s_train_masks
    graph_source.ndata['val_mask'] = s_val_masks
    graph_source.ndata['test_mask'] = s_test_masks

    num_features = args.features
    num_classes = args.classes
    args.num_features = num_features

    for i, seed in enumerate(seeds):
        print(f"####### Run {i} for seed {seed}")
        set_random_seed(seed)

        if logs:
            logger = TBLogger(
                name=f"{dataset_name}_loss_{loss_fn}_rpr_{replace_rate}_nh_{num_hidden}_nl_{num_layers}_lr_{lr}_mp_{max_epoch}_mpf_{max_epoch_f}_wd_{weight_decay}_wdf_{weight_decay_f}_{encoder_type}_{decoder_type}")
        else:
            logger = None
        model = build_model(args).to(device)
        optimizer = create_optimizer(optim_type, model, lr, weight_decay)

        if use_scheduler:
            logging.info("Use schedular")
            scheduler = lambda epoch: (1 + np.cos((epoch) * np.pi / max_epoch)) * 0.5
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
        else:
            scheduler = None

        x_target = graph_target.ndata["feat"]
        x_source = graph_source.ndata["feat"]
        y_source = graph_source.ndata["label"]
        if not load_model:
            pretrain(model, graph_target, x_target, graph_source, x_source, y_source, optimizer, max_epoch, device, scheduler, num_classes, lr_f,
                             weight_decay_f, max_epoch_f, linear_prob, logger)

if __name__ == "__main__":
    args = build_args()
    print(args)
    main(args)
