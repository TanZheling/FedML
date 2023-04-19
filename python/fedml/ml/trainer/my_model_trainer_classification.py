import torch
from torch import nn

from ...core.alg_frame.client_trainer import ClientTrainer
from ...core.dp.fedml_differential_privacy import FedMLDifferentialPrivacy
import logging
import copy
import logging
from collections import Counter, defaultdict
from sklearn.metrics import roc_auc_score

# from functorch import grad_and_value, make_functional, vmap


class ModelTrainerCLS(ClientTrainer):
    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, args):
        model = self.model

        model.to(device)
        model.train()

        # train and update
        criterion = nn.CrossEntropyLoss().to(device)  # pylint: disable=E1102
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.learning_rate,
            )
        else:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
                amsgrad=True,
            )

        epoch_loss = []
        for epoch in range(args.epochs):
            batch_loss = []

            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                model.zero_grad()
                log_probs = model(x)
                labels = labels.long()
                loss = criterion(log_probs, labels)  # pylint: disable=E1102
                loss.backward()
                optimizer.step()

                # Uncommet this following line to avoid nan loss
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # logging.info(
                #     "Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                #         epoch,
                #         (batch_idx + 1) * args.batch_size,
                #         len(train_data) * args.batch_size,
                #         100.0 * (batch_idx + 1) / len(train_data),
                #         loss.item(),
                #     )
                # )

                batch_loss.append(loss.item())
            if len(batch_loss) == 0:
                epoch_loss.append(0.0)
            else:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
            logging.info(
                "Client Index = {}\tEpoch: {}\tLoss: {:.6f}".format(
                    self.id, epoch, sum(epoch_loss) / len(epoch_loss)
                )
            )

    def train_iterations(self, train_data, device, args):
        model = self.model

        model.to(device)
        model.train()

        # train and update
        criterion = nn.CrossEntropyLoss().to(device)  # pylint: disable=E1102
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.learning_rate,
            )
        else:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
                amsgrad=True,
            )

        epoch_loss = []

        current_steps = 0
        current_epoch = 0
        while current_steps < args.local_iterations:
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                model.zero_grad()
                log_probs = model(x)
                labels = labels.long()
                loss = criterion(log_probs, labels)  # pylint: disable=E1102
                loss.backward()

                # Uncommet this following line to avoid nan loss
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()
                # logging.info(
                #     "Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                #         epoch,
                #         (batch_idx + 1) * args.batch_size,
                #         len(train_data) * args.batch_size,
                #         100.0 * (batch_idx + 1) / len(train_data),
                #         loss.item(),
                #     )
                # )
                batch_loss.append(loss.item())
                current_steps += 1
                if current_steps == args.local_iterations:
                    break
            current_epoch += 1
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            logging.info(
                "Client Index = {}\tEpoch: {}\tLoss: {:.6f}".format(
                    self.id, current_epoch, sum(epoch_loss) / len(epoch_loss)
                )
            )

    def test(self, test_data, device, args):
        model = self.model

        model.to(device)
        model.eval()
        criterion = nn.CrossEntropyLoss().to(device)
        if args.dataset == 'CRCK':
            bag={}
            bag_label={}
            bag_score={}
            bag_pred_dict = defaultdict(list)
            bag_results=[]
            # metrics = {"auc": 0, "test_loss": 0}
            with torch.no_grad():
                for batch_idx, (x, target, bagname) in enumerate(test_data):
                    x = x.to(device)
                    target = target.to(device)
                    pred = model(x)
                    target = target.long()
                    loss = criterion(pred, target)  # pylint: disable=E1102
                    for j in range(len(bagname)):
                        if bagname[j] not in bag:
                            bag[bagname[j]]=0
                            bag_label[bagname[j]]=int(target[j])
                            bag_score[bagname[j]]=0
                        bag_score[bagname[j]]+=pred[j][0]
                        bag[bagname[j]]+=1

                for b in bag_score.keys():
                    mean_0 = bag_score[b] / bag[b]
                    bag_pred_dict[b]=[mean_0, 1 - mean_0]
                bag_results = [kv[1][1].cpu() for kv in sorted(bag_pred_dict.items(), key=lambda x: x[0])]
                bag_labels = [kv[1] for kv in sorted(bag_label.items(), key=lambda x: x[0])]
                aucs = roc_auc_score(bag_labels,bag_results)
            metrics = {"auc":aucs}
            
        else:
            metrics = {"test_correct": 0, "test_loss": 0, "test_total": 0}

            # criterion = nn.CrossEntropyLoss().to(device)

            with torch.no_grad():
                for batch_idx, (x, target) in enumerate(test_data):
                    x = x.to(device)
                    target = target.to(device)
                    pred = model(x)
                    target = target.long()
                    loss = criterion(pred, target)  # pylint: disable=E1102

                    _, predicted = torch.max(pred, -1)
                    correct = predicted.eq(target).sum()

                    metrics["test_correct"] += correct.item()
                    metrics["test_loss"] += loss.item() * target.size(0)
                    metrics["test_total"] += target.size(0)
        return metrics
