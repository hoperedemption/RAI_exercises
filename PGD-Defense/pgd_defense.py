import torch
import torch.nn as nn
import torch.nn.functional as F
from fgsm.fgsm import fgsm, fgsm_targeted, fgsm_untargeted
from PGD.pgd import pgd

# this is a code snippet, to compile you can add tqdm, argparse etc

# important note for the PGD/TRADES defense, as well as for the attack,
# the untargeted version of fgsm has target label the gt label in a supervised setting

# Implementation of the PGD and TRADES defense training loop
for epoch in range(1, args.num_epochs + 1):
    # Training
    for batch_idx, (x_batch, y_batch) in enumerate(tqdm(train_loader)):

        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        # zero out the gradients for every batch
        opt.zero_grad()

        if args.defense == 'PGD':
            # evaluate model
            model.eval()
            # run PGD attack on model
            vechev_constant = 2.5 * args.eps / args.k
            x_max_batch = pgd(model, x_batch, y_batch, args.k, args.eps, vechev_constant) # in a supervised learning framework
            # we already know what the function outputs of the function should be, hence we just give y_batch
            # set model to train
            model.train()
            # model is now in training mode, where it trains on most problematic samples
            loss = ce_loss(model(x_max_batch), y_batch)

        elif args.defense == 'TRADES':
            # switch to training mode 
            model.train()
            logits = model(x_batch)
            proba = F.softmax(logits.detach(), dim=-1)
            # evaluate model
            model.eval()
            # run PGD attack
            vechev_constant = 2.5 * args.eps / args.k
            x_max_batch = pgd(model, x_batch, proba, args.k, args.eps, vechev_constant)
            # train model
            model.train()
            # evaluate the loss on x_batch
            loss_batch = ce_loss(model(x_batch), y_batch)
            loss_max_batch = ce_loss(model(x_max_batch), y_batch)
            loss = loss_batch + args.trades_fact * loss_max_batch

        loss.backward()
        opt.step()

    # Testing
    for batch_idx, (x_batch, y_batch) in enumerate(tqdm(test_loader)):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        out = model(x_batch)
        pred = torch.max(out, dim=1)[1]
        equal_mask = pred == y_batch
        acc = equal_mask.sum().item()

        # calculate accuracy under PGD attack
        # we don't do no grad since PGD requires gradient computation
        vechev_constant = 2.5 * args.eps / args.k

        x_adv = pgd(model, x_batch, y_batch, args.k, args.eps, vechev_constant)
        acc_adv = model(x_adv).argmax(dim=-1).eq(y_batch).sum().item()
        # A more correct defintion of adversarial robustness would use the following code:
        #eq_indices = torch.nonzero(equal_mask).flatten()
        #x_adv = pgd(model, x_batch[eq_indices], y_batch[eq_indices], args.k, args.eps, vechev_constant)
        #acc_adv = model(x_adv).argmax(dim=-1).eq(y_batch[eq_indices]).sum().item()

        # Accumulate accuracies
        tot_acc += acc
        tot_adv_acc += acc_adv
        tot_test += x_batch.size()[0]
