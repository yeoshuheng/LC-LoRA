import os
import torch
from torchvision import datasets
from torchvision import transforms
import ssl
import json
import numpy as np
import src.main as lc
from src.models.AlexNet import AlexNet
from src.models.AlexNet_LowRank import getBase, AlexNet_LowRank, load_sd_decomp
from src.utils.utils import evaluate_accuracy, lazy_restore, evaluate_compression

# Save path
HDFP = "/volumes/Ultra Touch"
# Rank of decomposition to test
RANK = 64 # 3 1

def data_loader():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])

    trainset = datasets.MNIST(root='./data', train=True,
                                          download=True, transform=transform)
    trainset.data = trainset.data
    trainset.targets = trainset.targets
    trainloader = torch.utils.data.DataLoader(trainset, batch_size = 32,
                                              shuffle=False, num_workers=2)

    testset = datasets.MNIST(root='./data', train=False,
                                         download=True, transform=transform)
    testset.data = testset.data
    testset.targets = testset.targets
    testloader = torch.utils.data.DataLoader(testset, batch_size = 32,
                                             shuffle=False, num_workers=2)
    return trainloader, testloader

def main():
    # Bypass using SSL unverified
    ssl._create_default_https_context = ssl._create_unverified_context
    # MNIST dataset 
    train_loader, test_loader = data_loader()

    SAVE_LOC = HDFP + "/lobranch-snapshot/lobranch/from_scratch"
    SAVE_LOC_FULL = HDFP + "/lobranch-snapshot/full/from_scratch"
    if not os.path.exists(SAVE_LOC):
        os.makedirs(SAVE_LOC)
    if not os.path.exists(SAVE_LOC_FULL):
        os.makedirs(SAVE_LOC_FULL)

    DECOMPOSED_LAYERS = ["classifier.1.weight", "classifier.4.weight"]

    # Set up weights for original AlexNet model
    original = AlexNet()
    model_original = AlexNet()

    # Load from "branch point" => not needed in "from scratch" training.
    #BRANCH_LOC = HDFP + "/sim-test/alexnet/full/model-0.709.pt"
    #original.load_state_dict(torch.load(BRANCH_LOC))
    #model_original.load_state_dict(torch.load(BRANCH_LOC))

    w, b = getBase(model_original)
    model = AlexNet_LowRank(w, b, rank = 100)
    #load_sd_decomp(torch.load(BRANCH_LOC), model, DECOMPOSED_LAYERS)
    load_sd_decomp(model_original.state_dict(), model, DECOMPOSED_LAYERS)
    learning_rate = 0.01
    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
    optimizer_full = torch.optim.SGD(model_original.parameters(), lr = learning_rate)

    full_accuracy = []
    decomposed_full_accuracy = []
    restored_accuracy = []
    current_iter = 0
    current_set = 0

    acc = lambda x, y : (torch.max(x, 1)[1] == y).sum().item() / y.size(0)

    for epch in range(15):
        for i, data in enumerate(train_loader, 0):
            print("Epoch: {}, Iteration: {}".format(epch, i))
            
            set_path = "/set_{}".format(current_set)
            if not os.path.exists(SAVE_LOC + set_path):
                os.makedirs(SAVE_LOC + set_path)
                os.makedirs(SAVE_LOC_FULL + set_path)

            if i == 0 and epch == 0: # first iteration, create baseline model
                base, base_decomp = lc.extract_weights(model, SAVE_LOC + 
                                                        "/set_{}".format(current_set), DECOMPOSED_LAYERS)
            else:
                if i % 10 == 0: 
                    # full snapshot!
                    new_model = lazy_restore(base, base_decomp, bias, AlexNet(), RANK, 
                                            original.state_dict(), DECOMPOSED_LAYERS)
                    original = new_model # Changing previous "original model" used to restore the loRA model.
                    
                    current_set += 1
                    current_iter = 0

                    set_path = "/set_{}".format(current_set)
                    if not os.path.exists(SAVE_LOC + set_path):
                        os.makedirs(SAVE_LOC + set_path)
                        os.makedirs(SAVE_LOC_FULL + set_path)
                    
                    # Rebuilding LoRA layers => reset model!
                    w, b = getBase(original)
                    model = AlexNet_LowRank(w, b, rank = RANK)
                    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
                    load_sd_decomp(original.state_dict(), model, DECOMPOSED_LAYERS)
                    base, base_decomp = lc.extract_weights(model, SAVE_LOC + 
                                                        "/set_{}".format(current_set), DECOMPOSED_LAYERS)

                else:
                    # Delta-compression
                     # Delta-compression
                    delta, decomp_delta, bias = lc.generate_delta(base, 
                                                                    base_decomp, model.state_dict(), DECOMPOSED_LAYERS)
                    compressed_delta, full_delta, compressed_dcomp_delta, full_dcomp_delta  = lc.compress_delta(delta, decomp_delta)

                    
                    # Saving checkpoint
                    lc.save_checkpoint(compressed_delta, compressed_dcomp_delta, bias, current_iter, SAVE_LOC + 
                                    "/set_{}".format(current_set))
        
                    base = np.add(base, full_delta) # Replace base with latest for delta to accumulate.
                    base_decomp = np.add(full_dcomp_delta, base_decomp)

                    current_iter += 1
            
            # ==========================
            # Training on Low-Rank Model
            # ==========================

            # Get the inputs and labels
            inputs, labels = data

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(inputs)
            loss = torch.nn.functional.cross_entropy(outputs,labels)
            loss.backward()
            optimizer.step()
                
            # ======================
            # Training on Full Model
            # ======================

            # Zero the parameter gradients
            optimizer_full.zero_grad()

            # Forward + backward + optimize
            outputs_full = model_original(inputs)
            loss_full = torch.nn.functional.cross_entropy(outputs_full,labels)
            loss_full.backward()
            optimizer_full.step()

            if i % 20 == 0:
                print("Training Accuracy | Decomposed: {}, Full : {}".format(acc(outputs, labels), 
                                                                            acc(outputs_full, labels)))

            if i != 0  and i % 100 == 0: # Evaluation on testing set
                full_accuracy.append(evaluate_accuracy(model_original, test_loader))
                decomposed_full_accuracy.append(evaluate_accuracy(model, test_loader))
                restored_model = lazy_restore(base, base_decomp, bias, AlexNet(), RANK, 
                                                        original.state_dict(), DECOMPOSED_LAYERS)
                restored_accuracy.append(evaluate_accuracy(restored_model, test_loader))
                print("Full accuracy: {}, Decomposed-Full accuracy: {}, Decomposed-Restored accuracy:: {}".format(
                    full_accuracy[-1], decomposed_full_accuracy[-1], restored_accuracy[-1]))
            

    write_path = HDFP + "/lobranch-snapshot/from-scratch-acc-results_{}.json".format(RANK)
    with open(write_path, 'w') as f:
        json.dump({"full_model" : full_accuracy, 
                    "decomposed_model" : decomposed_full_accuracy, 
                    "decomposed_restored" : restored_accuracy}, f)