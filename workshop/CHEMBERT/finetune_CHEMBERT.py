import re
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader

from pathlib import Path
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix

from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, r2_score, auc, confusion_matrix

## CHEMBERT MODEL
from CHEMBERT.model import Smiles_BERT, BERT_base

module_dir = Path(__file__).parent

class Vocab(object):
    def __init__(self):
        self.pad_index   = 0
        self.mask_index  = 1
        self.unk_index   = 2
        self.start_index = 3
        self.end_index   = 4

        # check 'Na' later
        self.voca_list = ['<pad>', '<mask>', '<unk>', '<start>', '<end>'] + ['C', '[', '@', 'H', ']', '1', 'O', \
                            '(', 'n', '2', 'c', 'F', ')', '=', 'N', '3', 'S', '/', 's', '-', '+', 'o', 'P', \
                             'R', '\\', 'L', '#', 'X', '6', 'B', '7', '4', 'I', '5', 'i', 'p', '8', '9', '%', '0', '.', ':', 'A']

        self.dict = {s: i for i, s in enumerate(self.voca_list)}

    def __len__(self):
        return len(self.voca_list)

class FinetuningDataset(Dataset):
    def __init__(self, data_df, vocab, seq_len):
        # data_df: pandas dataframe with 2 columns: SMILES, LABELS
        self.vocab = vocab
        self.atom_vocab = ['C', 'O', 'n', 'c', 'F', 'N', 'S', 's', 'o', 'P', 'R', 'L', 'X', 'B', 'I', 'i', 'p', 'A']
        self.smiles_dataset = []
        self.adj_dataset = []
        
        self.seq_len = seq_len

        assert len(data_df.columns) == 2, "ERROR: File must have 2 exactly columns, SMILES and LABELS."

        smiles_list = np.asarray(data_df['SMILES'].values)
        label_list  = np.asarray(data_df['LABELS'].values)

        self.label = label_list.reshape(-1,1)
        for i in smiles_list:
            self.adj_dataset.append(i)
            self.smiles_dataset.append(self.replace_halogen(i))

    def __len__(self):
        return len(self.smiles_dataset)

    def __getitem__(self, idx):
        item = self.smiles_dataset[idx]
        label = self.label[idx]

        input_token, input_adj_masking = self.CharToNum(item)

        input_data = [self.vocab.start_index] + input_token + [self.vocab.end_index]
        input_adj_masking = [0] + input_adj_masking + [0]

        smiles_bert_input = input_data[:self.seq_len]
        smiles_bert_adj_mask = input_adj_masking[:self.seq_len]

        padding = [0 for _ in range(self.seq_len - len(smiles_bert_input))]
        smiles_bert_input.extend(padding)
        smiles_bert_adj_mask.extend(padding)

        smiles=self.adj_dataset[idx]
        mol = Chem.MolFromSmiles(smiles)
        if mol != None:
            adj_mat = GetAdjacencyMatrix(mol)
            smiles_bert_adjmat = self.zero_padding(adj_mat, (self.seq_len, self.seq_len))
        else:
            print("BAD SMILES:", smiles)
            smiles_bert_adjmat = np.zeros((self.seq_len, self.seq_len), dtype=np.float32)

        output = {"smiles_bert_input": smiles_bert_input, "smiles_bert_label": label,  \
                    "smiles_bert_adj_mask": smiles_bert_adj_mask, "smiles_bert_adjmat": smiles_bert_adjmat}

        return {key:torch.tensor(value) for key, value in output.items()}

    def CharToNum(self, smiles):
        tokens = [i for i in smiles]
        adj_masking = []

        for i, token in enumerate(tokens):
            if token in self.atom_vocab:
                adj_masking.append(1)
            else:
                adj_masking.append(0)

            tokens[i] = self.vocab.dict.get(token, self.vocab.unk_index)

        return tokens, adj_masking


    def replace_halogen(self,string):
        """Regex to replace Br and Cl with single letters"""
        br = re.compile('Br')
        cl = re.compile('Cl')
        sn = re.compile('Sn')
        na = re.compile('Na')
        string = br.sub('R', string)
        string = cl.sub('L', string)
        string = sn.sub('X', string)
        string = na.sub('A', string)
        return string

    def zero_padding(self, array, shape):
        if array.shape[0] > shape[0]:
            array = array[:shape[0],:shape[1]]
        padded = np.zeros(shape, dtype=np.float32)
        padded[:array.shape[0], :array.shape[1]] = array
        return padded


def finetune_model(data_df,
                   pretrained_model=f'{module_dir}/model/pretrained_model.pt',
                   task='regression',split_ratio=0.8,max_time=720, max_epochs=15,
                   cpu_threads=24):


    print("#"*60)
    print(f"{'FINE-TUNING CHEMBERT MODEL':^60s}")
    print("#"*60)
    print("Pre-trained model:", pretrained_model)
    print("Task = ", task)
    print("Maximum time = ", max_time)
    print("\n")
    
    # PyTorch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available() is False:
        print(f"WARNING: USING CPU for Finetuning with {cpu_threads} threads.")
        torch.set_num_threads(cpu_threads)

    # Pre-trained model parameters
    # Those are the parameters used for training CHEMBERT.
    params = {'model':'Transformer',
              'optimizer':'Adam', 'batch_size':64,
              'dropout':0, 'learning_rate':0.00001}

    Smiles_vocab = Vocab()
    
    # Reads data
    dataset = FinetuningDataset(data_df, Smiles_vocab, seq_len=256)

    # Splits data
    
    labels = dataset.label

    if task == 'classification':
        #bin_labels = np.array([0 if i < 0.5 else 1 for i in labels])
        bin_labels = np.array(labels)
    else:
        kbd = KBinsDiscretizer(3, encode="ordinal", strategy="quantile")
        kbd.fit(labels)
        limits = kbd.bin_edges_[0]
        bin_labels = kbd.transform(labels)
        print("-"*35)
        print("Stratified dataset:")
        print(f"{'Group':>5s}\t{'Count':>5s}\t{'Interval':>13s}")
        for i in range(kbd.n_bins_[0]):
            count = np.sum(bin_labels == i)
            print(f"{i:>5d}\t{count:>5d}\t[{limits[i]:6.2f} {limits[i+1]:6.2f}]")
        print("-"*35)
    indices=list(range(len(dataset)))

    # First, split in general train/test
    X_train, X_testi, y_train, y_testi = train_test_split(indices, labels, 
                                                train_size=split_ratio,
                                                shuffle=True, stratify=bin_labels)

    # Now, break the test set into test and validation
    bin_labels = np.take(bin_labels, X_testi)
    X_test, X_valid, y_test,  y_valid  = train_test_split(X_testi,y_testi,
                                  train_size=0.5,
                                  shuffle=True, stratify=bin_labels)

    print("Final split sizes:")
    print(f"Training set:  {len(X_train):10,d}")
    print(f"Testing set:   {len(X_test) :10,d}")
    print(f"Validation set:{len(X_valid):10,d}")

    # Create the necessary 'chemebert' directiory.
    # If existent, cleanup first
    if Path('chembert').is_dir():
        for file in Path("chembert").glob("*"):
            file.unlink()
        Path('chembert').rmdir()
    Path('chembert').mkdir()

    pd.DataFrame({'SMILES':[dataset.adj_dataset[i] for i in X_train],
                  'LABELS':y_train.flatten().tolist()}).to_csv('chembert/train.smi', index=None)
    pd.DataFrame({'SMILES':[dataset.adj_dataset[i] for i in X_valid],
                  'LABELS':y_valid.flatten().tolist()}).to_csv('chembert/valid.smi', index=None)
    pd.DataFrame({'SMILES':[dataset.adj_dataset[i] for i in X_test ],
                  'LABELS':y_test.flatten().tolist() }).to_csv('chembert/test.smi' , index=None)

    
    train_sampler = SubsetRandomSampler(X_train)
    valid_sampler = SubsetRandomSampler(X_valid)
    test_sampler  = SubsetRandomSampler(X_test)

    # dataloader(train, valid, test)
    train_dataloader = DataLoader(dataset, batch_size=params['batch_size'], sampler=train_sampler, num_workers=4, pin_memory=True)
    valid_dataloader = DataLoader(dataset, batch_size=params['batch_size'], sampler=valid_sampler, num_workers=4)
    test_dataloader  = DataLoader(dataset, batch_size=params['batch_size'], sampler=test_sampler , num_workers=4)

    # Load pre-trained model model
    model = Smiles_BERT(len(Smiles_vocab),
                        max_len=256, nhead=16,
                        feature_dim=1024, feedforward_dim=1024,
                        nlayers=8, adj=True,
                        dropout_rate=params['dropout'])
    model.load_state_dict(torch.load(pretrained_model, map_location=device))
    output_layer = nn.Linear(1024, 1)

    model = BERT_base(model, output_layer)
    model.to(device)
    if torch.cuda.device_count() > 1:
        print("Using GPU acceleration")
        model = nn.DataParallel(model)

    #optim = Adam(model.parameters(), lr=params['learning_rate'], coeff_decay=0)
    optim = Adam(model.parameters(), lr=params['learning_rate'])
    if task == 'classification':
        print("Classification task. Criterion = BCEWithLogitsLoss")
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.MSELoss()

    test_crit = nn.MSELoss(reduction='none')

    train_loss_list  = []
    valid_loss_list  = []
    valid_score_list = []
    test_score_list  = []
    test_result_list = []

    min_valid_loss = 10000
    start_time = time.time()

    # #####################
    #    Start training
    # #####################
    print(f"\n{'='*35}")
    print(f"{'FINETUNING MODEL':^35s}")
    print("="*35)
    print(f"Attempting up to {max_epochs} epochs, or {max_time} minutes.")
    for epoch in range(1,max_epochs+1):
        epoch_start = time.time()
        print(f"\nEpoch {epoch} of a maximum of {max_epochs}.")
        avg_loss = 0
        valid_avg_loss = 0
        valid_rmse = []
        test_rmse = []
        test_pred = []
        test_true = []
        predicted_list = np.array([])
        target_list = np.array([])

        model.train() # Equivalent to model.train(True)
        for i, data in enumerate(tqdm(train_dataloader, desc="Training  ", 
                                      leave=False, ncols=80, unit='batch')):
            # train set
            data = {key:value.to(device) for key, value in data.items()}
            position_num = torch.arange(256).repeat(data["smiles_bert_input"].size(0),1).to(device)
            output = model.forward(data["smiles_bert_input"], position_num,
                                   adj_mask=data["smiles_bert_adj_mask"],
                                   adj_mat=data["smiles_bert_adjmat"])
            output = output[:,0].double()

            loss = criterion(output, data["smiles_bert_label"])
            optim.zero_grad()
            loss.backward()
            optim.step()

            avg_loss += loss.item()
        train_loss_list.append(avg_loss / len(train_dataloader))
        model.eval() # Equivalent to model.train(False)

        with torch.no_grad():
            # Ths only difference between valid and test sets is that the loss
            # is only calculated for the valid set, then this loss is used as
            # criterion to accept the new model or not.

            # The predictions on the test set are always calculated and stored,
            # but only the predictions for the 'best' model (lower validation loss)
            # is printed and plotted in the end

            # validation set
            # --------------
            for i, data in enumerate(tqdm(valid_dataloader, desc="Validating",
                                          leave=False, ncols=80, unit='batch')):
                data = {key:value.to(device) for key, value in data.items()}
                position_num = torch.arange(256).repeat(data["smiles_bert_input"].size(0),1).to(device)
                output = model.forward(data["smiles_bert_input"], position_num, adj_mask=data["smiles_bert_adj_mask"], adj_mat=data["smiles_bert_adjmat"])
                output = output[:,0]
                valid_loss = criterion(output, data["smiles_bert_label"])
                valid_avg_loss += valid_loss.item()

                if task == 'classification':
                    predicted = torch.sigmoid(output)
                    predicted_list = np.append(predicted_list,
                                               predicted.cpu().detach().numpy())
                    target_list    = np.append(target_list,
                                               data["smiles_bert_label"].cpu().detach().numpy())
                else:
                    test_loss = torch.sqrt(test_crit(output, data["smiles_bert_label"]))
                    valid_rmse.append(test_loss)
            valid_avg_loss = valid_avg_loss / len(valid_dataloader)
            valid_loss_list.append(valid_avg_loss)

            if task == 'classification':
                predicted_list = np.reshape(predicted_list, (-1))
                target_list = np.reshape(target_list, (-1))
                auc_score = roc_auc_score(target_list, predicted_list)
                valid_score_list.append(auc_score)
                #print(f"ROC_AUC Score on Validation Set = {auc_score:5.3f}")

                # plots the data
                fig, ax = plt.subplots()
                disc_pred = [0.0 if i < 0.5 else 1.0 for i in predicted_list]
                match_list = [x==y for (x,y) in zip(target_list,disc_pred)]
                match_lbl  = ['Correct' if x==y else 'Incorrect' for (x,y) in zip(target_list,disc_pred)]
                sns.stripplot(x=target_list, y=predicted_list, 
                              hue=match_lbl, palette={'Incorrect':'red','Correct':'green'},
                              jitter=True, alpha=0.5, ax=ax)
                ax.set_xlabel('True Values')
                ax.set_ylabel('Predictions')
                plt.suptitle(f'Validation set results for epoch {epoch}')
                ax.set_title(f'ROC_AUC Score on Validation Set= {auc_score:5.3f}')

                # Includes a confusion matrix in the plot
                cm = confusion_matrix(target_list, disc_pred)

                ax.vlines(x=0.5, ymin=0, ymax=1.0, alpha=0.5)
                ax.hlines(y=0.5, xmin=0, xmax=1.0, alpha=0.5)
                ax.annotate(f"Q1: {cm[1,1]}", xy=(0.55, 0.55), alpha=0.5)
                ax.annotate(f"Q2: {cm[0,1]}", xy=(0.32, 0.55), alpha=0.5)
                ax.annotate(f"Q3: {cm[0,0]}", xy=(0.32, 0.45), alpha=0.5)
                ax.annotate(f"Q4: {cm[1,0]}", xy=(0.55, 0.45), alpha=0.5)
                ax.legend(fancybox=True, framealpha=0.5)
                plt.show()
            else:
                valid_rmse = torch.cat(valid_rmse, dim=0).cpu().numpy()
                valid_rmse = sum(valid_rmse) / len(valid_rmse)
                valid_score_list.append(valid_rmse[0])
                print(f"Validation RMSE = {valid_rmse}")
            #-->End predictions on validation set 


            # Predictions on Test set
            # --------
            predicted_list = np.array([])
            target_list = np.array([])
            for i, data in enumerate(tqdm(test_dataloader, desc="Testing",
                                          leave=False, ncols=80, unit='batch')):
                data = {key:value.to(device) for key, value in data.items()}
                position_num = torch.arange(256).repeat(data["smiles_bert_input"].size(0),1).to(device)
                output = model.forward(data["smiles_bert_input"], position_num, adj_mask=data["smiles_bert_adj_mask"], adj_mat=data["smiles_bert_adjmat"])
                output = output[:,0]

                if task == 'classification':
                    predicted = torch.sigmoid(output)
                    predicted_list = np.append(predicted_list, predicted.cpu().detach().numpy())
                    target_list = np.append(target_list, data["smiles_bert_label"].cpu().detach().numpy())
                else:
                    test_loss = torch.sqrt(test_crit(output, data["smiles_bert_label"]))
                    test_rmse.append(test_loss)
                    test_pred.append(output)
                    test_true.append(data["smiles_bert_label"])

            if task == 'classification':
                predicted_list = np.reshape(predicted_list, (-1))
                target_list = np.reshape(target_list, (-1))
                test_result_list.append((target_list, predicted_list))
                auc_score = roc_auc_score(target_list, predicted_list)
                test_score_list.append(auc_score)
                print(f"ROC_AUC Score on Test Set       = {auc_score:5.3f}")
            else:
                test_rmse = torch.cat(test_rmse, dim=0).cpu().numpy()
                test_pred = torch.cat(test_pred, dim=0).cpu().numpy()
                test_true = torch.cat(test_true, dim=0).cpu().numpy()
                test_result_list.append((test_true, test_pred))
                test_rmse = sum(test_rmse) / len(test_rmse)
                print(f"Testing RMSE    = {test_rmse}")
                test_score_list.append(test_rmse[0])
            #-->End predictions on test set

            # If validation loss is reduced, save the new model
            if valid_avg_loss < min_valid_loss:
                save_path = f"chembert/Finetuned_model_{epoch}.pt"
                print(f"Validation loss improved, saving new model to: {save_path}")
                torch.save(model.state_dict(), save_path)
                model.to(device)
                min_valid_loss = valid_avg_loss

        epoch_time   = time.time() - epoch_start
        elapsed_time = time.time() - start_time
        print(f"Epoch time:   {epoch_time//60} minutes.")
        print(f"Elapsed time: {elapsed_time//60} minutes out of a maximum of {max_time}.")
        if elapsed_time > max_time * 60:
            print(f"Time limit reached, stopping.")
            break

    print("#"*60)
    print(f"{'FINISHED FINE-TUNING CHEMBERT MODEL':^60s}")
    print("#"*60)
    print(f"TOTAL ELAPSED TIME: {elapsed_time//60} minutes.")
    
    # plot the graph & return best score
    # plt.figure(1)
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,4))
    x_len = np.arange(1,len(train_loss_list)+1) # epochs count from 1.
    ax[0].plot(x_len, train_loss_list, marker='.', c='blue', label="Train loss")
    ax[0].plot(x_len, valid_loss_list, marker='.', c='red' , label="Validation loss")
    ax[0].legend(loc='upper right')
    ax[0].grid()
    ax[0].set_xlabel('epoch')
    ax[0].set_ylabel('loss')
    ax[0].set_title('Loss graph')
    #ax[0].savefig('chembert/loss_graph.png')

    output_csv = pd.DataFrame()
    output_csv['Train_loss']  = train_loss_list
    output_csv['Valid_loss']  = valid_loss_list
    output_csv['Valid_score'] = valid_score_list
    output_csv.to_csv('chembert/result.csv', index_label='Epoch')

    # roc curve or r2
    # These next plots are saved only for the model with the best validation loss
    best_step = np.argmin(valid_loss_list)
    best_epoch = best_step +1
    best_score = test_score_list[best_step]

    print(f"The lowest validation loss is from Epoch {best_epoch}.")
    print(f"which is saved as 'chembert/Finetuned_model_{best_epoch}.pt'.")
    print(f"Score in Testing Set: {best_score}")

    output_json = params
    output_json['best_step']  = int(best_step)
    output_json['best_score'] = round(best_score,5)

    if task == 'classification':
        output_json['metric'] = 'AUC-ROC'
        fpr, tpr, _ = roc_curve(test_result_list[best_step][0], test_result_list[best_step][1])
        #plt.figure(2)
        lw = 2
        ax[1].plot(fpr, tpr, color='darkorange', lw=lw, label='ROC Curve (area = %0.2f)' % auc(fpr, tpr))
        ax[1].plot([0,1], [0,1], color='navy', lw=lw, linestyle='--')
        ax[1].set_xlim([0.0, 1.0])
        ax[1].set_ylim([0.0, 1.05])
        ax[1].set_xlabel('False Positive Rate')
        ax[1].set_ylabel('True Positive Rate')
        ax[1].set_title('Receiver operating characteristic')
        ax[1].legend(loc='lower right')

    else:
        output_json['metric'] = 'RMSE'
        plt.figure(3)
        t_true, t_pred = test_result_list[best_step][0], test_result_list[best_step][1]
        plt.scatter(t_true, t_pred, s=10, alpha=0.5)

        # Linear regression to get a fit line
        b,a = np.polyfit(t_true.flatten(), t_pred.flatten(), deg=1)
        x_fit = np.linspace(min(t_true), max(t_true))
        plt.plot(x_fit, (a+b*x_fit), 
                 label=f"R-squared Score = {r2_score(t_true, t_pred):0.2f}",
                 color='red', lw=2.5)

        # Adds an x=y line.
        xpoints = plt.xlim()
        ypoints = plt.ylim()

        ax[1].plot(xpoints,ypoints, 
                   color='grey',  linestyle='--', lw=0.5)

        ax[1].xlabel('Actual')
        ax[1].ylabel('Predicted')
        ax[1].title(f'Test set results for model from epoch {best_epoch}')
        ax[1].legend(loc='lower right')
        
    plt.suptitle(f"CHEMBERT Finetuning TEST set results for model from epoch {best_epoch}",
                 fontsize = 'xx-large', weight = 'extra bold')
    plt.tight_layout()
    
    plt.show()
    with open('chembert/dm.json', 'w') as f:
        json.dump(output_json, f)

    # Cleanup
    Path(f"chembert/Finetuned_model_{best_epoch}.pt").rename("chembert/Finetuned_model_final.pt")
    print("\nFINAL MODEL:\nThe best model was saved in file 'chembert/Finetuned_model_final.pt'.")
    for epoch in range(1,max_epochs+1):
        if Path(f"chembert/Finetuned_model_{epoch}.pt").is_file():
            Path(f"chembert/Finetuned_model_{epoch}.pt").unlink()
    
    return 


if __name__ == "__main__":
    import argparse

    #-- Command line arguments
    parser = argparse.ArgumentParser(description='''Finetunes a CHEMBERT model''')

    parser.add_argument('smiles_file',
                        help='Path to the input SMILES file. Must have 2 columns: SMILES, data')

    parser.add_argument('-m', '--model',
                        help='Pre-trained model',
                        default=f'{module_dir}/properties/activity/CHEMBERT/model/pretrained_model.pt')

    parser.add_argument('-t', '--max_time', type=int,
                        help='Maximum time to train (minutes)',
                        default=720)

    args = parser.parse_args()
    
    smiles_file = args.smiles_file
    pretrained_model = args.model
    max_time = args.max_time
    #----------------------
    Path('chembert').mkdir(exist_ok=True)
    finetune_model(smiles_file, max_time=max_time)
