# BetaBarrelLSTM
Generation of Beta Barrel Protein Sequences with an LSTM recurrent neural network. 

This script uses Deep Learning Neural Networks (LSTM) with Keras on TensorFlow, to generate new beta barrel protein sequences. This script generates FASTA sequences.

Example of a generated beta barrel FASTA sequence:
```
GVEFDEETVDGRKVKSIITLDGGALVQVQKWDGKSTTIKRKRDGDKLVVECVMKGVTSTRVYERACPTLGGVGNQTTVDNGPDNSGGGDNVNGVAVGFVVVVPGGGTVGSTVGGGVISGVGGVTVDVTTIRVNIVVGRSVGTVVVDTTTVGTTIDSGDTNTVDGDDGTVTKAGGVRVDVVNFVGVGEGVNVPSLLVDKNVVIVRGTDGVNPGVRSYDG
```

# How To Install and Use:
Clone the directory:
```
git clone https://github.com/JesslynJanssen/BetaBarrelLSTM
```
Then, install all requirements:
```
sudo apt update && sudo apt full-upgrade && sudo apt install python3-pip python3-pandas python3-numpy python3-tensorflow tensorboard && pip3 install keras
```
Finally, train the model or generate new beta barrel protein sequences. 

# Training the Model
Beta barrel FASTA sequence data from the Protein Data Bank is already provided, but you can train the model with other beta barrel proteins or any other type of protein. 

To train the network yourself, run the following command:
```
python3 train betabarrel
```

# Generate Beta Barrel Sequences 
To generate new beta barrel FASTA sequences, run the following command:
```
python3 generate beta barrel 
```
