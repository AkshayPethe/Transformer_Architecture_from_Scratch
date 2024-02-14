import torch 
import torch.nn as nn

class Self_Attention(nn.Module):

    def __init__(self,embed_size,heads):

        super(Self_Attention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim  = embed_size // heads

        assert(self.head_dim*heads == embed_size)

        self.values = nn.Linear(self.head_dim,self.head_dim,bias = False)
        self.keys  = nn.Linear(self.head_dim,self.head_dim,bias = False)
        self.queries = nn.Linear(self.head_dim,self.head_dim,bias = False)
        self.fc_out = nn.Linear(self.head_dim*heads,embed_size,bias = False)
        
        
    
    def fordward(self,values,keys,queries,mask):

        """"
        In PyTorch or TensorFlow, tensors are usually represented in the format 
        (batch_size, sequence_length, embedding_size).
        queries.shape[0] returns the size of the first dimension, 
                which corresponds to the batch size.
        By assigning this value to N, you effectively obtain the batch size.

        Suppose we have a batch size of 2, with embedding sizes of 4, and sequence 
        lengths of 3 for values, keys, and queries. Additionally, let's assume 
        we have 2 attention heads.

        values shape: (2, 3, 4) (batch size 2, sequence length 3, embedding size 4)
        keys shape: (2, 3, 4) (batch size 2, sequence length 3, embedding size 4)
        queries shape: (2, 3, 4) (batch size 2, sequence length 3, embedding size 4)
        After the reshaping operation:

        values shape: (2, 3, 2, 2) (batch size 2, sequence length 3, 2 attention heads, embedding size 2)
        keys shape: (2, 3, 2, 2) (batch size 2, sequence length 3, 2 attention heads, embedding size 2)
        queries shape: (2, 3, 2, 2) (batch size 2, sequence length 3, 2 attention heads, embedding size 2)"""

        N = query.shape[0]

        value_len,key_len,query_len = values.shape[1],keys.shape[1],query.shape[1]

        #Split Embeding into self.heads pieces

        values = values.reshape(N, value_len, self.head_dim, self.head_dim)
        keys = keys.reshape(N, key_len, self.head_dim, self.head_dim)
        queries = queries.reshape(N, query_len, self.head_dim, self.head_dim)

        energy = torch.matmul(queries,values)

        energy /= torch.sqrt(torch.tensor(key.size(-1),dtype = torch.float32))

         #Or energy = torch.einsum("nqhd", "nkdh->nhqk",[queries,keys]) will do the same thing
        
        """torch.einsum performs contraction operations on tensors according to Einstein summation convention.

        "nqhd" and "nkdh->nhqk" are the subscript labels defining the Einstein summation notation.

        "nqhd" represents the dimensions of the input tensors.

        "nkdh->nhqk" represents the dimensions of the output tensor.

        The notation uses letters to represent dimensions:

        n: Batch size dimension
        q: Query length dimension
        h: Number of attention heads dimension
        d: Dimension of the attention values/keys/queries"""

        if mask is not None:
            energy = energy.masked_fill_(mask == 0, float('-inf'))  #Setting Mask =0 to -inf means pay very less attention to it

        attention = torch.softmax(energy/ (self.embed_size**(1/2)),dim = -1) #Dim represent the softmax fn applied to which dim

        #Instead torch.matmul and normalizing it we will use einsum
        output = torch.einsum("nhql","nlhd-->nqhd",[attention,values]).reshape(
            N,query_len,self.heads,self.head_dim)  #As concatinating to original dimension

        #attention shape (N,heads,query_len,key_len) i.e ("nhqk")
        #Values shape (N,value_len,heads,embed_size)
        #Output shape : (N,query_len,head,head_dim)

        output = fc_out(output)

        return output
    


class TransformerBloc(nn.Module):
    def __init__(self,embed_size,heads,dropout,forward_expansion):

        super(TransformerBloc,self).__init__()

        self.attention = Self_Attention(embed_size,heads)
        self.norm1 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size,forward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size,embed_size)

        )
        self.norm2 = nn.LayerNorm(embed_size)

        self.dropout = nn.Dropout(dropout)

    
    def forward(self,value,key,query,mask):

        attention = self.attention(value,key,query,mask)  #Attention Layer

        x = self.dropout(self.norm1(attention + query)) #Add and Normalize Layer

        forward = self.feed_forward(x)

        out = self.dropout(self.norm2(forward + x))  #Add and Normalize the output from previous Layer
        
        return out
    

class Encoder(nn.Module):

    def __init__(self,
                 src_vocab_size,
                 embed_size,
                 num_layers,
                 heads,
                 device,
                 forward_expansion,
                 dropout,
                 max_length):
        
        super(Encoder,self)__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embeding(src_vocab_size,embed_size)
        self.positional_embedding = nn.Embedding(max_length,embed_size) 


        self.layers = nn.ModuleList(
            [
            TransformerBloc(
                embed_size,
                heads,
                dropout = dropout,
                forward_expansion= forward_expansion 
            )
        ]

        )

        self.dropout = nn.Dropout(dropout)


    def forward(self,x,mask):

        N,seq_len = x.shape
        positions = torch.arange(0,seq_len).expand(N,seq_len).to(self.device)

        out = self.dropout(self.word_embedding(x) + self.positional_embedding(positions))

        for layer in self.layers:

            out = layer(out ,out ,out ,mask)  #As Query,Keys,Values has same dim and matrix at the start in traditional transformer
            

        return out








       

     


 
       
   

