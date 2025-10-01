**Tensor** is a multidimensional array efficent for linear algebra 
**Cuda** determents if GPU or CPU should be used (GPU is better for fast easy calc, CPU is slower but more power full)
**Dot product** is a sum of multipling tensors and adding the sums together 
`[1,2,3]` • `[4,5,6]` = `1*4 + 2*5 + 3*6 = 32`
To know if a number is a float or an intiger use .dtype
Floats and INtiges cant be multiplied together if you want to multiply these cast the .float function on the intigers 
block_size = how many intigers can the model see at a time 
batch_size = how many blocks can the model see at once

weights - they decide how importand the input is so if we are talking about weather the tempreature will weight a lot
biases -  it is like an adjustment added after the weights, it helps the model fit something 
better by moving the predictions up and down
example: in a recipe the weights are like the ingredients and bias is like adding more salt or sugar so it tastes just righht for you

GPT = Generative Pre-trained Transformer
forward pass = putting the input throught all the stept pass logits
## **PyTorch commands** 
**Radint**
used to generate random intigers in this example between -100 and 100
 (the comma in the size parameter is explicetly telling python that it is a tuple for a 1D array)
 ![[Pasted image 20250807010207.png]]
**Tensor** 
creates a tensor from data similar to a list or array, used for inputs, outputs and model weights
![[Pasted image 20250807010234.png]]
**Zeros**
creates a tensor field filled with zeros, used for padding sequences 
![[Pasted image 20250807010422.png]]
**Ones**
creates a tensor field filled with ones, used for padding and sometimes mask initialization
![[Pasted image 20250807010547.png]]
**Empty**
creates a tensor field and fills it with random massive and small numbers, used in performance-critical code
![[Pasted image 20250807010800.png]]
**Arange**
creates a 1D tensor with aranged input from 0 to N, useful for position indexing, batching, positional encodings
![[Pasted image 20250807011050.png]]
**Linspace**
lineary aranges steps from set values, occasionally used in plotting, interpolation, or custom embeddings
![[Pasted image 20250807011243.png]]
**Eye**
creates a diagonal line of ones in a set size matrix
![[Pasted image 20250807011446.png]]
**Cat**
combines two tensors with the second one must being only of the size one, used for shaping, moving data
![[Pasted image 20250807011816.png]]
**Tril**
creates a triangle of ones on the lower half and of zeros on the top as the l in the function states, used for causal attention masks in autoregressive LLMs (so a token can’t "see" future tokens)
![[Pasted image 20250807012049.png]]
**Triu**
creates a triangle of ones in the top half and of zeros on the bottom as the u in the function states, sometimes used in mask manipulation or reversing causal masks
![[Pasted image 20250807012227.png]]
**Transpose**
flips two positions inside of a tensor, very common in LLMs: e.g., converting `[batch, seq, hidden]` to `[seq, batch, hidden]`
![[Pasted image 20250807012406.png]]
**Stack**
stacks multiple tensors on top of each other, used in batching or processing sequences
![[Pasted image 20250807012455.png]]

### while making a class like: class BiagramLanguageModel
### and we use subclass like (nn.Module) so it looks like this:
### class BiagramLanguageModel(nn.Module): it will make all the nn. fuctions learnable 

## gradiant desent 
lest say we have a non trained language model with 80 characters than the chance of succesfuly predicting the next character is 1/80 not even 2% 
we can claculate the loss (the bigger the loss the worse for our model to be better)
with the formula -ln(1/80) - the log of 1/80 and negative we get 4,38 that is a terrbile loss
reducing this number is called the gradiant desent 
it is the optimizer for the network in other words 

## sum normalizing
example: [2, 4, 6] = 2+4+6 =12
2/12 = 0.167
4/12 = 0.33
6/12 = 0.5
[2, 4, 6] normalized = [0.167, 0.33, 0.5]
## logits
probability distribution of what we want to predict 
lets say we have the example from normalizing:
[2, 4, 6] normalized = [0.167, 0.33, 0.5]
lets say that these are   ab     ac     ad
from this we see that ad is most likely to come next 

### B, T, C
B = batch_size - line of intigers
T = time (time dimention) - we have a batch but we dont know what is next token is so it is a time because there is something we dont know yet and something we already do know like the time in the world dimention we probably live in there is the past we know and the future we ar eyet to know 
C = channels - the vocab_size 
Mrproper321

### .view
lets say we have a random tensor of the dimentions (2, 3, 5)
and we unpack the dimentions with .shape to x, y, z
and than we use .view to pack them back together into a tensot 

### N * C
what shape does torch expect when dealing with logits?
it expects (N, C) but we have **B, T, C** so we reshape it with this line:
logits = logits.view(B * T, C) so we multiply the B and T into the N we need clever 

---

len - how many items are in a list

set - gets rid of duplicates in a list

.shape - tells the dimensions of a tensor

.view - changes the dimensions of a tensor without changing the data 

.cross_entropy - mesures how wrong the model's prediction is 

.Embedding - without embedding words have just meaning less ids, embedding creates a 
table with each word and its attributes so the model can tell that some words are similar or have similar meaning (example: cat and dog they re similar so after one of those will come similar words)

enumerate - indexes each item in a list with their own id 

int - changes floats into integers by removing the decimal part 

nn.linear - it reshapes and mixes information  so models can learn useful patterns 

.mean - computes the average just like grades in school 
x = torch.tensor([1.0, 2.0, 3.0, 4.0])
print(x.mean())  # Output: 2.5
### AdamW 
popular optimizer, known for ability to improve deep learning models, especially when generation is critical

#### Modular operator (modulo) %
is a remainder after division 24/7 = 3 | 24%7 = 4

#### Decorator @
simple way to modify existing functions

def my_decorator(func):
    def wrapper():
        print("Before calling the function")
        func()
        print("After calling the function")
    return wrapper

@my_decorator
def say_hello():
    print("Hello!")

say_hello()

**Output:**
Before calling the function
Hello!
After calling the function

### Different forms of optimizing in ML
| Optimization Type       | Key Traits                      | Best Use Case                                |
| ----------------------- | ------------------------------- | -------------------------------------------- |
| Batch GD                | Accurate but slow               | Small datasets                               |
| SGD                     | Fast updates, noisy             | Large datasets, online learning              |
| Mini-batch GD           | Balanced                        | Most deep learning tasks                     |
| Adagrad                 | Adaptive for sparse features    | Sparse data                                  |
| RMSProp                 | Adaptive with decay             | RNNs, non-stationary data                    |
| Adam                    | Adaptive + momentum             | Default for most deep learning               |
| AdamW                   | Adam + proper weight decay      | Deep nets requiring regularization           |
| Momentum                | Accelerates SGD                 | Slow convergence or oscillations             |
| Nesterov                | Lookahead momentum              | Momentum situations with improved speed      |
| Newton                  | Uses Hessian                    | Small/medium problems with available Hessian |
| Quasi-Newton (L-BFGS)   | Approximate Hessian             | Medium problems, classical ML                |
| Evolutionary Algorithms | Gradient-free                   | Black-box or non-differentiable problems     |
| Bayesian Optimization   | Surrogate model for hyperparams | Hyperparameter tuning                        |
# Activation functions 
### ReLU
rectified linear unit what it does is that it takes a number that is 0 or lower and turns it into 0 and if the number is higher than 0 it keeps them the way they are 
![[Pasted image 20250811031724.png]]

### SIGMOID
e = 2.71
range - bteween 0 and 1
![[Pasted image 20250811032658.png]]

### Tanh
e = 2.71
range - between -1 and 1 
![[Pasted image 20250811033138.png]]

#### Diffrence between sigmoid and tanh
the difference is just in range 
![[Pasted image 20250811033232.png]]


# Transformer model architecture
### https://excalidraw.com/#json=2fbDMKEYeNfk5aY7YTFUm,VR4oKzQHMibfmgdNXpY7OQ

![[Pasted image 20250818025619.png]]
## Encoders
n_layers = how many encoders and decodres we have
each encoder is build out of **Multi-head Attention -> Residual connection (add) then normalize,-> Feed forward (Linear->ReLU->Linear) -> Resridual Connection (add )then normalize**
each encoder feeds the input into the next and the last feeds it into each decoder
- **Multi-head attention** → every word looks at all the other words.
- **Add & Normalize** → add the original input back, then normalize.
- **Feed-forward network** → a small neural net (linear → ReLU → linear).
- **Add & Normalize** → add the input back again, then normalize.

### Multi-head Attention
this is what makes the transformer model great, it is like if you give 10 different people the same book and all of the give you a different idea about the book, every one will see the enviroment a little different. that is what the heads do they all look at the same word but they all have a different perspectiove so the model learns faster and more efficent 

it is build out of **Key (K), Query (Q), Value (V) -> Scaled Dot-product Attention -> Concatenate resoults -> linear**
Scaled Dot-product Attention runs n  heads in parallel

Query is like a question a word is asking
Key is the index that helps answer the question (what words are important in that question)
Value is the info carried by the word 

**Scaled Dot-Product Attention**
For each word, compare its **Query** with all other **Keys** → this tells us how much attention it should pay to each word.
Use those scores to mix the **Values** into a new representation.
"Scaled" just means we divide by the vector size so the numbers don’t get too big.

**Parallel Heads (n heads)**
Instead of doing this once, the model does it **several times in parallel** (multiple “heads”).
Each head focuses on different kinds of relationships (grammar, meaning, position, etc.).

Concatenate Results - stuck all the heads output together side-by-side0

### Self Attention
it learns what words in a sentence are important, for example:

"Server, can i have the check?"
"Looks like i crashed the server!"

both have the word server in them but with a different meaning, server as the person and server as the device, so how can the model know witch is the one?
with self attention in the second sentence it will give a big attentyion score to the word crashed because you wount normaly crash into the human server

##### Mass Attention
is just saying we dont want to look in to the future cause thet would be chating we only want to guess with what we already know in our current timestamp and everything before it just like in life. so it uses torch.tril

### Pos Encoding
it calls this function:
![[Pasted image 20250818050735.png]]
lets say the unput is "hello world"
it does the first function with the sin for every 2i so even number and the cos for every odd number 
so in "hello world" 
h has the 0th pos so it is even and e has the 1st pos so it is odd 

## Standard deviation
![[Pasted image 20250819031210.png]]
N = the number of elements in the array
Xi = each of the elements
__
X = the average of the array

![[Pasted image 20250819034426.png]]

### Memory mapping
it is a way to oprn and look at pieces of disc files without opening the whole thing at once 

# Research and Development 
- efficency testing with time mapping
- RNNs
- quantazation of LLMs
- gardient acmumulation
