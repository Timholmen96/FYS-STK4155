# Changes 
## Neuralnetregressor
* Method input 
* "Output" input slik at aktiveringsfunksjonen kan varieres utifra dette
## Generelt
* Jeg må løse for Plain Gradient descent i tilleg til SGD

## Sjekk week44/43 classifier code for Epoch and batches


## diskusjon
- Cost() i SGD bruker lang tid, mye data kraft hvis den beregner for hver epoch; hvor mye?
  - Løsning, beregn halvveis og på slutten
  
- Jax.Grad blir testet men ikke implementert, jeg vet dette ville spart mye datakraft, men ikke hvor mye det ville spart,
* Jeg fikk ikke implementert dette da koden baserer seg på vanlig backpropagation, som muligens krever mer krefter og tid. 


## Tidtakning
Classification: 107m11.2s
Hidden: 5
Gamma: 4
Penalty none: 1
Penalty: 6


# Skriv over Tekst fra R & C til python objekter ( )


# Ubrukt tekst:
\subsection{NeuralNetwork}
The Neural network creates the architecture, predicts by feed forward, 
and calculates the cost function based on the input values and the target values.  
It is a class which is initiated with input size, output sizes,
activation functions and its derivatives, the cost function and
its derivative, and the seed which ensures consistency when
creating the random initial values (rng) of the weights. The chosen
seed is 2026 for the regression problem.  
The layer architecture is created inside the class and saved as an attribute.
For each layer, the class creates initial weights and biases, then uses the
current layer's output size as the next layer's input size:
\begin{align}
    W^{(l)} &\sim
        \frac{\operatorname{standard\_normal}
        (n_{\mathrm{in}}^{(l)}, n_{\mathrm{out}}^{(l)})}
        {\sqrt{n_{\mathrm{in}}^{(l)}}}, \\
    b^{(l)} &= 0.01, \\
    n_{\mathrm{in}}^{(l+1)} &= n_{\mathrm{out}}^{(l)}.
\end{align}
\subsubsection{Feed-forward pass}
This first part of the network is the feed forward,
it sends he input values through the layers, each layers 
activation function is applied and the output layer is
calculated by the chosen activaiton function, which is the
identity for the regression problem. 
% LLM Declaration: Edit skeleton
The feed-forward procedure is as follows:
\begin{align}
    \mathbf{a}^{(0)} &= \mathbf{x}, \\
    \mathbf{z}^{(l)} &= \mathbf{a}^{(l-1)} W^{(l)} + \mathbf{b}^{(l)}, \\
    \mathbf{a}^{(l)} &= \phi^{(l)}\!\left(\mathbf{z}^{(l)}\right),
    \qquad l = 1, \ldots, L, \\
    \hat{\mathbf{y}} &= \mathbf{a}^{(L)}.
\end{align}
For backpropagation, it is important to save the input values,
the z values and the output values a during the feed forward.

# OLS
The linear regression model with the lowest $CV-MSE$ had degree 11, with a score $CV-MSE = 0.0211 +/- 0.00324$, $SE = 0.00324$ which gives
a One SE threshod $0.02345$, the models with a $CV-MSE$ below the threshold are listed below.  
\begin{table}[htbp]
    \centering
    \caption{Linear Regression CV Results}
    \label{tab:OLS}
    \begin{tabular}{lccccc}
        \multicolumn{3}{c}{One SE Scores}\\
        \hline
        Degree & CV MSE   & SE       \\
        8      & 0.02253  & 0.00172  \\
        9      & 0.02376  & 0.00121  \\
        10     & 0.02400  & 0.00285  \\
        \hline
    \end{tabular}
\end{table}[htbp]

# CV2
This can be observed in the table below, which is from the 
model with lowest CV MSE; three hidden layer $(50,100,100)$.
\begin{table}[htbp]
    \centering
    \caption{Vanishing Gradients}
    \label{tab:VanishGrad}
    \begin{tabular}{lc}
        \hline
        Layer & L2 Norm$(dW, db)$ \\
        \hline
        1 & $0.00976$ \\
        2 & $0.01524$ \\
        3 & $0.01065$ \\
        4 & $0.03032$ \\
    \end{tabular}
\end{table}
\FloatBarrier
We see signs of vanishing gradients, as the L2 norm of the gradients in
the first layer, which is furtherst away from the output,
is smaller by a magnitude of around 3, than the fourth layer, which is the output layer,
the cross entropy, although none of these are close to zero, a trend can be observed 
which corresponds to the idea of vanishing gradients.

# CV2 Discussion
There is the problem of vanishing
gradients with sigmoid, as the network deepens, with more layers,
the gradient in the layers far away from the output can apporach zero,
and the weights stop updating, halting the learning process in those layers. 