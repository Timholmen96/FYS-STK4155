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