# Struktur for metodedelen i prosjekt 1

Dette dokumentet er en skriveplan for `Prosjekt1.tex`, basert på oppgaveteksten i `Project1.pdf`, skriveveiledningen i `projectwriting.ipynb`, vurderingskriteriene i det vedlagte bildet og implementasjonen i `../Prosjekt-Kode/Full_Project.ipynb`.

Oppgaveteksten og skriveveiledningen beskriver krav og råd for rapporten. Disposisjonen nedenfor er et forslag til hvordan materialet kan organiseres; overskriftene er ikke et krav fra oppgaven.

## Hovedidé

Organiser metodedelen tematisk, med en egen seksjon etterpå for implementasjon og validering. Forklar først hvordan dataene og modellene bygges, deretter hvordan modellene vurderes og optimaliseres.

Metodedelen skal gjøre leseren i stand til å forstå:

- Hva som beregnes.
- Hvorfor metodene passer til problemet.
- Hvordan forsøkene kan gjentas.

Introduksjonen bør kort presentere problemstillingen, motivasjonen og hva som undersøkes. Flytt de tomme metodeoverskriftene som nå ligger i introduksjonen til metodedelen. Matematiske utledninger og algoritmebeskrivelser hører hjemme der.

## Foreslått disposisjon for `Methods`

| Underseksjon | Innhold | Tilknytning til oppgaven |
|---|---|---|
| **1. Data and preprocessing** | Runges funksjon, generering av inngangsdata og støy, polynommodell, designmatrise, trenings-/testdeling og standardisering. | Felles grunnlag |
| **2. Regression models** | OLS, Ridge og Lasso: kostfunksjoner, løsninger, regularisering og metodens egnethet. | a, b, g |
| **3. Error measures and resampling** | MSE, R², bias–varians-dekomponering, bootstrap og k-fold-kryssvalidering. | a, c, d, i |
| **4. Gradient-based optimization** | GD, analytiske gradienter, automatisk derivasjon, Hessian, læringsrate, momentum, AdaGrad, RMSprop, Adam og Lasso-subgradienter. | e, f, g |
| **5. Mini-batch stochastic gradient descent** | Mini-batcher, stokking, epoker, læringsrateplan og sammenligning med GD. | h |
| **6. Experimental setup** | Konkrete forsøksinnstillinger, initialisering og stoppkriterier, gjerne samlet i en tabell. | Alle delene |

### 1. Data and preprocessing

Presenter datamodellen:

$$
f(x)=\frac{1}{1+25x^2},\qquad
y_i=f(x_i)+\varepsilon_i,\qquad
\varepsilon_i\sim\mathcal{N}(0,\sigma^2).
$$

Forklar at den kjente funksjonen gir en referanse for å undersøke regresjonsmetodene. I den nåværende koden trekkes inngangsverdiene fra en uniform fordeling på intervallet $[-1,1]$.

Definer polynommodellen og designmatrisen:

$$
\widetilde y_i=\sum_{j=0}^{d}\theta_jx_i^j,
\qquad X_{ij}=x_i^j.
$$

Forklar at modellen er lineær i koeffisientene selv om den inneholder potenser av $x$.

Beskriv deretter:

- Delingen i 70 % treningsdata og 30 % testdata.
- Standardiseringen av de ikke-konstante polynomkolonnene.
- At gjennomsnitt og standardavvik beregnes fra treningsdata og brukes på testdata.
- At konstantkolonnen beholdes som 1, og at $y$ ikke standardiseres.
- At skalering i kryssvalideringen læres separat fra treningsfoldene.
- At bias–varians- og bootstrap-cellene bruker uskalerte designmatriser, slik at rapporten må beskrive denne forskjellen.

### 2. Regression models

Bruk egne underavsnitt for OLS, Ridge og Lasso. Presenter kostfunksjonen, hvordan koeffisientene finnes, og hvorfor metoden er relevant for problemet.

Med $n_{\mathrm{tr}}$ treningsobservasjoner følger følgende kostfunksjoner normaliseringen i koden:

$$
C_{\mathrm{OLS}}(\theta)
=\frac{1}{n_{\mathrm{tr}}}\|y-X\theta\|_2^2,
$$

$$
C_{\mathrm{Ridge}}(\theta)
=C_{\mathrm{OLS}}(\theta)
+\frac{\lambda_{\mathrm{R}}}{n_{\mathrm{tr}}}
\sum_{j=1}^{d}\theta_j^2,
$$

$$
C_{\mathrm{Lasso}}(\theta)
=C_{\mathrm{OLS}}(\theta)
+\lambda_{\mathrm{L}}\sum_{j=1}^{d}|\theta_j|.
$$

Konstantleddet $\theta_0$ regulariseres ikke. Bruk gjerne ulike symboler for Ridge- og Lasso-straffen fordi de har forskjellig normalisering i implementasjonen.

For OLS og Ridge bør de lukkede løsningene presenteres, inkludert bruk av pseudoinvers og en straffematrise som unntar konstantleddet. For Ridge bør virkningen av regularisering også knyttes til krymping av singularverdiretningene, slik oppgaven ber om.

For Lasso bør du forklare $L_1$-straffen og muligheten for nullkoeffisienter. Absoluttverdien er ikke deriverbar ved null, og din egen implementasjon bruker subgradienter.

Forklar forholdet til Scikit-learns parameterkonvensjoner når referansemodellene beskrives:

- Ridge bruker `alpha = lambda_R` for den aktuelle normaliseringen.
- Lasso-referansen for din egen kostfunksjon bruker `alpha = lambda_L / 2`.
- I den avsluttende Lasso-kryssvalideringen angis Scikit-learns `alpha` direkte.

### 3. Error measures and resampling

Definer MSE og $R^2$, og forklar forskjellen mellom treningsfeil, testfeil og feil i valideringsfoldene.

Utled bias–varians-dekomponeringen:

$$
\mathbb{E}\big[(y-\widetilde y)^2\big]
=\operatorname{Bias}^2[\widetilde y]
+\operatorname{Var}[\widetilde y]+\sigma^2.
$$

Oppgaveteksten ber uttrykkelig om denne utledningen i teoridelen. Forklar hvilke tilfeldigheter forventningsverdiene tas over, og hva de tre bidragene betyr.

Beskriv bootstrap som gjentatte trekk med tilbakelegging fra treningssettet, med evaluering på et fast testsett. Forklar hvordan variasjonen mellom prediksjonene brukes til å estimere varians og et biasrelatert bidrag.

**Presisering om bootstrap-koden:** Gjennomsnittsprediksjonen sammenlignes med støyfulle `y_test`, ikke med den støyfrie funksjonen $f(x)$. Det målte «bias²»-bidraget påvirkes derfor også av teststøyen og absorberer støyvariansen i forventning. For de empiriske størrelsene i denne implementasjonen gjelder:

$$
\texttt{error}=\texttt{bias}+\texttt{variance}.
$$

Ikke legg til $\sigma^2$ én gang til i denne beregnede identiteten. Skill mellom den teoretiske dekomponeringen og det empiriske estimatet.

Beskriv deretter k-fold-kryssvalidering, med $k=5$ og $k=10$, og hvordan gjennomsnittlig validerings-MSE brukes til å sammenligne polynomgrader og regulariseringsverdier. Forklar hvorfor skalering må utføres innenfor hver fold for å unngå at valideringsdata påvirker treningen.

Den avsluttende kryssvalideringen bruker hele datasettet. Beskriv dermed resultatene som kryssvaliderte estimater brukt til modellvalg; de er ikke en separat evaluering på et urørt slutt-testsett.

### 4. Gradient-based optimization

Presenter først vanlig gradient descent:

$$
\theta_{k+1}=\theta_k-\eta\nabla_\theta C(\theta_k).
$$

Utled gradientene for OLS og Ridge med samme normalisering som kostfunksjonene. Forklar hvordan JAX brukes til automatisk derivasjon, og hvorfor dette skiller seg fra symbolsk derivasjon og endelige differanser. Oppgaven ber også om en kort diskusjon av beregningskostnaden.

Forklar sammenhengen mellom Hessianens egenverdier, kondisjonstall og konvergens. For vanlig GD på de positivt definite kvadratiske kostfunksjonene gjelder stabilitetsbetingelsen:

$$
0<\eta<\frac{2}{\lambda_{\max}(H)}.
$$

Denne grensen gjelder vanlig GD; den gir ikke automatisk optimale læringsrater for momentum eller Adam.

Bruk korte underavsnitt for momentum, AdaGrad, RMSprop og Adam. Presenter oppdateringsreglene og forklar hva hver metode forsøker å forbedre. Beskriv også initialisering av hjelpevariablene og parameterverdiene som brukes.

Avslutt med Lasso-subgradienten. Ved en nullkoeffisient velger koden `sign(0) = 0`, som er et gyldig valg fra subgradientintervallet $[-1,1]$ for absoluttverdien. Beskriv også kontrollen av hva JAX returnerer ved null. Manglende lukket løsning betyr ikke i seg selv at vanlig GD er den eneste mulige løsningsmetoden.

### 5. Mini-batch stochastic gradient descent

Forklar hvordan treningsdata stokkes og deles i mini-batcher, og skill mellom en parameteroppdatering og en epoke.

Beskriv faste læringsrater og den avtakende læringsrateplanen. Forklar hvordan regulariseringsleddet normaliseres slik at mini-batch-gradientene svarer til samme kostfunksjon som full GD.

Definer sammenligningsgrunnlaget mellom GD og SGD:

- Sluttavvik fra referansekoeffisientene.
- Antall behandlede treningspunkter.
- Eventuell kjøretid, dersom den måles.

Antall oppdateringer alene er ikke tilstrekkelig for å sammenligne beregningsarbeidet når oppdateringene bruker ulike mengder data.

### 6. Experimental setup

Samle innstillingene i en tabell slik at forsøkene kan gjentas. Oppgi verdiene per eksperiment, ettersom de varierer mellom delene av notebooken.

| Innstilling | Nåværende implementasjon / hva som bør oppgis |
|---|---|
| Antall datapunkter | $n=350$ |
| Støy | $\sigma=0.25$ |
| Tilfeldighetsfrø | Vanligvis 2026; gradientkontrollen bruker også 2027 |
| Trenings-/testdeling | 70/30 i holdout-forsøkene |
| Polynomgrader | 0–15 i hovedforsøkene for OLS/Ridge og bootstrap; 0–25 i trenings-/testfeilfiguren; 0–14 i hovedtabellene for kryssvalidering |
| Bootstrap | 40, 100 og 300 trekk |
| Kryssvalidering | 5 og 10 folder, med stokking og fast frø |
| Konvergensforsøk | Grad 5; Ridge-straff 0.01 |
| Regularisering | Oppgi separate søkegitter for hvert eksperiment og parameterkonvensjonen for Lasso |
| Initialisering | Felles tilfeldig start for GD-sammenligningene; nullvektor for GD/SGD-sammenligningen i del h |
| Stoppkriterier | Gradientnorm i del e; parameter-MSE mot referansen i del f/g; oppgi også iterasjonsgrensene |
| SGD | 100 epoker, batchstørrelse 100 og OLS-kostfunksjon i den nåværende kjøringen |
| Tidtaking | I del f: tre repetisjoner, median kjøretid og JAX-kompilering før tidtaking |

Skill mellom det som allerede er undersøkt og det som gjenstår. Oppgaven ber også om å undersøke variasjon i antall datapunkter og støynivå, samt batchstørrelse og epokeantall for SGD. Disse størrelsene er foreløpig faste i de aktuelle forsøkene.

## Egen seksjon: `Implementation and validation`

Skriveveiledningen fremhever at rapporten må forklare implementasjonen og vise hvordan koden er kontrollert. Vurderingsbildet gir 10 poeng til metoder og 10 poeng til kode, implementasjon og testing.

Beskriv kort kodestrukturen: datagenerering, designmatrise, regresjonsløsere, resampling, gradientfunksjoner og oppdateringsregler. Bruk gjerne pseudokode for en sentral algoritme og henvis til notebooken for full implementasjon.

Notebooken inneholder følgende valideringer som kan beskrives og dokumenteres:

- Egen OLS- og Ridge-kryssvalidering sammenlignes med Scikit-learn på samme folder.
- Analytiske gradienter sammenlignes med JAX-gradienter.
- GD-resultater sammenlignes med lukkede OLS- og Ridge-løsninger.
- Lasso sammenlignes med en Scikit-learn-referanse med tilsvarende kostfunksjon.

Forklar hvilke avvik som måles, hvilke toleranser som brukes, og hvilke begrensninger kontrollene har. Beskriv referansebaserte stoppkriterier som en del av benchmarkforsøkene; de forutsetter at referanseløsningen er tilgjengelig.

Skill mellom parameterfeil, kostnadsgap og test-MSE. De måler henholdsvis avvik fra referansekoeffisientene, avvik i treningsmålet og prediksjonsfeil.

Ikke skriv at en kontroll har bestått før de aktuelle kjøringsresultatene er verifisert.

## Fordeling mellom metode og resultater

Metodedelen forklarer hva som gjøres, hvordan det gjøres og hvorfor. Resultatdelen viser hva forsøkene faktisk ga og diskuterer betydningen.

Eksempler:

- **Metode:** Hvordan bootstrap brukes til å estimere variasjon mellom prediksjoner.
- **Resultat:** Hvordan de estimerte bidragene endrer seg med polynomgraden.
- **Metode:** Hvorfor læringsrater under og over GD-grensen undersøkes.
- **Resultat:** Hvilke kjøringer konvergerer, og hvor mange steg de trenger.
- **Metode:** Hvordan kryssvalidering brukes til å velge grad og regulariseringsstyrke.
- **Resultat:** Hvilken modell som får lavest kryssvalidert MSE blant de undersøkte alternativene.

Vurderingsbildet gir 40 poeng til resultatanalysen. Hold derfor metodebeskrivelsen konsentrert, og gi plass til tolkning og kritisk diskusjon av et utvalg tydelige figurer og tabeller.

## Presiseringer i det eksisterende rapportutkastet

- Inngangsdataene er uniformt fordelte; det er støyen i responsen som er normalfordelt.
- Runges funksjon er en rasjonal funksjon. Det er tilnærmingsmodellen som er et polynom.
- En lineær sammenheng beskrives av et førstegradspolynom, ikke et andregradspolynom.
- Lasso har en analytisk gradient for det kvadratiske dataleddet, men $L_1$-straffen er ikke deriverbar ved null. Bruk begrepet subgradient for den aktuelle implementasjonen.
- Beskriv skaleringen slik den faktisk utføres i hver del av koden.
- Skill mellom modellparametre, som koeffisientene, og hyperparametre, som grad, regulariseringsstyrke og læringsrate. GD oppdaterer koeffisientene; hyperparametrene sammenlignes i separate forsøk eller ved kryssvalidering.

## LaTeX-skjelett

```latex
\section{Methods}

\subsection{Data and preprocessing}

\subsection{Regression models}
\subsubsection{Ordinary least squares}
\subsubsection{Ridge regression}
\subsubsection{Lasso regression}

\subsection{Error measures and resampling}
\subsubsection{Performance measures}
\subsubsection{Bias--variance decomposition}
\subsubsection{Bootstrap}
\subsubsection{Cross-validation}

\subsection{Gradient-based optimization}
\subsubsection{Gradient descent and automatic differentiation}
\subsubsection{Learning rate and conditioning}
\subsubsection{Momentum and adaptive methods}
\subsubsection{Subgradient optimization for Lasso}

\subsection{Mini-batch stochastic gradient descent}

\subsection{Experimental setup}

\section{Implementation and validation}

\section{Results and discussion}
```

Henvis til relevante fagkilder der teori og algoritmer introduseres. Oppgaveteksten ber også om en egen LLM-erklæring etter konklusjonen; den bør inngå i den samlede rapportstrukturen.
