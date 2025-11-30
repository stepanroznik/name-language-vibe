# name-language-vibe

Estimate which languages a given name *resembles*.  
Feed it any name and get a ranked probability distribution across supported languages.  
Models are trained from Faker.js datasets.

## Features
- Predicts “language vibe” of a name  
- Separate male/female models  
- Simple CLI interface  
- Deterministic training using Faker.js

## Installation
```bash
npm ci
```
## Training
```bash
npm run train
```

## Prediction
```bash
npm run predict John
```

## You should get an output like this:
``` makefile
Male model
Prediction for "john" (male):
en: 0.403
de: 0.314
nl: 0.212
da: 0.046
fr: 0.009
es: 0.004
hr: 0.003
cs_CZ: 0.003
pl: 0.002
sr_RS_latin: 0.001
sk: 0.001
mk: 0.001
ro: 0.001
uk: 0.000
tr: 0.000
it: 0.000

Female model
Prediction for "john" (female):
en: 0.426
de: 0.239
nl: 0.133
cs_CZ: 0.069
da: 0.027
sr_RS_latin: 0.016
fr: 0.016
sk: 0.015
pl: 0.015
hr: 0.012
es: 0.008
tr: 0.006
mk: 0.005
ro: 0.005
uk: 0.005
it: 0.004
```