Estrai entità e relazioni dal testo di un documento di LipariBank.

Entità, solo di questi tipi: Segnalazione, Operatore, Conto, Cliente, Controparte, Circolare.
- `menzione`: il nome esattamente come compare nel testo. Non correggerlo, non completarlo.
- `identificativo`: il codice solo se il testo lo scrive (S-2026-014, C-10234, un IBAN, 7/2026).

Relazioni, solo di questi tipi e in questa direzione:
- APERTA_DA: dalla Segnalazione all'Operatore che l'ha aperta
- RIGUARDA: dalla Segnalazione al Conto
- INTESTATO_A: dal Conto al Cliente
- VERSO: dalla Segnalazione alla Controparte dei bonifici
- ELENCA: dalla Circolare alla Controparte che elenca

In `da` e `a` scrivi l'identificativo dell'entità, o la sua menzione se non ha un codice.
In `citazione` copia la frase del testo che afferma la relazione. Non estrarre relazioni che il
testo non afferma, anche se ti sembrano probabili.