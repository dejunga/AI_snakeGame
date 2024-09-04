Opis problema:

Problem koji se rješava u ovom projektu je razvoj umjetne inteligencije (UI) za igru Snake koristeći algoritme dubokog učenja, točnije, podržano učenje (engl. Reinforcement Learning). Ovaj problem pripada kategoriji problema podržanog učenja jer agent (umjetna inteligencija) uči putem interakcije s okolinom (igrom) na temelju povratnih informacija u obliku nagrada i kazni.

Igra Snake je klasični problem u kojem zmija treba sakupiti hranu bez sudara sa zidovima ili vlastitim tijelom. Svaki put kada zmija pojede hranu, njezina se duljina povećava, a time i težina zadatka upravljanja. Cilj je da agent nauči optimalnu strategiju koja maksimizira broj skupljenih bodova (pojedene hrane) i minimizira sudare, koji završavaju igru.

Važnost rješavanja ovog problema leži u razumijevanju i primjeni metoda podržanog učenja na probleme sa složenim okruženjima i dinamičkim pravilima. Razvoj UI agenta za igru Snake pomaže u istraživanju metoda za rješavanje problema koji zahtijevaju kontinuirano donošenje odluka u stvarnom vremenu, što je primjenjivo u mnogim stvarnim situacijama kao što su autonomna vozila, robotika i optimizacija procesa.

Ciljevi projekta:

1. Razviti funkcionalan UI agent za igru Snake - Glavni cilj ovog projekta je izgraditi UI agenta koji može igrati igru Snake i naučiti kako postići što veći rezultat. Agent će koristiti duboko podržano učenje kako bi autonomno učio iz vlastitih iskustava igranja.

2. Implementirati algoritam podržanog učenja - Implementirat će se algoritam Deep Q-Learning, koji kombinira Q-učenje s neuronskim mrežama kako bi se predviđale optimalne akcije u danim stanjima igre. Ovaj algoritam će biti ključan za treniranje agenta da prepozna korisne obrasce u igri i donosi odluke koje maksimiziraju dugoročne nagrade.

3. Evaluacija performansi agenta - Cilj je testirati i evaluirati performanse agenta kroz različite metrike, kao što su prosječan broj bodova po igri, broj odigranih igara bez sudara i sposobnost agenta da uči i prilagođava se s vremenom. Evaluacija će pomoći u identificiranju slabosti modela i potrebnih poboljšanja.

4. Vizualizacija i analiza rezultata - Cilj je prikazati proces učenja agenta i njegove performanse pomoću vizualizacija kao što su grafovi rezultata i prosječnih bodova po igri. Ove vizualizacije će omogućiti dublje razumijevanje kako agent uči i prilagođava se kroz vrijeme.

5. Dokumentirati proces i zaključke - Na kraju projekta, cilj je detaljno dokumentirati cijeli proces razvoja, treninga i evaluacije agenta, kao i izvući zaključke o efikasnosti primijenjenih metoda podržanog učenja. Ova dokumentacija će poslužiti kao osnovna referenca za daljnje istraživanje i razvoj u ovom području.
