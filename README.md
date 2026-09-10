# VisionXI – Player Detection & Performance Analysis

## Over het project

**VisionXI** is een Computer Vision-applicatie voor het automatisch detecteren en tracken van voetballers in wedstrijd- en trainingsvideo's.

Het doel van VisionXI is om voetbalcoaches, clubs en analisten te ondersteunen bij het analyseren van spelers en hun bewegingen. In plaats van volledige wedstrijden handmatig te bekijken, kan VisionXI automatisch relevante informatie uit videobeelden halen.

Het project is ontwikkeld met het oog op toepassingen binnen de **Surinaamse voetbalwereld**, waar data-gedreven analyse kan bijdragen aan talentontwikkeling, training en wedstrijdanalyse.



## Probleemstelling

Het analyseren van voetbalwedstrijden gebeurt vaak handmatig. Coaches en analisten moeten video's bekijken om informatie te verzamelen over de bewegingen en positionering van spelers.

Dit proces:

- kost veel tijd;
- is afhankelijk van menselijke observatie;
- kan moeilijk consistent worden uitgevoerd;
- kan ervoor zorgen dat talent en prestatiegegevens worden gemist;
- maakt het lastiger om grote hoeveelheden videobeelden te analyseren.

VisionXI probeert dit proces gedeeltelijk te automatiseren met behulp van Computer Vision.



## Oplossing

VisionXI gebruikt Computer Vision en Machine Learning om spelers in voetbalvideo's automatisch te:

1. **Detecteren**
2. **Tracken**
3. **Classificeren per team**
4. **Analyseren op basis van hun bewegingen**

De output kan vervolgens worden gebruikt voor verdere performance-analyse.



## Belangrijkste doelstellingen

- Voetbalspelers automatisch detecteren.
- Spelers gedurende een video volgen.
- Spelers onderscheiden en aan teams koppelen.
- Spelerbewegingen analyseren.
- Prestatiegegevens uit videobeelden verzamelen.
- Coaches en analisten ondersteunen bij videoanalyse.



## Technologieën

| Technologie | Gebruik |
|---|---|
| Python | Development en data processing |
| YOLOv8 | Player detection |
| OpenCV | Video- en beeldverwerking |
| ByteTrack / DeepSORT | Object tracking |
| Roboflow | Dataset preparation en annotation |
| Pandas | Data processing |
| NumPy | Numerical processing |
| Streamlit | Webapplicatie |
| GitHub | Version control en samenwerking |

---

## Pipeline

De verwerking van een voetbalvideo verloopt in verschillende stappen:

```text
Video Input
    ↓
Frame Extraction
    ↓
Player Detection
    ↓
Player Tracking
    ↓
Team Classification
    ↓
Player Movement Analysis
    ↓
Performance Data
    ↓
Results / Web Application
```

### 1. Video Input
De gebruiker uploadt een wedstrijd- of trainingsvideo.

### 2. Frame Extraction
De video wordt verwerkt als een reeks afzonderlijke frames.

### 3. Player Detection
YOLOv8 wordt gebruikt om spelers in de frames te detecteren.

### 4. Player Tracking
ByteTrack wordt gebruikt om spelers over meerdere frames te volgen.

### 5. Team Classification
De gedetecteerde spelers worden aan een team toegewezen.

### 6. Movement Analysis
De trackinggegevens kunnen worden gebruikt om bewegingen en positionering van spelers te analyseren.

### 7. Performance Data
De resultaten kunnen worden gebruikt voor verdere performance-analyse en rapportage.



## Webapplicatie

VisionXI bevat een webapplicatie waarmee de gebruiker videoanalyse kan uitvoeren.

De algemene workflow is:

```text
Upload Video
    ↓
Start Analysis
    ↓
Player Detection
    ↓
Player Tracking
    ↓
Team Classification
    ↓
Analysis Results
```

De applicatie is bedoeld om voetbalvideoanalyse toegankelijker te maken zonder dat de gebruiker rechtstreeks met de onderliggende Computer Vision-code hoeft te werken.



## Data

Voor het project worden voetbalwedstrijd- en trainingsvideo's gebruikt. De video's worden verwerkt en omgezet naar frames. Relevante objecten, zoals spelers, worden vervolgens voorzien van bounding boxes.

De dataset bestaat uit:

- **Training set** – voor het trainen van het model.
- **Validation set** – voor controle en verbetering tijdens de ontwikkeling.
- **Test set** – voor het evalueren van de uiteindelijke prestaties.



## Performance Analysis

De detectie- en trackingresultaten vormen de basis voor verdere analyse.

VisionXI kan worden gebruikt om onder andere inzicht te krijgen in:

- spelerbewegingen;
- positionering;
- betrokkenheid bij het spel;
- verplaatsingen op het veld;
- algemene bewegingspatronen.

Deze gegevens kunnen coaches en analisten ondersteunen bij het evalueren van spelers en teams.



## Resultaat

Na verwerking van een video kan het systeem informatie tonen over de gedetecteerde spelers en hun bewegingen.

De resultaten kunnen worden gebruikt als ondersteuning voor:

- wedstrijdanalyse;
- trainingsevaluatie;
- spelersontwikkeling;
- talentontwikkeling;
- tactische analyse.

VisionXI is hierbij bedoeld als **ondersteunend analyse-instrument** en niet als vervanging van de expertise van een coach of voetbalanalist.



## Doelgroep

VisionXI is voornamelijk gericht op:

- **Voetbalclubs in Suriname: SV Robinhood**
- **Voetbalcoaches: Roberto Gödeken**
- **Jeugdvoetbalacademies: FC Strati Academie**
- **Voetbalanalisten: Desney Romeo**
- **De Surinaamse Voetbalbond (SVB)**

De applicatie kan vooral interessant zijn voor organisaties die spelersontwikkeling en wedstrijdanalyse meer datagedreven willen uitvoeren.



## Klantwaarde

### Tijd besparen
Automatische detectie en tracking kunnen het handmatige analyseproces verminderen.

### Objectievere informatie
Videodata kan aanvullende informatie bieden naast de persoonlijke observatie van een coach.

### Spelersontwikkeling
Tracking- en bewegingsgegevens kunnen worden gebruikt om de ontwikkeling van spelers over tijd te volgen.

### Data-gedreven besluitvorming
Coaches en analisten kunnen videodata gebruiken als ondersteuning bij hun beslissingen.



## Uitdagingen

Tijdens de ontwikkeling zijn enkele uitdagingen vastgesteld:

- De detectie kan worden beïnvloed door beeldkwaliteit.
- Spelers kunnen tijdelijk worden gemist wanneer zij elkaar overlappen.
- Teamclassificatie is niet altijd correct.
- Lange video's vereisen relatief veel verwerkingstijd.
- Grote videobestanden kunnen langer duren om te uploaden en te verwerken.
- De huidige analyse is nog beperkt in vergelijking met professionele performance-analysis systemen.

Bij bepaalde video's werden scheidsrechters als Team A gedetecteerd en spelers van beide teams soms als Team B geclassificeerd. Teamclassificatie kan hierbij worden beïnvloed door shirtkleur, belichting, camerahoek, achtergrond en overlap tussen spelers.



## Toekomstige uitbreidingen

VisionXI kan in toekomstige versies verder worden uitgebreid met:

### Ball Tracking
Automatisch detecteren en volgen van de bal.

### Uitgebreidere Performance Analysis
Toevoegen van meetbare prestatie-indicatoren zoals:

- afgelegde afstand;
- snelheid;
- sprintbewegingen;
- aantal bewegingen;
- positionering.

### Automatische Talentanalyse
Op basis van verzamelde gegevens kunnen spelersprofielen worden opgesteld om talentontwikkeling verder te ondersteunen.

### Verbeterde Team Classification
Een betrouwbaarder classificatiemodel kan worden ontwikkeld om spelers, scheidsrechters en andere personen beter van elkaar te onderscheiden.

### Snellere verwerking
Optimalisatie van de verwerking om langere video's sneller te kunnen analyseren.



## Projectstructuur

```text
VisionXI/
│
├── data/
│   ├── train/
│   ├── validation/
│   └── test/
│
├── src/
│   ├── detection/
│   ├── tracking/
│   ├── classification/
│   └── analysis/
│
├── tests/
├── svg/
│
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
├── .python-version
└── README.md
```

De exacte structuur kan veranderen tijdens de verdere ontwikkeling van het project.



## Projectteam

**Team 4 – DataVision**


*Bronne Amresh &
Setropawiro Jennifer*

**Project:** VisionXI

**Projectnaam:**  
*Player Detection & Performance Analysis*

VisionXI is ontwikkeld als onderdeel van Advanced Computer Vision.



## Future Scope

Het uiteindelijke doel van VisionXI is om een basis te leggen voor een toegankelijk Computer Vision-platform voor voetbalanalyse in Suriname.

Door toekomstige functies zoals **ball tracking, uitgebreidere performance metrics en automatische talentanalyse** toe te voegen, kan het systeem verder worden ontwikkeld tot een uitgebreider hulpmiddel voor coaches, clubs en voetbalorganisaties.
