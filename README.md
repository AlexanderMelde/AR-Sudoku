# Prüfungsleistung Bildverarbeitung
| Details              |                                                                                               |
|----------------------|-----------------------------------------------------------------------------------------------|
| Autoren              | Alexander Melde (7939560)<br>Anja Ohlhäuser (6986288)<br>Nina Zaske (4627174)                 |
| Betreuer             | Stefan Gehrig                                                                                 |
| Vorlesung            | Digitale Bildverarbeitung (6. Semester)                                                       |
| Studiengang/Kurs     | B. Sc. Angewandte Informatik – Kommunikationsinformatik TINF15K                               |
| Titel der Arbeit     | Projekt zu Grundlagen der Bildverarbeitung: Entwicklung eines Augmented Reality Sudoku-Lösers |
| Bearbeitungszeitraum | 14.03.2018 - 10.04.2018                                                                       |
| Abgabedatum          | 11.04.2018                                                                                    |


### Aufgabe
Im Rahmen einer Vorlesung sollte unter Verwendung des OpenCV Frameworks ein beliebiges Programm entwickelt werden, dass Techniken der digitalen Bildverarbeitung verwendet.

### Umsetzung
Das mit Python 3 entwickelte Programm sucht im Kamerabild einer Computer-Webcam nach Sudoku-Feldern, extrahiert das Sudoku-Feld sowie jedes einzelne der 9x9 Zahlen-Felder, erkennt die Zahlen in den Feldern, löst das Sudoku und ergänzt die fehlenden Zahlen des Sudokus in Echtzeit im Kamerabild.

#### Eigenanteil
- Erkennung des Sudoku-Quadrats im Gesamtbild
- Aufteilung des Sudoku-Quadrats in 9x9 Felder mit zwei Versionen:
    - Erkennen der kleinen Quadrate anhand deren Konturen (Genauere Erkennung des Sudoku Felds und ermöglicht Validierung ob Quadrat ein Sudoku-Feld ist, oft werden aber nicht alle Felder erkannt.)
    - Feste Aufteilung des 9x9 Felds mit mathematischen Rechnungen (Es werden immer 9x9 Felder „erkannt“, kann nicht zur Validierung von echten Sudoku-Feldern genutzt werden)
- Anpassen eines Sudoku-Lösungs-Algorithmus an unsere Datenstruktur
- Einbinden eines Machine Learning Frameworks und Trainieren des Modells auf handschriftliche Zahlen (hierfür gab es bereits einen Datensatz)
- Vorbereiten der kleinen Sudoku-Bildausschnitte für die Ziffern-Klassifikation (Zuschnitt, Resizing, Farbwertumwandlung)
- Auswertung der Klassifikations-Antwort (Wahrscheinlichkeiten berücksichtigen um leere Felder zu unterscheiden)
- Anzeige der gelösten Sudoku-Ziffern im zugeschnittenen Bild und Ausgabe des Bilds
- Option zum Einlesen von Beispielbildern und zur Verwendung der Webcam als Eingabemöglichkeit

### Funktionstest
- Grundsätzlich wird die Zielsetzung für manche Bilder erreicht
- In das Bild gehalten Sudokus werden in den meisten Fällen korrekt erkannt
- Ziffern werden oft nicht korrekt erkannt, weshalb der Sudoku-Solver eine falsche Eingabe erhält und dann auch die Ausgabe nicht stimmt.
- In Modultests wurde die Funktionalität des Sudoku-Solvers bestätigt 

### Weiterentwicklungsmöglichkeiten
- Einblendung des gelösten Sudokus in Original-Kamerabild (statt in Sudoku-Zuschnitt, erfordert rückrechnung der zum zugeschnittenen Sudoku relativen 9x9-Positionen zum Originialbild)
- Verbesserung der Ziffern-Erkennung (Datensatz mit höherer Auflösung und Computerschriftarten verwenden)

### Entwicklungsumgebung
Zur Entwicklung wurde die Python Version 3.6.3 unter Windows und Mac verwendetet.

**Verwendete Python Module**
- opencv-python
- pillow
- imutils
- numpy
- scipy
- skikit-learn
- sklearn
