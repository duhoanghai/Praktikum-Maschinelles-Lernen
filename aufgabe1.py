'''
Praktikum 1: Grundlagen einfacher neuronaler Netze
Gruppe 6

@author: Hai Hoang Du, Vivien Schulze, Eric Staudt
'''

import numpy as np
from pathlib import Path

#Definition der Parameter
'''
NumHLneurons: Number of hidden layer neurons
NumWeights: Number of weights (input+bias layer,hidden layer,output layer)
NumEpochs: Number of epochs
NumTrainPoints: Number of grid points
NumValPoints: Number of validation points
Xstart: Startpunkt 
Xend: Endpunkt (PI=3.1415, 3*PI=12.566)
alpha: konstante Lernrate für den Gradientenabstieg
beta: Koeffizient des Momentumterms
'''

# Korrektur zum Originalkommentar: 3*pi ≈ 9.42478; 4*pi ≈ 12.56637.
# Standardwerte für den Test aus Aufgabe 1; Aufgabe 11/12 setzt [0, pi].
NumTrainPoints = 3  #3
NumValPoints   = 100
Xstart = 0
Xend   = 1           #1
alpha  = 0.1
beta   = 0.9 

#Definition des Gewichtsvektors
'''
Der Gewichtsvektor enthält die Gewichte wie folgt: 
w=(input->hidden,
   bias->hidden,
   hidden->output)^T
Der obere Gewichtsvektor enthält Werte für die Berechnungen
mit 5 hidden layer neurons, während der zweite Gewichtsvektor
für die Berechnung mit 10 hidden layer neurons zu verwenen ist.
'''


def computeGridPoints(NumTrainPoints, NumValPoints):
   '''
   Berechnung der Gitterpunkte im Intervall [Xstart,Xend].
   Die Gitterpunkte sind gleichverteilt.
   '''
      
   # Aufgabe 3: linspace schließt beide Intervallgrenzen ein.
   TrainGrid = np.linspace(Xstart, Xend, NumTrainPoints)    #(0,1,3) -> [0, 0.5,1]
   ValGrid   = np.linspace(Xstart, Xend, NumValPoints)      #(0,1,100) -> [0, 0.01010101, 0.02020202, ..., 0.98989899, 1]
   
   return TrainGrid, ValGrid

   
def NeuralNetwork(GridPoint,w):
   '''
   Berechnet für einen einzelnen x-Wert die Ausgabe des neuronalen Netzes.
   '''
   
   # Die Neuronenzahl folgt aus den drei gleich langen Gewichtsblöcken.
   # Dadurch funktioniert diese Funktion auch nach einem Import der Datei.
   if np.ndim(w) != 1 or len(w) == 0 or len(w) % 3 != 0:
      raise ValueError("w muss ein Vektor mit 3 Gewichten pro Neuron sein.")
   NumHLNeurons = len(w) // 3
   # Aufgabe 4: Die drei Blöcke enthalten nu, eta und rho in dieser Reihenfolge.
   nu  = w[:NumHLNeurons]                    #[:5]
   eta = w[NumHLNeurons:2 * NumHLNeurons]    #[5:10]
   rho = w[2 * NumHLNeurons:]                #[10:]

   z = nu * GridPoint + eta   # gewichtete Summe
   TanhNeurons = np.tanh(z)   # Tanh-Aktivierungsfunktion

   # Das Ausgabeneuron ist linear und besitzt hier keinen zusätzlichen Bias.
   OutputNeuron = np.dot(rho, TanhNeurons)   # Skalarprodukt
   
   return TanhNeurons,OutputNeuron       


def LossFunction(GridPoint,OutputNeuron):  
   '''
   Berechnung des Fehlers.
   '''
   
   # Aufgabe 5: Einzelpunkt-Loss E_i = 1/2 * (N(x_i) - cos(x_i))**2.
   # Die Tabellenwerte sind gerundet; hier wird cos ohne Vorrundung verwendet.
   error = OutputNeuron - np.cos(GridPoint)
   loss = 0.5 * error ** 2
   
   return loss

   
def LossFunctionGradient(GridPoint,TanhNeurons,OutputNeuron,w):   
   ''' 
   Berechnung der Ableitung des Fehlers -> in welche Richtung müssen die Gewichte 
   verändert werden, um den Fehler zu verringern.
   '''

   if np.ndim(w) != 1 or len(w) == 0 or len(w) % 3 != 0:
      raise ValueError("w muss ein Vektor mit 3 Gewichten pro Neuron sein.")
   NumHLNeurons = len(w) // 3
   if np.shape(TanhNeurons) != (NumHLNeurons,):
      raise ValueError("Die Anzahl der Tanh-Werte passt nicht zu w.")
   # Aufgabe 6: Kettenregel, mit tanh'(z) = 1 - tanh(z)**2.
   error = OutputNeuron - np.cos(GridPoint)
   rho = w[2 * NumHLNeurons:]
   tanhDerivative = 1.0 - TanhNeurons ** 2      # Ableitung

   # Gemeinsamer Faktor für die Ableitungen nach Eingangs- und Biasgewichten.
   hiddenGradient = error * rho * tanhDerivative
   gradNu = hiddenGradient * GridPoint
   gradEta = hiddenGradient
   gradRho = error * TanhNeurons

   # Dieselbe Reihenfolge wie im Gewichtsvektor; noch keine Mittelung.
   lossGrad = np.concatenate((gradNu, gradEta, gradRho))
   
   return lossGrad

      
def GradientDescent(w,dw,lossGrad):
   '''
   Training des neuronalen Netzes mit GradientDescent -> Update der Gewichte
   '''

   # Aufgabe 7: dw enthält beim Aufruf das Update der vorherigen Epoche.
   # lossGrad ist bereits der über alle Trainingspunkte gemittelte Gradient.
   dw = -alpha * lossGrad + beta * dw
   w = w + dw
   
   return w, dw


def MeanAbsoluteError(NumPoints, grid, w):
   '''
   Berechnung des numerischen Fehlers. 
   '''
   
   # Aufgabe 8: Nur auswerten, die Gewichte werden hier nicht verändert.
   if NumPoints <= 0 or NumPoints != len(grid):
      raise ValueError("NumPoints muss positiv sein und zur Gitterlänge passen.")
   NNsolution = np.zeros(NumPoints)
   
   for i in range(NumPoints):
      _, NNsolution[i] = NeuralNetwork(grid[i], w)
   Cosine = np.cos(grid)
   MAE = np.mean(np.abs(NNsolution - Cosine))
   
   return NNsolution, Cosine, MAE
   

# Aufgabe 9: Der Hauptteil 
if __name__ == "__main__":

   task = input("Aufgabe wählen: 10 = Test 1-9/Theorie, 11 oder 12: ").strip()
   if task not in ("10", "11", "12"):
      raise ValueError("Bitte '10', '11' oder '12' eingeben.")

   if task == "10":
      # Vorgegebener Vergleichsfall: genau 5 Neuronen, 3 Punkte, 1 Epoche.
      choice = "5"
      NumTrainPoints, NumEpochs = 3, 1
      Xstart, Xend = 0, 1
   else:
      choice = input("Wählen Sie die Anzahl der hidden layer neurons (5 oder 10): ").strip()
      NumTrainPoints = 10
      Xstart, Xend = 0, np.pi
      # Matplotlib wird nur für die Diagramme in Aufgabe 11 und 12 benötigt.
      import matplotlib.pyplot as plt
      OutputDir = Path(__file__).resolve().parent / "ergebnisse" / f"aufgabe{task}_{choice}neuronen"
      OutputDir.mkdir(parents=True, exist_ok=True)

   if choice == "5":
      NumHLNeurons = 5
      NumWeights   = 3 * NumHLNeurons

      #Gewichte für 5 hidden layer neurons
      w = np.array([0.112,0.205,-0.050,0.524,-0.333,
                  -0.185,0.241,-0.311,-0.198,0.124,
                  0.096,-0.413,0.157,0.390,-0.215])
      
   elif choice == "10":
      NumHLNeurons = 10
      NumWeights   = 3 * NumHLNeurons

      #Gewichte für 10 hidden layer neurons
      w = np.array([0.112,0.205,-0.050,0.524,-0.333,0.712,-0.451,-0.574,0.856,-0.756,
                  -0.185,0.241,-0.311,-0.198,0.124,0.077,0.527,0.659,0.546,0.603,
                  0.096,-0.413,0.157,0.390,-0.215,0.124,0.298,0.830,-0.784,0.933])

   else:
      raise ValueError("Ungültige Eingabe. Bitte '5' oder '10' eingeben.")

   if len(w) != NumWeights:
      raise ValueError("Die Anzahl der Gewichte passt nicht zur Neuronenzahl.")

   TrainGrid, ValGrid = computeGridPoints(NumTrainPoints, NumValPoints)
   InitialWeights = w.copy()
   InitialNN, TrainCosine, InitialMAE = MeanAbsoluteError(NumTrainPoints, TrainGrid, InitialWeights)
   InitialValNN, ValCosine, InitialValMAE = MeanAbsoluteError(NumValPoints, ValGrid, InitialWeights)
   InitialLoss = np.mean(0.5 * (InitialNN - TrainCosine) ** 2)

   print("=" * 72)
   print("Neuronales Netz: Training von cos(x)")
   print("=" * 72)
   print(f"Hidden-Neuronen: {NumHLNeurons}")
   print(f"Trainingspunkte: {NumTrainPoints} | Validierungspunkte: {NumValPoints}")
   print(f"Intervall: [{Xstart}, {Xend}]")
   print()
   print("Vor dem Training")
   print("-" * 72)
   print(f"{'x':>8} {'cos(x)':>14} {'Netz vorher':>16} {'Fehler':>16}")
   print("-" * 72)
   for x, target, prediction in zip(TrainGrid, TrainCosine, InitialNN):
      print(f"{x:8.3f} {target:14.6f} {prediction:16.6f} {prediction - target:16.6f}")
   print("-" * 72)
   print(f"Mittlerer Loss: {InitialLoss:.10f}")
   print(f"MAE:            {InitialMAE:.10f}")
   print(f"MAE Validierung:{InitialValMAE:.10f}")
   print()

   if task == "10":
      # Eine vollständige Epoche; alle Punkte verwenden dieselben Startgewichte.
      dw = np.zeros(NumWeights)
      loss = 0.0
      lossGrad = np.zeros(NumWeights)
      print("Einzelpunkte vor dem Update (ungerundeter Kosinus):")
      for step in range(NumTrainPoints):
         GridPoint = TrainGrid[step]
         TanhNeurons, OutputNeuron = NeuralNetwork(GridPoint, w)
         PointLoss = LossFunction(GridPoint, OutputNeuron)
         PointGrad = LossFunctionGradient(GridPoint, TanhNeurons, OutputNeuron, w)
         loss += PointLoss
         lossGrad += PointGrad
         print(f"x={GridPoint:.1f}: N={OutputNeuron:.10f}, Loss={PointLoss:.10f}")
         print("Gradient am Punkt:", PointGrad)
      loss /= NumTrainPoints
      lossGrad /= NumTrainPoints
      w, dw = GradientDescent(w, dw, lossGrad)
      TrainNN, TrainCosine, TrainMAE = MeanAbsoluteError(NumTrainPoints, TrainGrid, w)
      ValNN, ValCosine, ValMAE = MeanAbsoluteError(NumValPoints, ValGrid, w)
      FinalLoss = np.mean(0.5 * (TrainNN - TrainCosine) ** 2)
      print("Gemittelter Gradient:", lossGrad)
      print("Gewichtsupdate:", dw)
      print("Gewichte nach einer Epoche:", w)
      print(f"Loss vor Update: {loss:.10f}")
      print(f"Loss nach Update: {FinalLoss:.10f}")
      print(f"MAE Training: {TrainMAE:.10f}")
      print(f"MAE Validierung: {ValMAE:.10f}")

   elif task == "11":
      # Aufgabe 11: Vereinfachung der Eingabe unterschiedlicher Epochen mit Schleife
      EpochList = [10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000]
      LossValues = []
      TrainMAEValues = []
      ValMAEValues = []

      SavedCurves = {}
      PlotEpochs  = [10, 50, 100, 1000, 100000]

      # Ursprüngliche Gewichte sichern
      StartWeights = InitialWeights.copy()

      print()
      print("=" * 90)
      print("Zusammenfassung aller Trainingsläufe")
      print("=" * 90)
      print(f"{'Epochen':>10} {'Loss':>15} {'MAE Train':>15} {'MAE Val':>15}")
      print("-" * 90)

      for NumEpochs in EpochList:

         # Für jeden Lauf wieder mit denselben Startgewichten beginnen
         w = StartWeights.copy()

         # Momentum zurücksetzen
         dw = np.zeros(NumWeights)

         for epoch in range(NumEpochs):

            loss = 0.0
            lossGrad = np.zeros(NumWeights)

            for step in range(NumTrainPoints):

                  GridPoint = TrainGrid[step]

                  TanhNeurons, OutputNeuron = NeuralNetwork(GridPoint, w)

                  loss += LossFunction(GridPoint, OutputNeuron)

                  lossGrad += LossFunctionGradient(
                     GridPoint,
                     TanhNeurons,
                     OutputNeuron,
                     w
                  )

            # Mittelung über alle Trainingspunkte
            loss /= NumTrainPoints
            lossGrad /= NumTrainPoints

            # Genau ein Update pro Epoche
            w, dw = GradientDescent(w, dw, lossGrad)

         # Nach Abschluss dieses Trainingslaufes auswerten
         TrainNN, TrainCosine, TrainMAE = MeanAbsoluteError(
            NumTrainPoints,
            TrainGrid,
            w
         )

         ValNN, ValCosine, ValMAE = MeanAbsoluteError(
            NumValPoints,
            ValGrid,
            w
         )

         if NumEpochs in PlotEpochs:
            SavedCurves[NumEpochs] = ValNN.copy()

         FinalLoss = np.mean(
            0.5 * (TrainNN - TrainCosine) ** 2
         )

         LossValues.append(FinalLoss)
         TrainMAEValues.append(TrainMAE)
         ValMAEValues.append(ValMAE)

         print(
            f"{NumEpochs:10d}"
            f"{FinalLoss:15.8f}"
            f"{TrainMAE:15.8f}"
            f"{ValMAE:15.8f}"
         )

      print("=" * 90)
      np.savetxt(OutputDir / "ergebnisse.csv",
                 np.column_stack((EpochList, LossValues, TrainMAEValues, ValMAEValues)),
                 delimiter=";", header="Epochen;Loss;MAE_Training;MAE_Validierung",
                 comments="", fmt="%.12g")
      print(f"Tabellen und Diagramme werden gespeichert in: {OutputDir}")

         # Diagramm 1: Loss
      plt.figure(figsize=(8,5))

      plt.plot(
         EpochList,
         LossValues,
         marker="o",
         linewidth=2,
         label="Loss"
      )

      plt.xscale("log")
      plt.grid(True)

      plt.xlabel("Epochen")
      plt.ylabel("Loss")
      plt.title(f"Loss-Verlauf ({NumHLNeurons} Hidden-Neuronen)")
      plt.legend()

      # Vor dem Anzeigen für den Bericht speichern.
      plt.tight_layout()
      plt.savefig(OutputDir / "loss.png", dpi=200)
      plt.show()
      plt.close()



      # Diagramm 2: MAE
      plt.figure(figsize=(8,5))

      plt.plot(
         EpochList,
         TrainMAEValues,
         marker="o",
         linewidth=2,
         label="MAE Training"
      )

      plt.plot(
         EpochList,
         ValMAEValues,
         marker="s",
         linewidth=2,
         label="MAE Validierung"
      )

      plt.xscale("log")
      plt.grid(True)

      plt.xlabel("Epochen")
      plt.ylabel("MAE")
      plt.title(f"Mittlerer Absolutfehler ({NumHLNeurons} Hidden-Neuronen)")
      plt.legend()

      # Vor dem Anzeigen für den Bericht speichern.
      plt.tight_layout()
      plt.savefig(OutputDir / "mae.png", dpi=200)
      plt.show()
      plt.close()


      # Diagramm 3: Funktionsvergleich
      plt.figure(figsize=(10,6))

      # Referenzfunktion
      plt.plot(
         ValGrid,
         ValCosine,
         color="black",
         linewidth=3,
         label="cos(x)"
      )

      # Netzvorhersagen verschiedener Epochen
      for epochs in PlotEpochs:
         plt.plot(
            ValGrid,
            SavedCurves[epochs],
            label=f"NN ({epochs} Epochen)"
         )

      plt.xlabel("x")
      plt.ylabel("y")
      plt.title(
         f"Entwicklung der Cosinus-Approximation ({NumHLNeurons} Hidden-Neuronen)"
      )

      plt.grid(True)
      plt.legend()
      # Vor dem Anzeigen für den Bericht speichern.
      plt.tight_layout()
      plt.savefig(OutputDir / "funktionsvergleich.png", dpi=200)
      plt.show()
      plt.close()



   elif task == "12":
      NumEpochs = 100000

      TrainPointList = [5, 10, 20, 40, 80]

      StartWeights = InitialWeights.copy()

      print()
      print("=" * 95)
      print("Einfluss der Anzahl der Trainingspunkte")
      print("=" * 95)
      print(f"{'TrainPunkte':>12} {'Loss':>15} {'MAE Train':>15} {'MAE Val':>15}")
      print("-" * 95)

      LossValues     = []
      TrainMAEValues = []
      ValMAEValues   = []
      SavedCurves    = {}

      for NumTrainPoints in TrainPointList:

         # Trainingsgitter für aktuelle Punktzahl neu erzeugen
         TrainGrid, ValGrid = computeGridPoints(
            NumTrainPoints,
            NumValPoints
         )

         # Ausgangsgewichte zurücksetzen
         w = StartWeights.copy()

         # Momentum zurücksetzen
         dw = np.zeros(NumWeights)

         # Training
         for epoch in range(NumEpochs):

            loss = 0.0
            lossGrad = np.zeros(NumWeights)

            for step in range(NumTrainPoints):

                  GridPoint = TrainGrid[step]

                  TanhNeurons, OutputNeuron = NeuralNetwork(
                     GridPoint,
                     w
                  )

                  loss += LossFunction(
                     GridPoint,
                     OutputNeuron
                  )

                  lossGrad += LossFunctionGradient(
                     GridPoint,
                     TanhNeurons,
                     OutputNeuron,
                     w
                  )

            loss /= NumTrainPoints
            lossGrad /= NumTrainPoints

            w, dw = GradientDescent(
                  w,
                  dw,
                  lossGrad
            )

         # Training auswerten
         TrainNN, TrainCosine, TrainMAE = MeanAbsoluteError(
            NumTrainPoints,
            TrainGrid,
            w
         )

         # Validierung auswerten
         ValNN, ValCosine, ValMAE = MeanAbsoluteError(
            NumValPoints,
            ValGrid,
            w
         )

         SavedCurves[NumTrainPoints] = ValNN.copy()

         FinalLoss = np.mean(
            0.5 * (TrainNN - TrainCosine) ** 2
         )

         LossValues.append(FinalLoss)
         TrainMAEValues.append(TrainMAE)
         ValMAEValues.append(ValMAE)

         print(
            f"{NumTrainPoints:12d}"
            f"{FinalLoss:15.8f}"
            f"{TrainMAE:15.8f}"
            f"{ValMAE:15.8f}"
         )

      print("=" * 95)
      np.savetxt(OutputDir / "ergebnisse.csv",
                 np.column_stack((TrainPointList, LossValues, TrainMAEValues, ValMAEValues)),
                 delimiter=";", header="Trainingspunkte;Loss;MAE_Training;MAE_Validierung",
                 comments="", fmt="%.12g")
      print(f"Tabellen und Diagramme werden gespeichert in: {OutputDir}")

      # Diagramm 1: Loss
      plt.figure(figsize=(8,5))

      plt.plot(
         TrainPointList,
         LossValues,
         marker="o",
         linewidth=2
      )

      plt.grid(True)

      plt.xlabel("Trainingspunkte")
      plt.ylabel("Loss")

      plt.title(
         f"Loss in Abhängigkeit der Trainingspunkte ({NumHLNeurons} Neuronen)"
      )

      # Vor dem Anzeigen für den Bericht speichern.
      plt.tight_layout()
      plt.savefig(OutputDir / "loss.png", dpi=200)
      plt.show()
      plt.close()


      # Diagramm 2: MAE
      plt.figure(figsize=(8,5))

      plt.plot(
         TrainPointList,
         TrainMAEValues,
         marker="o",
         linewidth=2,
         label="MAE Training"
      )

      plt.plot(
         TrainPointList,
         ValMAEValues,
         marker="s",
         linewidth=2,
         label="MAE Validierung"
      )

      plt.grid(True)

      plt.xlabel("Trainingspunkte")
      plt.ylabel("MAE")

      plt.title(
         f"Fehler in Abhängigkeit der Trainingspunkte ({NumHLNeurons} Neuronen)"
      )

      plt.legend()
      # Vor dem Anzeigen für den Bericht speichern.
      plt.tight_layout()
      plt.savefig(OutputDir / "mae.png", dpi=200)
      plt.show()
      plt.close()


      # Diagramm 3: Funktionsvergleich
      plt.figure(figsize=(10,6))

      plt.plot(
         ValGrid,
         ValCosine,
         color="black",
         linewidth=3,
         label="cos(x)"
      )

      for points in TrainPointList:

         plt.plot(
            ValGrid,
            SavedCurves[points],
            label=f"{points} Trainingspunkte"
         )

      plt.xlabel("x")
      plt.ylabel("y")

      plt.title(
         f"Einfluss der Trainingspunktanzahl ({NumHLNeurons} Neuronen)"
      )

      plt.grid(True)
      plt.legend()

      # Vor dem Anzeigen für den Bericht speichern.
      plt.tight_layout()
      plt.savefig(OutputDir / "funktionsvergleich.png", dpi=200)
      plt.show()
      plt.close()


   else:
      raise ValueError("Ungültige Eingabe. Bitte '11' oder '12' eingeben.")
      

   # Implementierung für Aufgabe 1-9
   # # Vor der ersten Epoche gibt es noch kein vorheriges Update: Delta w(0) = 0.
   # dw = np.zeros(NumWeights)

   # for epoch in range(NumEpochs):
   #    # Die Summen werden zu Beginn jeder Epoche auf null zurückgesetzt.
   #    loss = 0.0
   #    lossGrad = np.zeros(NumWeights)

   #    for step in range(NumTrainPoints):
   #       GridPoint = TrainGrid[step]
   #       TanhNeurons, OutputNeuron = NeuralNetwork(GridPoint, w)
   #       loss += LossFunction(GridPoint, OutputNeuron)
   #       lossGrad += LossFunctionGradient(GridPoint, TanhNeurons, OutputNeuron, w)
   #       # Innerhalb dieser Schleife bleiben alle Gewichte unverändert.

   #    # Batch-Gradientenabstieg: Erst mitteln, dann genau ein Update pro Epoche.
   #    loss /= NumTrainPoints
   #    lossGrad /= NumTrainPoints
   #    w, dw = GradientDescent(w, dw, lossGrad)
   #    print(f"Epoche {epoch + 1}: Loss vor Update = {loss:.10f}")

   # # Abschließende Auswertung mit den aktualisierten Gewichten, ohne Training.
   # TrainNN, TrainCosine, TrainMAE = MeanAbsoluteError(NumTrainPoints, TrainGrid, w)
   # ValNN, ValCosine, ValMAE = MeanAbsoluteError(NumValPoints, ValGrid, w)
   # FinalLoss = np.mean(0.5 * (TrainNN - TrainCosine) ** 2)

   # print("Nach dem Training")
   # print("-" * 72)
   # print(f"{'x':>8} {'cos(x)':>14} {'Netz vorher':>16} {'Netz nachher':>16} {'Fehler nachher':>16}")
   # print("-" * 72)
   # for x, target, before, after in zip(TrainGrid, TrainCosine, InitialNN, TrainNN):
   #    print(f"{x:8.3f} {target:14.6f} {before:16.6f} {after:16.6f} {after - target:16.6f}")
   # print("-" * 72)
   # print(f"Mittlerer Loss: {FinalLoss:.10f} (vorher: {InitialLoss:.10f})")
   # print(f"MAE Training:   {TrainMAE:.10f} (vorher: {InitialMAE:.10f})")
   # print(f"MAE Validierung: {ValMAE:.10f} (vorher: {InitialValMAE:.10f})")
   # print()
   # print("Gewichtsaenderungen")
   # print("-" * 72)
   # print(f"{'Nr.':>5} {'vorher':>16} {'nachher':>16} {'Aenderung':>16}")
   # print("-" * 72)
   # for number, (before, after) in enumerate(zip(InitialWeights, w), start=1):
   #    print(f"{number:5d} {before:16.6f} {after:16.6f} {after - before:16.6f}")
   # print("=" * 72)