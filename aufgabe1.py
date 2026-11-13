import numpy as np


#Definition der Parameter
'''
NumHLneurons: Number of hidden layer neurons
NumWeights: Number of weights (input+bias layer,hidden layer,output layer)
NumEpochs: Number of epochs
NumTrainPoints: Number of grid points
Xstart: Startpunkt 
Xend: Endpunkt (PI=3.1415, 3*PI=12.566)
alpha: konstante Lernrate für den Gradientenabstieg
beta: Koeffizient des Momentumterms
'''


# Aufgabe 1: Testparameter aus der Aufgabenstellung.
NumHLneurons = 5
NumWeights = 3 * NumHLneurons
NumEpochs = int(1)
NumTrainPoints = 3
NumValPoints = 100
Xstart = 0
Xend = 1
alpha = 0.1
beta = 0.9
# Hinweis zum unveränderten Kommentar oben: 3 * pi ist ungefähr 9.425;
# 12.566 entspricht ungefähr 4 * pi. Hier wird das Testintervall [0, 1] benutzt.

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

#Gewichte für 5 hidden layer neurons
w = np.array([0.112,0.205,-0.050,0.524,-0.333,
              -0.185,0.241,-0.311,-0.198,0.124,
              0.096,-0.413,0.157,0.390,-0.215])

#Gewichte für 10 hidden layer neurons
"""
w = [0.112,0.205,-0.050,0.524,-0.333,0.712,-0.451,-0.574,0.856,-0.756,
     -0.185,0.241,-0.311,-0.198,0.124,0.077,0.527,0.659,0.546,0.603,
     0.096,-0.413,0.157,0.390,-0.215,0.124,0.298,0.830,-0.784,0.933]
"""

def computeGridPoints(NumTrainPoints, NumValPoints):
   '''
   Berechnung der Gitterpunkte im Intervall [Xstart,Xend].
   Die Gitterpunkte sind gleichverteilt.
   '''
   
      
   # Aufgabe 3: linspace schließt beide Intervallgrenzen ein.
   TrainGrid = np.linspace(Xstart, Xend, NumTrainPoints)
   ValGrid = np.linspace(Xstart, Xend, NumValPoints)
   return TrainGrid, ValGrid
   
def NeuralNetwork(GridPoint,w):
   '''
   Berechnung des Neuronales Netzwerkes.
   '''
   
   
   # Aufgabe 4: Die drei Blöcke enthalten nu, eta und rho in dieser Reihenfolge.
   nu = w[:NumHLneurons]
   eta = w[NumHLneurons:2 * NumHLneurons]
   rho = w[2 * NumHLneurons:]
   # Das Biasneuron liefert 1; sein Beitrag ist deshalb direkt eta.
   z = nu * GridPoint + eta
   TanhNeurons = np.tanh(z)
   # Das Ausgabeneuron ist linear und besitzt hier keinen zusätzlichen Bias.
   OutputNeuron = np.dot(rho, TanhNeurons)
   return TanhNeurons,OutputNeuron       

def LossFunction(GridPoint,OutputNeuron):  
   '''
   Berechnung der Lossfunktion.
   '''
   
   
   # Aufgabe 5: Einzelpunkt-Loss E_i = 1/2 * (N(x_i) - cos(x_i))**2.
   # Die Tabellenwerte sind gerundet; hier wird cos ohne Vorrundung verwendet.
   error = OutputNeuron - np.cos(GridPoint)
   loss = 0.5 * error ** 2
   return loss
   
def LossFunctionGradient(GridPoint,TanhNeurons,OutputNeuron,w):   
   ''' 
   Berechnung des Gradienten der Lossfunktion für das Training.
   '''


   # Aufgabe 6: Kettenregel, mit tanh'(z) = 1 - tanh(z)**2.
   error = OutputNeuron - np.cos(GridPoint)
   rho = w[2 * NumHLneurons:]
   tanhDerivative = 1.0 - TanhNeurons ** 2
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
   Training des neuronalen Netzes mit GradientDescent.
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
   
   


# Aufgabe 9: Der Hauptteil wird nur beim direkten Start der Datei ausgeführt.
# Damit können die Funktionen auch separat importiert und überprüft werden.
if __name__ == "__main__":
   if len(w) != NumWeights:
      raise ValueError("Die Anzahl der Gewichte passt nicht zur Neuronenzahl.")

   TrainGrid, ValGrid = computeGridPoints(NumTrainPoints, NumValPoints)
   # Vor der ersten Epoche gibt es noch kein vorheriges Update: Delta w(0) = 0.
   dw = np.zeros(NumWeights)

   for epoch in range(NumEpochs):
      # Die Summen werden zu Beginn jeder Epoche auf null zurückgesetzt.
      loss = 0.0
      lossGrad = np.zeros(NumWeights)

      for step in range(NumTrainPoints):
         GridPoint = TrainGrid[step]
         TanhNeurons, OutputNeuron = NeuralNetwork(GridPoint, w)
         loss += LossFunction(GridPoint, OutputNeuron)
         lossGrad += LossFunctionGradient(GridPoint, TanhNeurons, OutputNeuron, w)
         # Innerhalb dieser Schleife bleiben alle Gewichte unverändert.

      # Batch-Gradientenabstieg: Erst mitteln, dann genau ein Update pro Epoche.
      loss /= NumTrainPoints
      lossGrad /= NumTrainPoints
      w, dw = GradientDescent(w, dw, lossGrad)
      print(f"Epoche {epoch + 1}: mittlerer Loss vor Update = {loss:.10f}")

   # Abschließende Auswertung mit den aktualisierten Gewichten, ohne Training.
   TrainNN, TrainCosine, TrainMAE = MeanAbsoluteError(NumTrainPoints, TrainGrid, w)
   ValNN, ValCosine, ValMAE = MeanAbsoluteError(NumValPoints, ValGrid, w)
   FinalLoss = np.mean(0.5 * (TrainNN - TrainCosine) ** 2)
   print("Trainingspunkte:", TrainGrid)
   print("Gewichte nach dem Training:", w)
   print(f"Mittlerer Trainingsloss nach Training = {FinalLoss:.10f}")
   print(f"MAE Training = {TrainMAE:.10f}")
   print(f"MAE Validierung = {ValMAE:.10f}")
