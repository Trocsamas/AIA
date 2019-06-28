# Ampliación de Inteligencia Artificial. Curso 18-19
# Implementación de clasificadores lineales
# Dpto. de C. de la Computación e I.A. (Univ. de Sevilla)
# ===================================================================


# --------------------------------------------------------------------------
# Autor(a) del trabajo:
#
# APELLIDOS: Fernández García
# NOMBRE: Jesús
#
# Segundo componente (si se trata de un grupo):
#
# APELLIDOS: Sala Mascort
# NOMBRE: Jaime Emilio
# ----------------------------------------------------------------------------


# *****************************************************************************
# HONESTIDAD ACADÉMICA Y COPIAS: un trabajo práctico es un examen, por lo que
# debe realizarse individualmente o con el compañero asignado (en el caso de
# un grupo). La discusión y el intercambio de información de carácter general
# con el resto de compañeros se permite (e incluso se recomienda), pero NO AL
# NIVEL DE CÓDIGO. Igualmente el remitir código de terceros, OBTENIDO A TRAVÉS
# DE LA RED o cualquier otro medio, se considerará plagio. Si tiene
# dificultades para realizar el ejercicio, consulte con el profesor. 

# Cualquier plagio o compartición de código que se detecte significará
# automáticamente la calificación de cero en la asignatura para TODOS los
# alumnos involucrados. Por tanto a estos alumnos NO se les conserva, ni para
# la actual ni para futuras convocatorias, ninguna nota que hubiesen obtenido
# hasta el momento. Sin perjuicio de las correspondientes MEDIDAS
# DISPCIPLINARIAS que se pudieran llevar a cabo.
# *****************************************************************************

import numpy as np
import random
import math

# IMPORTANTE: NO CAMBIAR EL NOMBRE NI A ESTE ARCHIVO NI A LAS CLASES Y MÉTODOS
# QUE SE PIDEN

# NOTA: En este trabajo se permite (y se aconseja) usar numpy. NO SE PUEDE
# usar scikit learn, salvo exclusivamente para cargar los datos de cáncer de
# mama, como se explica más adelante.

# ====================================================
# PARTE I: MODELOS LINEALES PARA CLASIFICACIÓN BINARIA
# ====================================================

# En esta primera parte se pide implementar en Python los siguientes
# clasificadores BINARIOS, todos ellos vistos en el tema 5.

# - Perceptron umbral
# - Regresión logística maximizando la verosimilitud:
#      * Versión batch
#      * Versión estocástica
#      * Versión mini-batch


# --------------------------------------------
# I.1. Generando conjuntos de datos aleatorios
# --------------------------------------------

# Previamente a la implementación de los clasificadores, conviene tener
# funciones que generen aleatoriamente conjuntos de datos fictícios. 
# En concreto, se pide implementar estas dos funciones:

# * Función genera_conjunto_de_datos_l_s(rango,dim,n_datos): 

#   Debe devolver dos arrays X e Y, generados aleatoriamente. El array X debe
#   tener dos dimensiones (n_datos,dim), conteniendo un número n_datos de
#   ejemplos totales, cada uno con dim características, con valores entre
#   -rango y rango. El array unidimensional Y debe tener la clasificación
#   binaria (1 o 0) de cada ejemplo del conjunto X, en el mismo orden. El
#   conjunto de datos debe ser linealmente separable.

#   SUGERENCIA: generar en primer lugar un hiperplano aleatorio (mediante sus
#   coeficientes, elegidos aleatoriamente entre -rango y rango). Luego generar
#   aleatoriamente cada ejemplo de igual manera y clasificarlo como 1 o 0
#   dependiendo del lado del hiperplano en el que se situe. Eso asegura que el
#   conjunto de datos es linealmente separable.


# * Función genera_conjunto_de_datos_n_l_s(rango,dim,size,prop_n_l_s=0.1):

#   Como la anterior, pero el conjunto de datos debe ser no linealmente
#   separable. Para ello generar el conjunto de datos con la función anterior
#   y cambiar de clase a una proporción pequeña del total de ejemplos (por
#   ejemplo el 10%). La proporción se da con prop_n_l_s. 


def genera_conjunto_de_datos_l_s(rango,dim,n_datos): 
    
    xn = np.random.uniform(-rango,rango,(n_datos,dim))
    
    wn = np.random.uniform(-rango,rango,(1,dim))
    
    w0 = random.uniform(-1,1)
    
    y = np.sum(xn*wn, axis=1)+w0    
    
    y = (y>=0).astype(int)
    
    return xn, y
    

    
def genera_conjunto_de_datos_n_l_s(rango,dim,size,prop_n_l_s=0.1):
    
    xn, y = genera_conjunto_de_datos_l_s(rango,dim,size)

    for _ in range(0,int(size*prop_n_l_s)):
        index = random.randint(0,size)
        y[index] = int(not(y[index]))
        
    return xn, y




# -----------------------------------
# I.2. Clases y métodos a implementar
# -----------------------------------

# En esta sección se pide implementar cada uno de los clasificadores lineales
# mencionados al principio. Cada uno de estos clasificadores se implementa a
# través de una clase python, que ha de tener la siguiente estructura general:

# class NOMBRE_DEL_CLASIFICADOR():

#     def __init__(self,clases,normalizacion=False,
#                 rate=0.1,rate_decay=False,batch_tam=64):

#          .....
         
#     def entrena(self,entr,clas_entr,n_epochs,
#                 reiniciar_pesos=False,pesos_iniciales=None):

#         ......

#     def clasifica_prob(self,ej):


#         ......

#     def clasifica(self,ej):


#         ......
        

# Explicamos a continuación cada uno de estos elementos:

# * NOMBRE_DEL_CLASIFICADOR:
# --------------------------


#  Este es el nombre de la clase que implementa el clasificador. 
#  Obligatoriamente se han de usar cada uno de los siguientes
#  nombres:

#  - Perceptrón umbral: 
#                       Clasificador_Perceptron
#  - Regresión logística, maximizando verosimilitud, batch: 
#                       Clasificador_RL_ML_Batch
#  - Regresión logística, maximizando verosimilitud, estocástico: 
#                       Clasificador_RL_ML_St
#  - Regresión logística, maximizando verosimilitud, mini-batch: 
#                       Clasificador_RL_ML_MiniBatch



# * Constructor de la clase:
# --------------------------

#  El constructor debe tener los siguientes argumentos de entrada:

#  - Una lista clases con los nombres de las clases del problema de
#    clasificación, tal y como aparecen en el conjunto de datos. 
#    Por ejemplo, en el caso del problema de las votaciones, 
#    esta lista sería ["republicano","democrata"]

#  - El parámetro normalizacion, que puede ser True o False (False por
#    defecto). Indica si los datos se tienen que normalizar, tanto para el
#    entrenamiento como para la clasificación de nuevas instancias.  La
#    normalización es una técnica que suele ser útil cuando los distintos
#    atributos reflejan cantidades numéricas de muy distinta magnitud.
#    En ese caso, antes de entrenar se calcula la media m_i y la desviación
#    típica d_i en cada componente i-esima (es decir, en cada atributo) de los
#    datos del conjunto de entrenamiento.  A continuación, y antes del
#    entrenamiento, esos datos se transforman de manera que cada componente
#    x_i se cambia por (x_i - m_i)/d_i. Esta misma transformación se realiza
#    sobre las nuevas instancias que se quieran clasificar.  NOTA: se aconseja
#    usar los métodos de numpy para calcular la media y la desviación típica.

#  - rate: si rate_decay es False, rate es la tasa de aprendizaje fija usada
#    durante todo el aprendizaje. Si rate_decay es True, rate es la
#    tasa de aprendizaje inicial. Su valor por defecto es 0.1.

#  - rate_decay, indica si la tasa de aprendizaje debe disminuir en
#    cada epoch. En concreto, si rate_decay es True, la tasa de
#    aprendizaje que se usa en el n-ésimo epoch se debe de calcular
#    con la siguiente fórmula: 
#       rate_n= (rate_0)*(1/(1+n)) 
#    donde n es el número de epoch, y rate_0 es la cantidad
#    introducida en el parámetro rate anterior. Su valor por defecto es False 

#  - batch_tam (sólo aplicable a mini batch): indica el tamaño de los mini
#    batches (por defecto 64) que se usan para calcular cada actualización de
#    pesos.    


# * Método entrena:
# -----------------

#  Este método es el que realiza el entrenamiento del clasificador. 
#  Debe calcular un conjunto de pesos, mediante el correspondiente
#  algoritmo de entrenamiento. Describimos a continuación los parámetros de
#  entrada:  

#  - entr y clas_entr, son los datos del conjunto de entrenamiento y su
#    clasificación, respectivamente. El primero es un array con los ejemplos,
#    y el segundo un array con las clasificaciones de esos ejemplos, en el
#    mismo orden. 

#  - n_epochs: número de veces que se itera sobre todo el conjunto de
#    entrenamiento.


#  - reiniciar_pesos: si es True, cada vez que se llama a entrena, se
#    reinicia al comienzo del entrenamiento el vector de pesos de
#    manera aleatoria (típicamente, valores aleatorios entre -1 y 1).
#    Si es False, solo se inician los pesos la primera vez que se
#    llama a entrena. En posteriores veces, se parte del vector de
#    pesos calculado en el entrenamiento anterior. Esto puede ser útil
#    para continuar el aprendizaje a partir de un aprendizaje
#    anterior, si por ejemplo se dispone de nuevos datos.     


#
#  - pesos_iniciales: si es None (por defecto), se indica que los pesos deben
#    iniciarse aleatoriamente (por ejemplo, valores aleatorios entre -1 y
#    1). Si no es None, entonces se debe proporcionar la lista de pesos
#    iniciales. Esto puede ser útil para continuar el aprendizaje a partir de
#    un aprendizaje anterior, si por ejemplo se dispone de nuevos datos.    

#  NOTA: En las versiones estocásticas y mini batch, y en el perceptrón
#  umbral, en cada epoch recorrer todos los ejemplos del conjunto de
#  entrenamiento en un orden aleatorio distinto cada vez. 
#  SUGERENCIA IMPORTANTE : se aconseja no reordenar todos los ejemplos cada
#  vez, sino la lista de índices que nos sirven para acceder a los ejemplos. 


# * Método clasifica_prob:
# ------------------------

#  El método que devuelve la probabilidad de pertenecer a la clase (la que se
#  ha tomado como clase 1), calculada para un nuevo ejemplo. Este método no es
#  necesario incluirlo para el perceptrón umbral.


        
# * Método clasifica:
# -------------------
    
#  El método que devuelve la clase que se predice para un nuevo ejemplo. La
#  clase debe ser una de las clases del problema (por ejemplo, "republicano" o
#  "democrata" en el problema de los votos).  


# Si el clasificador aún no ha sido entrenado, tanto "clasifica" como
# "clasifica_prob" deben devolver una excepción del siguiente tipo:

class ClasificadorNoEntrenado(Exception): pass

#  NOTA: Se aconseja probar el funcionamiento de los clasificadores con
#  conjuntos de datos generados por las funciones del apartado anterior. 

# Ejemplo de uso:

# ------------------------------------------------------------

# Generamos un conjunto de datos linealmente separables, 
# In [1]: X1,Y1=genera_conjunto_de_datos_l_s(4,8,400)

# Lo partimos en dos trozos:
# In [2]: X1e,Y1e=X1[:300],Y1[:300]

# In [3]: X1t,Y1t=X1[300:],Y1[300:]

# Creamos el clasificador (perceptrón umbral en este caso): 
# In [4]: clas_pb1=Clasificador_Perceptron([0,1],rate_decay=True,rate=0.001)

# Lo entrenamos con elprimero de los conjuntos de datos:
# In [5]: clas_pb1.entrena(X1e,Y1e,100)

# Clasificamos un ejemplo del otro conjunto, y lo comparamos con su clase real:
# In [6]: clas_pb1.clasifica(X1t[0]),Y1t[0]
# Out[6]: (1, 1)

# Comprobamos el porcentaje de aciertos sobre todos los ejemplos de X2t
# In [7]: sum(clas_pb1.clasifica(x) == y for x,y in zip(X1t,Y1t))/len(Y1t)
# Out[7]: 1.0

# Repetimos el experimento, pero ahora con un conjunto de datos que no es
# linealmente separable: 
# In [8]: X2,Y2=genera_conjunto_de_datos_n_l_s(4,8,400,0.1)

# In [8]: X2e,Y2e=X2[:300],Y2[:300]

# In [9]: X2t,Y2t=X2[300:],Y2[300:]

# In [10]: clas_pb2=Clasificador_Perceptron([0,1],rate_decay=True,rate=0.001)

# In [11]: clas_pb2.entrena(X2e,Y2e,100)

# In [12]: clas_pb2.clasifica(X2t[0]),Y2t[0]
# Out[12]: (1, 0)

# In [13]: sum(clas_pb2.clasifica(x) == y for x,y in zip(X2t,Y2t))/len(Y2t)
# Out[13]: 0.82
# ----------------------------------------------------------------

class Clasificador_Perceptron():
    
    def __init__(self,clases,normalizacion=False,
                 rate=0.1,rate_decay=False):
        
        self.clases = clases
        self.normalizacion = normalizacion
        self.rate = rate
        self.rate_decay = rate_decay
        self.pesos = None
        self.mean = None
    
    def entrena(self,entr,clas_entr,n_epochs,
                reiniciar_pesos=False,pesos_iniciales=None):
        #Normalizacion, se crean atributos de clase para reusar
        #los valores en la clasificacion
        if(self.normalizacion==True):
            #Comprobamos que haya una normalizacion anterior 
            if( (type(self.mean) is type(None)) or reiniciar_pesos):
                self.mean = entr.mean(axis=0)
                self.std = entr.std(axis=0)
            an = (entr-self.mean)/self.std
        else:
            an = entr
            
        #Condiciones de entrenamieto
        if(reiniciar_pesos):
            wn = np.random.uniform(-1,1,(1,len(an[0])+1))   
        elif(pesos_iniciales):
            wn = pesos_iniciales
        elif(type(self.pesos) is type(None)):
            wn = np.random.uniform(-1,1,(1,len(an[0])+1))
        else:
            wn = self.pesos
            
        for n in range(0,n_epochs):
            #Rate Decay
            if(self.rate_decay):
                rate_n = (self.rate)*(1/(1+n))
            else:
                rate_n = self.rate
                
            #wn = wn+n*an[random index each epoch](y-o)
            ls_index = np.arange(0,len(an))
            np.random.shuffle(ls_index) 
            for index in ls_index:
                oum = (((np.sum(wn[:,1:]*an[index]))+wn[:,:1])>=0).astype(int)   
                wn[:,:1] = wn[:,:1] + rate_n*1*(clas_entr[index] - oum)
                wn[:,1:] = wn[:,1:] + rate_n*an[index]*(clas_entr[index] - oum)
        
        self.pesos = wn
        
    def clasifica(self,ej):
        
        #Normalizacion con los valores del entrenamiento
        if(self.normalizacion==True):
            an = (ej-self.mean)/self.std
        else:
            an = ej
            
        oum = (((np.sum(self.pesos[:,1:]*an))+self.pesos[:,:1])>=0)
        if(oum):
            res = self.clases[1]
        else:
            res = self.clases[0]
        return res
    

# ----------------------------------------------------------------

class Clasificador_RL_ML_Batch(): 
    
    def __init__(self,clases,normalizacion=False,
                 rate=0.1,rate_decay=False):
        
        self.clases = clases
        self.normalizacion = normalizacion
        self.rate = rate
        self.rate_decay = rate_decay
        self.pesos = None
        self.mean = None

    
    def entrena(self,entr,clas_entr,n_epochs,
                reiniciar_pesos=False,pesos_iniciales=None):
        
        #Normalizacion, se crean atributos de clase para reusar
        #los valores en la clasificacion
        if(self.normalizacion==True):
            #Comprobamos que haya una normalizacion anterior 
            if( (type(self.mean) is type(None)) or reiniciar_pesos):
                self.mean = entr.mean(axis=0)
                self.std = entr.std(axis=0)
            an = (entr-self.mean)/self.std
        else:
            an = entr
            
        #Condiciones de entrenamieto
        if(reiniciar_pesos):
            wn = np.random.uniform(-1,1,(1,len(an[0])+1))   
        elif(pesos_iniciales):
            wn = pesos_iniciales
        elif(type(self.pesos) is type(None)):
            wn = np.random.uniform(-1,1,(1,len(an[0])+1))
        else:
            wn = self.pesos
        
        for n in range(0,n_epochs):
                        
            if(self.rate_decay):
                rate_n = (self.rate)*(1/(1+n))
            else:
                rate_n = self.rate
            
            #wn <- wn + rate*sum((y+funcionprob(on))*x)
            #on = sum(pesos*entr) me da las sumas de cada peso por ejemplo
            #np.concatenate((a, b),axis=1) concatena a y b en la ultima columna
            #para añadir una lista de 1 al principio de todos los ejemplos para el w0
            
            on = np.sum(wn*np.concatenate((np.ones((len(an),1)),an),axis=1),axis=1)
            probn = 1/(1+np.power(math.e, -on))
            en = (clas_entr-probn)
            
            wn = wn + rate_n*np.sum(en.reshape(len(en),1)*an)
            

        
        self.pesos = wn
        
                
   
    def clasifica_prob(self,ej):
        
        if(self.normalizacion==True):
            an = (ej-self.mean)/self.std
        else:
            an = ej
        
        w_por_x = ((np.sum(self.pesos[:,1:]*an))+self.pesos[:,:1])
        
        prob = (1/(1+math.exp(-w_por_x)))
        
        return prob
        
    def clasifica(self,ej):
        
        if(self.normalizacion==True):
            an = (ej-self.mean)/self.std
        else:
            an = ej
        
        w_por_x = ((np.sum(self.pesos[:,1:]*an))+self.pesos[:,:1])
        
        prob = (1/(1+math.exp(-w_por_x)))
        
        if(prob > 0.5):
            res = self.clases[1]
        else:
            res = self.clases[0]
        
        return res


# ----------------------------------------------------------------
        
class Clasificador_RL_ML_St():
    
    def __init__(self,clases,normalizacion=False,
                 rate=0.1,rate_decay=False):
        
        self.clases = clases
        self.normalizacion = normalizacion
        self.rate = rate
        self.rate_decay = rate_decay
        self.pesos = None
        self.mean = None


    def entrena(self,entr,clas_entr,n_epochs,
                reiniciar_pesos=False,pesos_iniciales=None):
        
        #Normalizacion, se crean atributos de clase para reusar
        #los valores en la clasificacion
        if(self.normalizacion==True):
            #Comprobamos que haya una normalizacion anterior 
            if( (type(self.mean) is type(None)) or reiniciar_pesos):
                self.mean = entr.mean(axis=0)
                self.std = entr.std(axis=0)
            an = (entr-self.mean)/self.std
        else:
            an = entr
        
        #Condiciones de entrenamieto            
        if(reiniciar_pesos):
            wn = np.random.uniform(-1,1,(1,len(an[0])+1))   
        elif(pesos_iniciales):
            wn = pesos_iniciales
        elif(type(self.pesos) is type(None)):
            wn = np.random.uniform(-1,1,(1,len(an[0])+1))
        else:
            wn = self.pesos
        
        for n in range(0,n_epochs):
                        
            if(self.rate_decay):
                rate_n = (self.rate)*(1/(1+n))
            else:
                rate_n = self.rate
            
            ls_index = np.arange(0,len(an))
            np.random.shuffle(ls_index)
            
            for index in ls_index:
                
                w_por_x = ((np.sum(wn[:,1:]*an[index]))+wn[:,:1])
                prob = 1/(1+math.exp(-w_por_x))
                
                wn[:,:1] = wn[:,:1] + rate_n*(clas_entr[index] - prob)
                wn[:,1:] = wn[:,1:] + rate_n*an[index]*(clas_entr[index] - prob)
        
        self.pesos = wn
   
    def clasifica_prob(self,ej):
        
        if(self.normalizacion==True):
            an = (ej-self.mean)/self.std
        else:
            an = ej
        
        w_por_x = ((np.sum(self.pesos[:,1:]*an))+self.pesos[:,:1])
        
        prob = (1/(1+math.exp(-w_por_x)))
        
        return prob
        
    def clasifica(self,ej):
        
        if(self.normalizacion==True):
            an = (ej-self.mean)/self.std
        else:
            an = ej
        
        w_por_x = ((np.sum(self.pesos[:,1:]*an))+self.pesos[:,:1])
        
        prob = (1/(1+math.exp(-w_por_x)))
        
        if(prob > 0.5):
            res = self.clases[1]
        else:
            res = self.clases[0]
        
        return res

# ----------------------------------------------------------------

class Clasificador_RL_ML_MiniBatch():
    
    def __init__(self,clases,normalizacion=False,
                 rate=0.1,rate_decay=False,batch_tam=64):
        
        self.clases = clases
        self.normalizacion = normalizacion
        self.rate = rate
        self.rate_decay = rate_decay
        self.batch_tam
        self.pesos = None
        self.mean = None

    
    def entrena(self,entr,clas_entr,n_epochs,
                reiniciar_pesos=False,pesos_iniciales=None):
        
        #Normalizacion, se crean atributos de clase para reusar
        #los valores en la clasificacion
        if(self.normalizacion==True):
            #Comprobamos que haya una normalizacion anterior 
            if( (type(self.mean) is type(None)) or reiniciar_pesos):
                self.mean = entr.mean(axis=0)
                self.std = entr.std(axis=0)
            an = (entr-self.mean)/self.std
        else:
            an = entr
            
        #Condiciones de entrenamieto            
        if(reiniciar_pesos):
            wn = np.random.uniform(-1,1,(1,len(an[0])+1))   
        elif(pesos_iniciales):
            wn = pesos_iniciales
        elif(type(self.pesos) is type(None)):
            wn = np.random.uniform(-1,1,(1,len(an[0])+1))
        else:
            wn = self.pesos
        
        for n in range(0,n_epochs):
                        
            if(self.rate_decay):
                rate_n = (self.rate)*(1/(1+n))
            else:
                rate_n = self.rate
            
            ls_index = np.arange(0,len(an))
            np.random.shuffle(ls_index)
   
    def clasifica_prob(self,ej):
        
        if(self.normalizacion==True):
            an = (ej-self.mean)/self.std
        else:
            an = ej
        
        w_por_x = ((np.sum(self.pesos[:,1:]*an))+self.pesos[:,:1])
        
        prob = (1/(1+math.exp(-w_por_x)))
        
        return prob
        
    def clasifica(self,ej):
        
        if(self.normalizacion==True):
            an = (ej-self.mean)/self.std
        else:
            an = ej
        
        w_por_x = ((np.sum(self.pesos[:,1:]*an))+self.pesos[:,:1])
        
        prob = (1/(1+math.exp(-w_por_x)))
        
        if(prob > 0.5):
            res = self.clases[1]
        else:
            res = self.clases[0]
        
        return res


    

# --------------------------
# I.3. Curvas de aprendizaje
# --------------------------

# Se pide mostrar mediante gráficas la evolución del aprendizaje de los
# distintos algoritmos. En concreto, para cada clasificador usado con un
# conjunto de datos generado aleatoriamente con las funciones anteriores, las
# dos siguientes gráficas: 

# - Una gráfica que indique cómo evoluciona el porcentaje de errores que
#   comete el clasificador sobre el conjunto de entrenamiento, en cada epoch.    
# - Otra gráfica que indique cómo evoluciona el error cuadrático o la log
#   verosimilitud del clasificador (dependiendo de lo que se esté optimizando
#   en cada proceso de entrenamiento), en cada epoch.

# Para realizar gráficas, se recomiendo usar la biblioteca matplotlib de
# python: 

import matplotlib.pyplot as plt


# Lo que sigue es un ejemplo de uso, para realizar una gráfica sencilla a 
# partir de una lista "errores", que por ejemplo podría contener los sucesivos
# porcentajes de error que comete el clasificador, en los sucesivos epochs: 


# plt.plot(range(1,len(errores)+1),errores,marker='o')
# plt.xlabel('Epochs')
# plt.ylabel('Porcentaje de errores')
# plt.show()

# Basta con incluir un código similar a este en el fichero python, para que en
# la terminal de Ipython (en spyder, por ejemplo) se genere la correspondiente
# gráfica. 

# Se pide generar una serie de gráficas que permitan explicar el
# comportamiento de los algoritmos, con las distintas opciones, y con
# conjuntos separables y no separables. Comentar la interpretación de las
# distintas gráficas obtenidas. 

# NOTA: Para poder realizar las gráficas, debemos modificar los
# algoritmos de entrenamiento para que además de realizar el cálculo de los
# pesos, también calcule las listas con los sucesivos valores (de errores, de
# verosimilitud,etc.) que vamos obteniendo en cada epoch. Esta funcionalidad
# extra puede enlentecer algo el proceso de entrenamiento y es conveniente
# quitarla una vez se realice este apartado.
def graficaerroresporepoch(clasificador,clases,entr,
                           clas_entr,n_epochs,
                           rate=0.1,
                           rate_decay=False,
                           normalizacion=False,
                           batch=64):
    
    if (clasificador is Clasificador_RL_ML_MiniBatch):
        clas=Clasificador_RL_ML_MiniBatch(clases,normalizacion=normalizacion,
                                    rate=rate,rate_decay=rate_decay,batch_tam=batch)
        
    elif (clasificador is Clasificador_Perceptron):
        clas=Clasificador_Perceptron(clases,normalizacion=normalizacion,
                                    rate=rate,rate_decay=rate_decay)
        
    elif (clasificador is Clasificador_RL_ML_Batch):
        clas=Clasificador_RL_ML_Batch(clases,normalizacion=normalizacion,
                                    rate=rate,rate_decay=rate_decay)
        
    elif (clasificador is Clasificador_RL_ML_St):
        clas=Clasificador_RL_ML_St(clases,normalizacion=normalizacion,
                                    rate=rate,rate_decay=rate_decay)
    
    errores = []
    for n in range(1,n_epochs+1):
            if(rate_decay):
                clas.rate = rate*(1/(1+n))
            
            clas.entrena(clas_entr=clas_entr,entr=entr,n_epochs=1)
            error = 1 - rendimiento(clas,entr,clas_entr)
            errores.append(error)
        
    
    plt.plot(range(1,len(errores)+1),errores,marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Porcentaje de errores')
    plt.show()



# ==================================
# PARTE II: CLASIFICACIÓN MULTICLASE
# ==================================

# Se pide implementar un algoritmo de regresión logística para problemas de
# clasificación en los que hay más de dos clases. En concreto, usar la técnica
# de "One vs Rest" (Uno frente al Resto)

#  Como se ha visto, esta técnica construye un clasificador multiclase
#  a partir de clasificadores binarios que devuelven probabilidades
#  (como es el caso de la regresión logística). Para cada posible
#  valor de clasificación, se entrena un clasificador que estime cómo
#  de probable es pertemecer a esa clase, frente al resto. Este
#  conjunto de clasificadores binarios se usa para dar la
#  clasificación de un ejemplo nuevo, sin más que devolver la clase
#  para la que su correspondiente clasificador binario da una mayor
#  probabilidad.

#  En concreto, se pide implementar una clase python Clasificador_RL_OvR con
#  la siguiente estructura, y que implemente el entrenamiento y la
#  clasificación como se ha explicado. 


# class RegresionLogisticaOvR():

#    def __init__(self,class_clasif,clases,
#                  rate=0.1,rate_decay=False,batch_tam=64):

#          .....
         
#    def entrena(self,entr,clas_entr,n_epochs,
#                reiniciar_pesos=False,pesos_iniciales=None):

#         ......

#    def clasifica(self,ej):


#         ......
        

#  Excepto "class_clasif", los restantes parámetros de los métodos significan
#  lo mismo que en el apartado anterior, excepto que ahora "clases" puede ser
#  una lista con más de dos elementos. El parámetro class_clasif es el nombre
#  de la clase que implementa el clasificador binario a partir del cual se
#  forma el clasificador multiclase.   

#  Un ejemplo de sesión, con el problema del iris:

# ---------------------------------------------------------------
# In [28]: from iris import *

# In [29]: iris_clases=["Iris-setosa","Iris-virginica","Iris-versicolor"]

# Creamos el clasificador, a partir de RL binaria estocástico:
# In [30]: clas_rlml1=RegresionLogisticaOvR(Clasificador_RL_ML_St,iris_clases,rate_decay=True,rate=0.01)

# Lo entrenamos: 
# In [32]: clas_rlml1.entrena(iris_entr,iris_entr_clas,100)

# Clasificamos un par de ejemplos, comparándolo con su clase real:
# In [33]: clas_rlml1.clasifica(iris_entr[25]),iris_entr_clas[25]
# Out[33]: ('Iris-setosa', 'Iris-setosa')

# In [34]: clas_rlml1.clasifica(iris_entr[78]),iris_entr_clas[78]
# Out[34]: ('Iris-versicolor', 'Iris-versicolor')
# ----------------------------------------------------------------


class RegresionLogisticaOvR():
    
    def __init__(self,class_clasif,clases,
                 rate=0.1,rate_decay=False,batch_tam=64):
        
        self.clasificador = class_clasif
        self.clases = clases
        self.rate = rate
        self.rate_decay = rate_decay
        self.batch_tam = batch_tam
        self.clas_instanciadas = dict()
    
    def entrena(self,entr,clas_entr,n_epochs,
                 reiniciar_pesos=False,pesos_iniciales=None):
        
        clasifs = dict()
        
#----------Genero el diccionario con la clasificacion de cada clase------------        
        
        for c in range(0,len(self.clases)):
        
            copia = clas_entr.copy()
             
            for x in range(0,len(copia)):
                
                if(copia[x] == self.clases[c]):
                    copia[x] = 1
                else:
                    copia[x] = 0
            
            clasifs['clase'+str(c)] = copia
        
#----------------------Genero los pesos para cada clase------------------------
        for i in range(0,len(self.clases)):
            
            if(self.clasificador == Clasificador_RL_ML_MiniBatch):
                
                self.clas_instanciadas['clase'+str(i)] = self.clasificador([0,1],self.rate,
                                              self.rate_decay,self.batch_tam)
            else:
                self.clas_instanciadas['clase'+str(i)] = self.clasificador([0,1]
                                                    ,self.rate,self.rate_decay)
            
            print(clasifs['clase'+str(i)])
            #self.clas_instanciadas['clase'+str(i)].entrena(entr,clasifs['clase'+str(i)]
                                   #,n_epochs,reiniciar_pesos,pesos_iniciales)
                
                
     
    def clasifica(self,ej):
        
        prob = []
        
        for i in range(0,len(self.clases)):
            prob.append(self.clas_instanciadas['clase'+str(i)].clasifica(ej))
        
        prob_max = max(prob)
        
        return self.clases[prob.index(prob_max)]
            


# ===========================================
# PARTE III: APLICACIÓN DE LOS CLASIFICADORES
# ===========================================

# En este apartado se pide aplicar alguno de los clasificadores implementados
# en el apartado anterior, para analizar tres conjuntos de datos:
# - Votos
# - Dígitos escritos a mano
# - Cáncer de mama 

# -------------------------------------
# III.1 Implementación del rendimiento
# -------------------------------------

# Una vez que hemos entrenado un clasificador, podemos medir su rendimiento
# sobre un conjunto de ejemplos de los que se conoce su clasificación,
# mediante el porcentaje de ejemplos clasificados correctamente. Se ide
# definir una función rendimiento(clf,X,Y) que calcula el rendimiento de
# clasificador concreto clf, sobre un conjunto de datos X cuya clasificación
# conocida viene dada por Y. 
# NOTA: clf es un objeto de las clases definidas en
# los apartados anteriores, que además debe estar ya entrenado. 


# Por ejemplo (conectando con el ejemplo anterior):

# ---------------------------------------------------------
# In [36]: rendimiento(clas_rlml1,iris_entr,iris_entr_clas)
# Out[36]: 0.9666666666666667
# ---------------------------------------------------------


# También se pide implementar una función matriz_confusion(clf,X,Y) que
# imprime por pantalla la matriz de confusión de un clasificador clf, sobre un
# conjunto de datos X cuya clasificación conocida viene dada por Y. Se deja
# libre el formato para la impresión de la matriz de confusión. 

def rendimiento(clf,X,Y):
    return sum(clf.clasifica(x) == y for x,y in zip(X,Y))/len(Y)




def matriz_confusion(clf,X,Y):
    
    valores = np.zeros((len(clf.clases),len(clf.clases)))
    
    
    for i in range(0,len(clf.clases)):
        for j in range(0,len(clf.clases)):
            
            valores[i][j] = sum(1 for x,y in zip(X,Y) 
                if clf.clasifica(x) == clf.clases[i] and y == clf.clases[j])

    
    def representa(asig):
        
        def cadena_fila(i,asig):
            cadena = "| "
            for j in range(0,len(clf.clases)):
                accion = str(asig[i][j])
                cadena += accion
                cadena += " | "
            return cadena
        
        print("-"*7*len(clf.clases))
        for i in range(0,len(clf.clases)):
            print(cadena_fila(i,asig))
            print("-"*7*len(clf.clases))

    representa(valores)
# ----------------------------------
# III.2 Aplicando los clasificadores
# ----------------------------------

#  Obtener un clasificador para AL MENOS DOS de los siguientes problemas,
#  intentando que el rendimiento obtenido sobre un conjunto independiente de
#  ejemplos de prueba sea lo mejor posible:

#  - Predecir el partido de un congresista en función de lo que ha votado en
#    las sucesivas votaciones, a partir de los datos en el archivo votos.py que
#    se suministra.  

#  - Predecir, a partir de una serie de características que se observan sobre
#    una biopsia de un tumor en la mama, si se trata de un tumor maligno o
#    no. Para obtener los datos, se recomienda usar load_breast_cancer, del
#    módulo datasets de scikit learn. También se permite usar train_test_split
#    (del módulo model_selection) para partir los datos en entrenamiento,
#    validación y prueba. 

#  - Predecir el dígito que se ha escrito a mano y que se dispone en forma de
#    imagen pixelada, a partir de los datos que están en el archivo digidata.zip
#    que se suministra.  Cada imagen viene dada por 28x28 píxeles, y cada pixel
#    vendrá representado por un caracter "espacio en blanco" (pixel blanco) o
#    los caracteres "+" (borde del dígito) o "#" (interior del dígito). En
#    nuestro caso trataremos ambos como un pixel negro (es decir, no
#    distinguiremos entre el borde y el interior). En cada conjunto las imágenes
#    vienen todas seguidas en un fichero de texto, y las clasificaciones de cada
#    imagen (es decir, el número que representan) vienen en un fichero aparte,
#    en el mismo orden. Será necesario, por tanto, definir funciones python que
#    lean esos ficheros y obtengan los datos en el mismo formato python en el
#    que los necesitan los algoritmos. NOTA: este conjunto de datos puede
#    necesitar un mayor tiempo de ejecución que los anteriores. Si fuera
#    necesario, se admite tomar un conjunto de entrenamiento reducido.



#  Nótese que en cualquiera de los tres casos, consiste en encontrar el
#  clasificador adecuado, entrenado con los parámetros y opciones
#  adecuadas. El entrenamiento ha de realizarse sobre el conjunto de
#  entrenamiento, y el conjunto de validación se emplea para medir el
#  rendimiento obtenido con los distintas combinaciones de parámetros y
#  opciones con las que se experimente. Finalmente, una vez elegido la mejor
#  combinación de parámetros y opciones, se da el rendimiento final sobre el
#  conjunto de test. Es importante no usar el conjunto de test para decididir
#  sobre los parámetros, sino sólo para dar el rendimiento final.

#  En nuestro caso concreto, estas son las opciones y parámetros con los que
#  hay que experimentar: 

#  - En primer lugar, el tipo de clasificador usado (si es batch, mini-batch o
#    estocástico, si es basado en error cuadrático o en verosimilitud,...)
#  - n_epochs: el número de epochs realizados influye en el tiempo de
#    entrenamiento y evidentemente también en la calidad del clasificador
#    obtenido. Con un número bajo de epochs, no hay suficiente entrenamiento,
#    pero también hay que decir que un número excesivo de epochs puede
#    provocar un sobreajuste no deseado. 
#  - El valor de "rate" usado. 
#  - Si se usa "rate_decay" o no.
#  - Si se usa normalización o no. 

# Se pide describir brevemente el proceso de experimentación en cada uno de
# los casos, y finalmente dar el clasificador con el que se obtienen mejor
# rendimiento sobre el conjunto de test correspondiente.

# Por dar una referencia, se pueden obtener clasificadores para el problema de
# los votos y del cáncer de mama con un rendimiento sobre el test mayor al
# 90%, y para los dígitos un rendimiento superior al 80%.
