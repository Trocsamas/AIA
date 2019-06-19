Trabajo de Ampliación de Inteligencia Artificial 2018-19

En este trabajo implementaremos los algoritmos de clasificación lineal vistos en el Tema 5, y los aplicaremos a dos de los tres problemas de clasificación siguientes: adivinar el partido político (republicano o demócrata) de un congresista USA a partir de lo votado a lo largo de un año, reconocer un tumor maligno, y reconocer un dígito a partir de una imagen del mismo escrito a mano.

Para realizar el trabajo es necesario conocer los contenidos que se dan en el tema 5 de la asignatura.
Conjuntos de datos
Pediremos en primer lugar que se generen conjuntos de datos de manera aleatoria, que servirán para probar los primeros clasificadores (binarios), a medida que se implementen. Otro conjunto de datos que nos puede servir para probar los algoritmos es el de datos sobre clasificación de la flor de iris a partir de su longitud y anchura de sépalo y pétalo. Este es un conjunto de datos muy popular, que en nuestro caso sólo usaremos para probar alguna implementación.

Una vez implementados todos los clasificadores, pediremos que se apliquen a al menos dos de los siguientes tres conjuntos de datos, donde mediremos la capacidad de los modelos aprendidos. A su vez cada conjunto de datos se distribuye en tres partes: conjunto de entrenamiento, conjunto de validación y conjunto de prueba. El primero de ellos se usará para el aprendizaje, el segundo para ajustar determinados parámetros y opciones de los clasificadores que finalmente se aprendan, y el tercero para dar una medida final y "neutral" del rendimiento de los mismos.

Los datos que usaremos son:

    Datos sobre votos de cada uno de los 435 congresitas de Estados Unidos en 17 votaciones realizadas durante 1984. En votos.py están estos datos, en formato python. Este conjunto de datos está tomado de UCI Machine Learning Repository, donde se puede encontrar más información sobre el mismo.

    Un conjunto de imágenes (en formato texto), con una gran cantidad de dígitos (de 0 a 9) escritos a mano por diferentes personas, tomado de la base de datos MNIST. En digitdata.zip están todos los datos en formato comprimido.

    Datos de tumores en la mama, clasificados en benignos y malignos. Son los conocidos como "datos Wisconsin sobre el cáncer de mama". Para obtener los datos, se recomienda usar load_breast_cancer, del módulo datasets de scikit learn. También se permite usar train_test_split (del módulo model_selection) para partir los datos en entrenamiento, validación y prueba. Atención: son las únicas funciones de scikit learn que se permiten usar en este trabajo. 

Implementación de los clasificadores lineales
La implementación deberá realizarse en Python 3.0 o superior, siguiendo los apartadosque se enuncian en el fichero clasificadores_lineales.py, siguiendo las indicaciones que aparecen en el mismo. Este fichero (con el código completo) es lo único que hay que entregar.

Aunque el código se aplicará a los conjuntos de datos anteriores, deben implementarse de manera general, para que sea posible aplicarlo a cualquier otro problema de clasificación. Se recomienda el uso de numpy, pero no se permite usar scikit-learn (salvo lo descrito anteriormente para la carga de datos del cáncer de mama).

Aplicación de los clasificadores
Finalmente, se pide aplicar uno varios de los clasificadores implementados a al menos dos de los tres problemas mencionados anteriormente. Nótese que tanto en cada caso, los conjuntos de datos se parten en tres subconjuntos: entrenamiento, validación y test. El primero de ellos se usará en todos los casos para el entrenamiento de los modelos. Además, estos modelos tienen una serie de parámetros y opciones (por ejemplo, la tasa de aprendizaje, entre otros) cuyo mejor valor se decide tomando el que tenga mejor rendimiento sobre el conjunto de validación. Finalmente, una vez elegidos los parámetros y opciones del modelo, se toma el rendimiento sobre el conjunto de test, como medida "neutral" del rendimiento obtenido.

Realización y entrega del trabajo

    Los trabajos se pueden hacer en grupos de dos o individualmente, sin que esto influya en la forma en que se evaluarán. Por tanto, se recomienda hacerlo en grupo.

    IMPORTANTE: Los grupos de dos se han de comunicar previamente por correo electrónico al profesor, hasta el 7 de junio de 2019. Se entenderá que aquellos alumnos que no hayan comunicado su grupo antes de esa fecha van a realizar el trabajo individualmente (y esto se aplica a las tres convocatorias, incluida diciembre).

    Para resolución de dudas sobre este trabajo, contactar con el profesor José Luis Ruiz Reina (jruiz en el correo de la Universidad de Sevilla).

    El trabajo ha de ser entregado a través de la página web de la asignatura, entregando simplemente el archivo clasificadores_lineales.py, con ese nombre. Si se trata de un grupo, basta con que lo entregue uno de los autores. Muy importante: no ovidar incluir el nombre del autor o autores en la cabecera de los ficheros.

    Fecha tope de entrega del trabajo:
        Primera convocatoria: 1 de julio de 2019. 

Presentación del trabajo
Pasada la fecha tope de entrega, se anunciará convenientemente la hora y fecha para realizar la presentación del trabajo.

La presentación consistirá en mostrar el funcionamiento de la implementación realizada. En concreto, se realizará el entrenamiento de algunos modelos y se usarán los clasificadores aprendidos para clasificar ejemplos en los casos escogidos

Además, se han de explicar con claridad y soltura las partes del código implementado que se requieran. En el caso de que el trabajo haya sido realizado por un grupo de dos, cada miembro del código debe poder responder a cualquier parte del código, sin que sea distinguible la parte que ha realizado cada uno.

Plagios

La discusión y el intercambio de información de carácter general con los compañeros se permite (e incluso se recomienda), pero NO al nivel de código. Igualmente, no se permite incluir ningún código de terceros en el trabajo enviado (por ejemplo, no se permite usar código obtenido a través de la red).

Cualquier plagio o compartición de código que se detecte significará automáticamente la calificación de cero en la asignatura para TODOS los alumnos involucrados. Por tanto a estos alumnos NO se les conserva, ni para la actual ni para futuras convocatorias, ninguna nota que hubiesen obtenido hasta el momento. Sin perjuicio de las correspondientes medidas disciplinarias que se pudieran llevar a cabo.

Criteros de evaluación

    Rendimiento de los clasificadores.

    Interés y adecuación de las aplicaciones elegidas

    Eficiencia en la ejecución

    Documentación del código

    Claridad y buen estilo de programación.

    Presentación realizada.
