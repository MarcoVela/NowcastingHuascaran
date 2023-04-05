# Generación de Conjuntos de Datos

Para entrenar los modelos se utiliza un proceso de generación de Datasets.
Este proceso consiste en grillar los datos obtenidos con cierta resolución espacial y temporal. 
Posteriormente se binarizan y se ubican los clusters con mayor densidad de tormentas. 
Luego de identificar los clusters con tormentas se toman parches tridimensionales con dimensiones especificadas. 
Estos parches son escritos a disco en formato HDF5.

En este paquete se hace uso extensivo de [`ClimateBase.ClimArray`](https://juliaclimate.github.io/ClimateBase.jl/dev/#ClimArray:-the-core-data-structure).


