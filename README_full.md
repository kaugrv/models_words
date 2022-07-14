Read this in _[English](#)_


# Models and words


## Avant-propos

_wip_

## Dataset 

Notre étude repose sur un ensemble de 10 000 images semblables, appelé "dataset", généré à l'aide du logiciel Processing. Le code permettant de générer ces images est un script Java (.pde) disponible dans le [dossier lines](https://github.com/kaugrv/gantraining/tree/main/lines).

Chaque image de notre dataset est au format 128 par 128 pixels (`size(128,128)`) sur fond noir (`background(0)`). On génère ensuite 3 lignes d'épaisseur 5 pixels (`strokeWeight(5)`, fonction `line`). Une est verticale et deux sont horizontales. Leur position dans l'image est aléatoirement définie. Leurs couleurs[^1] sont également aléatoires (le choix suit la loi uniforme sur [0,3]) et chacune des 3 lignes peut être :
- blanche : `stroke(255)`
- jaune : `stroke(224,184,7)`
- rouge : `stroke(185,33,19)`
- ou bleue : `stroke(54,79,179)`


[^1]: Les couleurs choisies rendent hommage au travail de Piet Mondrian, en référence au _Mondrian Project_ de Frieder Nake.
  
<img 
    style="display: block; 
           text-align:center;
           width: 50%;"
    src="https://user-images.githubusercontent.com/103901906/176928811-a7b9bb3e-0644-400c-b5ca-548f69b5f41e.png" 
    alt="Extrait du dataset">
</img>

On peut calculer le nombre d'images qu'il est possible d'obtenir avec ce code. Chacune des 3 lignes a une abscisse (si verticale) ou une ordonnée (si horizontale) comprise entre 0 et 128, et possède une couleur choisie aléatoirement parmi les 4. Le nombre de lignes verticales et horizontales étant fixé, on a donc :
$$ \prod_{i=1}^{3} (4 \times 128) = (4 \times 128)^3 = 134217728 $$  

c'est-à-dire 134 millions de possibilités d'images générées. En générant 10 000 d'entre elles, on ne couvre que 0.007% de ces possibilités – autant dire que nous sommes certain de générer 10 000 images strictement différentes bien que similaires. Cela sera utile pour entraîner notre réseau neuronal. Nous chargerons ce dataset dans l'entraînement à l'aide de l'outil [loaders.py](https://github.com/kaugrv/gantraining/blob/main/utils/loaders.py).

## Architecture

Notre réseau est un GAN (_Generative Adversarial Network_) codé en Python, basé sur le travail et les explications de David Foster (voir Ressources). Nous utilisons aussi les librairies _TensorFlow_ et _Keras_.

Cette architecure est écrite dans le modèle [WGANGP.py](https://github.com/kaugrv/gantraining/blob/main/models/WGANGP.py). Dans notre version de l'entraînement, nous travaillons sur des images de 128x128 ; voici les paramètres que nous avons choisis pour les différentes _layers_ (discriminateur, générateur), leur _learning_rate_ et leurs matrices de convolution (_filters, kernel_ et _strides_) :

```
gan = WGANGP(input_dim=(IMAGE_SIZE, IMAGE_SIZE, 3), 
             critic_conv_filters=[128, 256, 512, 1024],
             critic_conv_kernel_size=[3,3,3,3], 
             critic_conv_strides=[2, 2, 2, 2], 
             critic_batch_norm_momentum=None, 
             critic_activation='leaky_relu', 
             critic_dropout_rate=None, 
             critic_learning_rate=0.0002, 
             generator_initial_dense_layer_size=(8, 8, 512), 
             generator_upsample=[1, 1, 1, 1], 
             generator_conv_filters=[512, 256, 128, 3], 
             generator_conv_kernel_size=[6,6,6,6], 
             generator_conv_strides=[2, 2, 2, 2], 
             generator_batch_norm_momentum=0.9, 
             generator_activation='leaky_relu', 
             generator_dropout_rate=None, 
             generator_learning_rate=0.0002, 
             optimiser='adam', 
             grad_weight=10, 
             z_dim=100, 
             batch_size=BATCH_SIZE
             )
 ```
Notre vecteur Z est de dimension 100. 


## Entraînement

La partie effective de l'entraînement est exécutable dans le Notebook [Training](https://github.com/kaugrv/gantraining/blob/main/python%20notebooks/Training%20128.ipynb). Il est exécutable depuis un environnement Jupyter (Anaconda,...) mais nous conseillons de le lancer directement depuis Google Colaboratory[^2].

Après avoir cloné ce dépôt, on travaille à sa racine /gantraining. Il est possible d'utiliser le dataset _lines_ ou bien d'en utiliser un nouveau (en l'uploadant directement dans l'environnement Colab, via une cellule ou en remplaçant le fichier output.zip). Au moins 10 000 images (format 128x128) sont conseillées, à placer dans un fichier _output.zip_, mais selon la complexité des images, davantage seront peut-être nécessaires.

Les premières cellules servent à charger le modèle et les bibliothèques, puis à charger le dataset.
Il est possible de configurer les paramètres de l'entraînement, notamment le numéro de l'entraînement que l'on va lancer (`RUN_ID`), ou le `BATCH_SIZE`, c'est-à-dire le nombre d'images présentées au réseau lors d'une itération. Par défaut, il est paramétré à 16, l'augmenter rendra le temps de calcul plus long, mais peut permettre d'améliorer la précision de l'entraînement. Après avoir chargé le dataset (celui-ci devrait bien renvoyer `Found ... images belonging to 1 classes.`) et paramétré le GAN (comme expliqué plus haut dans _Architecture_), d'autres paramètres sont modifiables :



- `EPOCHS` : le nombre de passages dans le réseau neuronal. Il est généralement de l'ordre de 1000 époques, mais selon la complexité des images à analyser il peut être inférieur ou supérieur. 
- `PRINT_EVERY_N_BATCHES` : au lieu de générer une image à tous les passages, on peut choisir la fréquence d'enregistrement. Les images générées se trouveront dans run/gan, dans le dossier correspondant au numéro de l'entraînement. Par exemple, pour `EPOCHS = 1201` et `PRINT_EVERY_N_BATCHES = 100`) :   

![Annotation 2022-07-01 184353](https://user-images.githubusercontent.com/103901906/176936405-c4fbce75-1ece-419f-a47e-5bd0e4547ef4.png)

- `rows` et `columns` : chaque sample généré ne contient pas forcément une seule image générée par le réseau mais plutôt une grille d'images, afin d'observer l'évolution de l'apprentissage de manière globale. Par exemple, pour `rows = 5`, `columns = 5`, et pour `rows = 2`, `columns = 2` respectivement :

  
<img 
    style="display: inline; 
           width: 20%;"
    src="https://user-images.githubusercontent.com/103901906/176937177-ef705ff3-603f-4fb1-9243-b15b1783b3a0.png" >
</img><img 
    style="display: inline; 
           width: 20%;"
    src="https://user-images.githubusercontent.com/103901906/176937301-ef2df143-63bb-4dad-8ce9-86d777c364f6.png" >
</img>


Il est possible d'observer l'évolution de la perte (_loss_) textuellement au cours des passages : 

``` 
...
452 (5, 1) [D loss: (-92.2)(R -71.9, F -58.8, G 3.8)] [G loss: 41.4]
453 (5, 1) [D loss: (-116.2)(R -179.5, F 4.4, G 5.9)] [G loss: 1.4]
454 (5, 1) [D loss: (-95.4)(R -113.7, F -34.5, G 5.3)] [G loss: 46.7]
455 (5, 1) [D loss: (-104.6)(R -114.9, F -40.9, G 5.1)] [G loss: 51.2]
456 (5, 1) [D loss: (-108.6)(R -120.9, F -28.5, G 4.1)] [G loss: 5.9]
457 (5, 1) [D loss: (-82.6)(R -87.7, F -38.1, G 4.3)] [G loss: 40.1]
458 (5, 1) [D loss: (-92.0)(R -173.4, F 16.8, G 6.5)] [G loss: -37.0]
459 (5, 1) [D loss: (-120.8)(R -118.1, F -61.4, G 5.9)] [G loss: 61.5]
460 (5, 1) [D loss: (-101.2)(R -94.1, F -60.1, G 5.3)] [G loss: 84.7]
461 (5, 1) [D loss: (-86.1)(R -167.7, F 22.9, G 5.9)] [G loss: -37.0]
462 (5, 1) [D loss: (-92.4)(R -127.9, F -16.1, G 5.2)] [G loss: 26.5]
463 (5, 1) [D loss: (-109.2)(R -116.9, F -36.9, G 4.5)] [G loss: 7.6]
464 (5, 1) [D loss: (-103.0)(R -105.2, F -52.5, G 5.5)] [G loss: 30.6]
465 (5, 1) [D loss: (-76.8)(R -90.0, F -34.5, G 4.8)] [G loss: 58.8]
466 (5, 1) [D loss: (-107.0)(R -145.8, F -23.6, G 6.2)] [G loss: 13.6]
467 (5, 1) [D loss: (-106.2)(R -191.6, F 15.8, G 7.0)] [G loss: -42.8]
...
```

Nous pouvons aussi générer un graphique montrant la convergence de l'entraînement (l'image de ce graphe sera enregistrée avec les _samples_ en tant que _converge.png_) :

![Converge](https://user-images.githubusercontent.com/103901906/176938153-df0678b7-bbd9-4e2a-b308-a00e3440b805.png)

Finalement des cellules permettent pour un environnement Google Colab de télécharger premièrement toutes les images, et enfin le fichier generator.h5, le fichier du générateur du réseau lié à cet entraînement, qui sera indispensable pour la partie suivante : l'_Inference_.

[^2]: Lien du Colab _Training_ : [www](https://colab.research.google.com/drive/1YspG4yXfPcr9Nixi8gEnk0vb0mssW9aQ?usp=sharing)

## Inference


## Ressources

Basé sur [davidADSP/GDL_code](https://github.com/davidADSP/GDL_code)  
Forked depuis [leogenot/DeepDrawing](https://github.com/leogenot/DeepDrawing)

* [_Generative Deep Learning: Teaching Machines to Paint, Write, Compose, and Play,_ David Foster](https://www.amazon.fr/Generative-Deep-Learning-Teaching-Machines/dp/1492041947)
* [Keras Documentation](https://keras.io/api/)
* [Tensorflow Documentation](https://www.tensorflow.org)
