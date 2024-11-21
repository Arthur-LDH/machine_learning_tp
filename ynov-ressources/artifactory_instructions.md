## Instructions pour uploader votre modèle sur l'artifactory

#### 1. (optionnel) Editer le fichier proprietes.json

Le fichier proprietes.json contient les configurations/paramètres à afficher dans l'artifactory.

Vous pouvez par exemple y rajouter une adresse mail de contact.

#### 2. Télécharger utils-ads

Dans un virtualenv (celui qui vous voulez, vous pouvez même en créer un) :
```bash
pip install --trusted-host repository.pole-emploi.intra --index-url http://repository.pole-emploi.intra/artifactory/api/pypi/pypi-dev-virtual/simple utils-ads
```

#### 3. Compresser le dossier du modèle en zip
On ne garde que les fichiers nécessaires à la mise en prod.

Sur linux : `cd model_dir_path_identifier & zip -r ynov.zip configurations.json *.pkl *.hdf5`

(si besoin `sudo apt install zip unzip`)


#### 4. Uploader le modèle

/!\ A faire par vous : changer le numéro de version et éventuellement le cadre (model-datascience-prod-local si prod) et le nom du composant /!\

Pour avoir le "bon numéro de version", se référer à la dernière version disponible dans l'artifactory (http://repository.pole-emploi.intra/artifactory, anciennement http://artefact-repo.pole-emploi.intra/).

Attention, le nom du composant sera le nom affiché/versionné sur l'artifactory. Pour une tâche donnée, il doit rester le même pour gérer le versionning. De même pour le cadre.


Avec le virtual env d'activé (celui où utils-ads est installé) :
```bash
cd model_dir_path_identifier
interface_artifactory -t depot -x model-datascience-dev-local -c ynov -v 1.0.0 -f ynov.zip -p proprietes.json
```

INFOS :
- `-x` : Cadre à utiliser
- `-c` : Nom du composant
- `-v` : Numéro de version
- `-f` : Fichier à upload
- `-p` : Propriétés à afficher dans artifactory

Dans artifactory, ça donne donc l'arborescence suivante : x > c > v > f

/!\ Si vous avez une erreur du type : URI too long, il faudra diminuer le contenu du fichier proprietes.json /!\

#### Plus tard... Download du modèle

On considère ici que :
- un venv est activé avec utils-ads d'installé
- Le path `/path/to/folder/models` correspond au dossier où stocker les modèles

```bash
cd /path/to/folder/models # A changer par le chemin du dossier 'models'
interface_artifactory -t telecharge -x model-datascience-dev-local -c ynov -v 1.0.0  -f ynov.zip -s . -p proprietes.json
```

INFOS :
- `-x` : Cadre à utiliser
- `-c` : Nom du composant
- `-v` : Numéro de version
- `-f` : Fichier à download
- `-s` : Dossier où faire le téléchargement
- `-p` : Fichier de propriétes à créer